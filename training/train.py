"""Bounded uncased-BERT training; decoded development selection and local ONNX export."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re
import shutil
import time
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizerFast

from evaluate import collect_proposals, evaluate_records, choose_calibration, choose_joint_calibration, load_evaluation, qualified
from pairs import evaluation_keys, hash_file, normalized
from rl_objective import anchored_loss, REWARDS, checkpoint_hashes, validate_coefficient, validate_checkpoint_labels

BASE = "google/bert_uncased_L-2_H-128_A-2"
REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
VERIFIED_BASE_LICENSES = {
    (BASE, REVISION): "Apache-2.0",
    ("google/bert_uncased_L-4_H-256_A-4", "387825ce42dbb39b87911cdf8e383ee3b25184f8"): "Apache-2.0",
}
THRESHOLDS = [.6, .7, .8, .85, .9, .95, .98, .995]


def load_split(path, tokenizer, label_to_id, max_length, allow_empty=False):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    accepted, rejected = [], {"unsupported_vocabulary": 0, "token_budget": 0}
    for row in rows:
        if any(tag not in label_to_id for tag in row["tags"]):
            rejected["unsupported_vocabulary"] += 1
        elif len(tokenizer(row["tokens"], is_split_into_words=True)["input_ids"]) > max_length:
            rejected["token_budget"] += 1
        else:
            accepted.append(row)
    if not accepted:
        if allow_empty: return None, {"population": len(rows), "accepted": 0, "rejected": rejected}
        raise ValueError(f"{path.name} has no supported rows within the context budget")
    encoded = tokenizer([r["tokens"] for r in accepted], is_split_into_words=True,
                        padding="max_length", max_length=max_length, return_tensors="pt")
    labels = torch.full_like(encoded["input_ids"], -100)
    for i, row in enumerate(accepted):
        previous = None
        for j, word_id in enumerate(encoded.word_ids(i)):
            if word_id is not None and word_id != previous:
                labels[i, j] = label_to_id[row["tags"][word_id]]
            previous = word_id
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"], labels), {
        "population": len(rows), "accepted": len(accepted), "rejected": rejected}


def metrics(summary, gold, threshold, disabled=False):
    # Only maximum confidence and ID are stored, rather than N x length x labels logits.
    confidence, prediction = summary
    prediction = torch.where((confidence >= threshold) & (not disabled), prediction, 0)
    valid = gold != -100
    edits = valid & (prediction != 0)
    actual = valid & (gold != 0)
    tp = ((prediction == gold) & edits).sum().item()
    fp = (edits & (prediction != gold)).sum().item()
    fn = (actual & (prediction != gold)).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    clean = ~actual.any(-1)
    clean_count = clean.sum().item()
    return {"threshold": threshold, "edit_precision": precision, "edit_recall": recall,
            "edit_f0_5": 1.25 * precision * recall / max(1e-12, .25 * precision + recall),
            "sentence_tag_accuracy": ((prediction == gold) | ~valid).all(-1).float().mean().item(),
            "clean_sentence_false_positive_rate": edits.any(-1)[clean].float().mean().item() if clean_count else None,
            "true_positive_edits": tp, "false_positive_edits": fp, "missed_edits": fn,
            "sentences": gold.shape[0], "scope": "Supported tagged subset; no deployment guards."}


def predict(model, loader, device):
    if loader is None: return None, None
    model.eval()
    confidence, prediction, gold = [], [], []
    with torch.inference_mode():
        for ids, mask, types, labels in loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           token_type_ids=types.to(device)).logits
            batch_confidence, batch_prediction = torch.softmax(logits, -1).max(-1)
            confidence.append(batch_confidence.cpu())
            prediction.append(batch_prediction.cpu())
            gold.append(labels)
    return (torch.cat(confidence), torch.cat(prediction)), torch.cat(gold)


def torch_infer(model, device):
    def infer(feed):
        with torch.inference_mode():
            return model(**{key: torch.from_numpy(value).to(device) for key, value in feed.items()}).logits.cpu().numpy()
    return infer


def calibration(rows, records, args):
    candidates = [evaluate_records(rows, records, threshold, scorer=args.scorer) for threshold in THRESHOLDS]
    no_edit = evaluate_records(rows, records, 1., disabled=True, scorer=args.scorer)
    result = choose_calibration(candidates, no_edit, args.min_precision, args.max_clean_fp, args.min_dev_edits)
    # Optional stricter family thresholds must improve the full development score
    # without relaxing global safety constraints; test never enters this selection.
    if args.calibrate_categories and not result["disable_model_edits"]:
        selected = result["selected"]
        thresholds = {}
        families = sorted({p["category"] for record in records for p in record["proposals"]})
        support = {family: sum(p["category"] == family for record in records for p in record["proposals"])
                   for family in families}
        for family in families:
            if support[family] < 20: continue
            for threshold in THRESHOLDS:
                if threshold <= selected["threshold"]: continue
                trial = {**thresholds, family: threshold}
                score = evaluate_records(rows, records, selected["threshold"], scorer=args.scorer, category_thresholds=trial)
                if (qualified(score, args.min_precision, args.max_clean_fp, args.min_dev_edits)
                    and score["edit_f0_5"] > selected["edit_f0_5"]):
                    thresholds, selected = trial, score
        result["category_support"] = support
        result["category_thresholds"] = thresholds
        result["selected"] = selected
    return result


def evaluation_populations(args):
    directory = args.evaluation_dir
    if directory is None and (args.data / "evaluation").exists(): directory = args.data / "evaluation"
    if directory is not None:
        rows = {split: load_evaluation(directory, split) for split in ["dev", "test"]}
    else:
        directory = args.data
        rows = {split: [dict(row, references=[row["target"]]) for row in
                       (json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines())]
                for split in ["dev", "test"]}
        if any(not values for values in rows.values()):
            raise ValueError("Provide a separate complete evaluation directory when tagged dev/test are empty")
    hashes = {split: hash_file(directory / f"{split}.jsonl") for split in rows}
    # Exclude exact source and every reference target from training before fitting.
    heldout = evaluation_keys([row for values in rows.values() for row in values])
    for line in (args.data / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if any(normalized(row[field]).casefold() in heldout for field in ["source", "target"]):
            raise ValueError("Training rows overlap an independent evaluation source/reference")
    return rows, hashes


def score_origins(rows, records, threshold, disabled, scorer, category_thresholds):
    """Separate original human utterances from added reference clean controls."""
    origins = sorted({row.get("origin", "unspecified") for row in rows})
    result = {}
    for origin in origins:
        indices = [index for index, row in enumerate(rows) if row.get("origin", "unspecified") == origin]
        result[origin] = evaluate_records([rows[index] for index in indices], [records[index] for index in indices],
                                         threshold, disabled, scorer=scorer, category_thresholds=category_thresholds)
    return result


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def main(args):
    if args.epochs < 1 or args.batch_size < 1 or args.inference_batch_size < 1 or args.min_dev_edits < 1 or not 4 <= args.max_length <= 512 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive epochs, batch size, learning rate, development support and max-length 4..512 are required")
    objective = getattr(args, "objective", "supervised")
    coefficient = getattr(args, "rl_coefficient", .05)
    validate_coefficient(coefficient)
    if objective not in {"supervised", "anchored-reinforce"}:
        raise ValueError("Unknown training objective")
    if objective == "supervised": coefficient = 0.
    initial_checkpoint = getattr(args, "initial_checkpoint", None)
    if not 0 <= args.min_precision <= 1 or not 0 <= args.max_clean_fp <= 1:
        raise ValueError("Calibration constraints must be between zero and one")
    for path in [args.output, args.checkpoint]:
        if path.exists() and any(path.iterdir()): raise ValueError("Training output/checkpoint already exists; choose fresh directories")
    if not Path(args.base_model).exists() and not re.fullmatch(r"[0-9a-f]{40}", args.base_revision):
        raise ValueError("Remote base models require an immutable 40-character revision")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable")
    print(f"device={device}", flush=True)
    labels = json.loads((args.data / "labels.json").read_text())
    dataset_manifest = json.loads((args.data / "manifest.json").read_text())
    license_files = dataset_manifest.get("license_files", {})
    for filename, expected in license_files.items():
        if Path(filename).name != filename or hash_file(args.data / filename) != expected:
            raise ValueError("Dataset license notice hash mismatch or unsafe filename")
    if not labels or labels[0] != "KEEP": raise ValueError("KEEP must be label zero")
    initial_hashes = validate_checkpoint_labels(initial_checkpoint, labels) if initial_checkpoint else None
    rows, evaluation_hashes = evaluation_populations(args)
    label_to_id = {label: i for i, label in enumerate(labels)}
    source = str(initial_checkpoint) if initial_checkpoint else args.base_model
    load_options = {"local_files_only": True} if initial_checkpoint else {
        "revision": args.base_revision, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **load_options)
    model = AutoModelForTokenClassification.from_pretrained(
        source, **load_options,
        num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label_to_id,
        attn_implementation="eager").to(device)
    if not isinstance(tokenizer, BertTokenizerFast) or not tokenizer.do_lower_case or model.config.model_type != "bert":
        raise ValueError("Browser runtime requires an uncased BERT WordPiece tokenizer and BERT model")
    normalizer = json.loads(tokenizer.backend_tokenizer.normalizer.__getstate__())
    if (normalizer.get("type") != "BertNormalizer" or not normalizer.get("lowercase")
            or not normalizer.get("handle_chinese_chars") or not normalizer.get("clean_text")
            or normalizer.get("strip_accents") not in {None, True}):
        raise ValueError("BERT tokenizer preprocessing does not match the bundled browser tokenizer")
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Context exceeds base model position embeddings")
    data, subset_coverage = {}, {}
    for split in ["train", "dev", "test"]:
        data[split], subset_coverage[split] = load_split(args.data / f"{split}.jsonl", tokenizer, label_to_id,
                                                       args.max_length, allow_empty=split != "train")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(values, batch_size=args.batch_size, shuffle=split == "train",
                                 generator=generator if split == "train" else None, num_workers=0) if values is not None else None
               for split, values in data.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    weights = torch.ones(len(labels), device=device)
    weights[0] = .3
    sampling_generator = torch.Generator(device=device).manual_seed(args.seed)
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    schema = dataset_manifest.get("edit_schema", 1)
    best_score = None
    history = []
    updates, examples_seen, checkpoint_epoch = 0, 0, None
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.
        supervised_loss, policy_loss = 0., 0.
        reward_sum, expected_reward_sum, sampled_tokens = 0., 0., 0
        for ids, mask, types, gold in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), token_type_ids=types.to(device)).logits
            loss, diagnostics = anchored_loss(logits, gold.to(device), weights, coefficient, sampling_generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item()
            supervised_loss += diagnostics["supervised_loss"]
            policy_loss += diagnostics["policy_gradient_loss"]
            if coefficient and diagnostics["valid_tokens"]:
                count = diagnostics["valid_tokens"]
                reward_sum += diagnostics["sampled_reward"] * count
                expected_reward_sum += diagnostics["expected_reward"] * count
                sampled_tokens += count
            updates += 1
            examples_seen += ids.shape[0]
        model.eval()
        records, inference = collect_proposals(rows["dev"], tokenizer, torch_infer(model, device), labels,
                                               args.max_length, schema, args.inference_batch_size)
        if inference["inference_failures"]:
            raise ValueError("Development inference failed; calibration cannot qualify a partially evaluated model")
        selected = calibration(rows["dev"], records, args)
        development = selected["selected"] if selected["constraints_met"] else selected["best_unconstrained"]
        score = (selected["constraints_met"], development["edit_f0_5"])
        history.append({"epoch": epoch + 1, "loss": total_loss / len(loaders["train"]),
                        "supervised_loss": supervised_loss / len(loaders["train"]),
                        "policy_gradient_loss": policy_loss / len(loaders["train"]),
                        "sampled_reward": reward_sum / sampled_tokens if sampled_tokens else None,
                        "expected_reward": expected_reward_sum / sampled_tokens if sampled_tokens else None,
                        "development": development, "constraints_met": selected["constraints_met"],
                        "inference": inference})
        print(json.dumps(history[-1]), flush=True)
        if best_score is None or score > best_score:
            best_score = score
            checkpoint_epoch = epoch + 1
            model.save_pretrained(args.checkpoint)
            tokenizer.save_pretrained(args.checkpoint)
            shutil.copyfile(args.data / "labels.json", args.checkpoint / "labels.json")
            for filename in license_files:
                shutil.copyfile(args.data / filename, args.checkpoint / filename)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint, attn_implementation="eager",
                                                           local_files_only=True).to(device).eval()
    dev_records, dev_inference = collect_proposals(rows["dev"], tokenizer, torch_infer(model, device), labels,
                                                   args.max_length, schema, args.inference_batch_size)
    if dev_inference["inference_failures"]:
        raise ValueError("Selected checkpoint failed development inference validation")
    calibrated = calibration(rows["dev"], dev_records, args)
    chosen = calibrated["selected"]
    threshold, disabled = chosen["threshold"], calibrated["disable_model_edits"]
    category_thresholds = calibrated.get("category_thresholds", {})
    test_records, test_inference = collect_proposals(rows["test"], tokenizer, torch_infer(model, device), labels,
                                                     args.max_length, schema, args.inference_batch_size)
    if test_inference["inference_failures"]:
        raise ValueError("Selected checkpoint failed complete test inference validation")
    complete_test = evaluate_records(rows["test"], test_records, threshold, disabled,
                                     scorer=args.scorer, category_thresholds=category_thresholds)
    diagnostic_test = evaluate_records(rows["test"], test_records,
                                      calibrated["best_unconstrained"]["threshold"], scorer=args.scorer)
    supported = {}
    for split in ["dev", "test"]:
        summary, gold = predict(model, loaders[split], device)
        supported[split] = metrics(summary, gold, threshold, disabled) if summary is not None else None
    report = {"base_model": args.base_model, "base_revision": args.base_revision, "seed": args.seed,
              "device": device, "parameters": sum(p.numel() for p in model.parameters()),
              "schedule": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                           "max_length": args.max_length}, "training_seconds": round(time.monotonic() - started, 2),
              "objective": {"name": objective, "rl_coefficient": coefficient, "keep_weight": .3,
                            "rewards": REWARDS if objective == "anchored-reinforce" else None,
                            "baseline": "Detached exact per-token expected reward" if coefficient else None,
                            "scope": "Training edit-tag supervision; independent decoded development/test evaluation."},
              "training_budget": {"optimizer_updates": updates, "examples_seen": examples_seen,
                                  "rows_per_epoch": len(data["train"]), "selected_checkpoint_epoch": checkpoint_epoch,
                                  "optimizer_state": "AdamW starts fresh; initial checkpoint supplies model weights only."},
              "initialization": {"checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
                                 "files": initial_hashes, "sampling_seed": args.seed},
              "epochs": history, "calibration": calibrated, "test": complete_test,
              "diagnostic_unconstrained_test": {**diagnostic_test,
                  "deployment_policy": "Diagnostic only: threshold chosen on dev, without enforcing precision/clean-text constraints."},
              "evaluation_hashes": evaluation_hashes, "supported_tagged_subset": supported,
              "supported_tagged_coverage": subset_coverage,
              "inference": {"dev": dev_inference, "test": test_inference},
              "scope": f"Complete decoded population, neural model with deployment guards and one pass; {args.scorer} token-span scoring, not combined browser quality or official JFLEG GLEU.",
              "dataset_scope": dataset_manifest.get("evaluation_scope", dataset_manifest.get("limitation")),
              "publication_allowed": dataset_manifest.get("publication_allowed", True)}
    report["toolchain"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                           "onnxruntime": ort.__version__, "transformers": version("transformers"),
                           "numpy": np.__version__}
    report["code_sha256"] = {name: hash_file(Path(__file__).with_name(name)) for name in
                             ["train.py", "rl_objective.py", "evaluate.py", "legacy_spelling.py", "edit_ops.py", "data.py", "prepare_pairs.py", "pairs.py"]}
    report["code_sha256"]["dictionary.generated.ts"] = hash_file(Path(__file__).resolve().parent.parent / "packages/engine/src/dictionary.generated.ts")
    report["code_sha256"]["spelling.ts"] = hash_file(Path(__file__).resolve().parent.parent / "packages/engine/src/spelling.ts")
    report["dataset_manifest_sha256"] = hash_file(args.data / "manifest.json")
    report["training_data_sha256"] = hash_file(args.data / "train.jsonl")
    report["labels_sha256"] = hash_file(args.data / "labels.json")
    report["checkpoint_files"] = checkpoint_hashes(args.checkpoint)
    base_license = VERIFIED_BASE_LICENSES.get((args.base_model, args.base_revision), "unverified")
    report["base_license"] = base_license
    report["publication_allowed"] = report["publication_allowed"] and base_license != "unverified"
    if initial_checkpoint:
        report["publication_allowed"] = False
        report["publication_restriction"] = "Local warm-start provenance is recorded but not independently qualified for weight publication."
    base_path = Path(args.base_model)
    if base_path.is_dir():
        report["local_base_files"] = {path.name: hash_file(path) for path in sorted(base_path.iterdir())
                                      if path.is_file() and path.name in {"config.json", "model.safetensors", "pytorch_model.bin", "vocab.txt"}}
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in license_files:
        shutil.copyfile(args.data / filename, args.output / filename)
    model = model.cpu().eval()
    dummy = tokenizer("She have a book.", return_tensors="pt")
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(ExportModel(model), tuple(dummy[key] for key in input_names), str(args.output / "model.onnx"),
                      input_names=input_names, output_names=["logits"],
                      dynamic_axes={key: {0: "batch", 1: "sequence"} for key in input_names + ["logits"]},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(args.output / "model.onnx"))
    quantize_dynamic(str(args.output / "model.onnx"), str(args.output / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    tokenizer.save_pretrained(args.output)
    shutil.copyfile(args.data / "labels.json", args.output / "labels.json")
    model.config.save_pretrained(args.output)
    shutil.copyfile(args.data / "manifest.json", args.output / "dataset-manifest.json")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    quantized = ort.InferenceSession(str(args.output / "model_quantized.onnx"), sess_options=options,
                                    providers=["CPUExecutionProvider"])
    quantized_dev, quantized_dev_inference = collect_proposals(
        rows["dev"], tokenizer, lambda feed: quantized.run(["logits"], feed)[0],
        labels, args.max_length, schema, args.inference_batch_size)
    if quantized_dev_inference["inference_failures"]:
        raise ValueError("INT8 development inference failed; deployment policy cannot be calibrated")
    floating = ort.InferenceSession(str(args.output / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    float_dev, float_dev_inference = collect_proposals(rows["dev"], tokenizer,
        lambda feed: floating.run(["logits"], feed)[0], labels, args.max_length, schema, args.inference_batch_size)
    if float_dev_inference["inference_failures"]: raise ValueError("FP32 development inference failed")
    quantized_calibration = calibration(rows["dev"], quantized_dev, args)
    fp32_candidates = [evaluate_records(rows["dev"], float_dev, value, scorer=args.scorer) for value in THRESHOLDS]
    deployed_calibration = choose_joint_calibration(quantized_calibration["candidates"], fp32_candidates,
        evaluate_records(rows["dev"], quantized_dev, 1., disabled=True, scorer=args.scorer), args.min_precision, args.max_clean_fp, args.min_dev_edits)
    # Family thresholds are retained only if they also qualify FP32 and improve
    # the shared policy. No test predictions participate in this decision.
    family = quantized_calibration.get("category_thresholds", {})
    if family and not deployed_calibration["disable_model_edits"]:
        selected = quantized_calibration["selected"]
        float_score = evaluate_records(rows["dev"], float_dev, selected["threshold"], scorer=args.scorer, category_thresholds=family)
        shared = deployed_calibration["selected"]
        if (qualified(selected, args.min_precision, args.max_clean_fp, args.min_dev_edits)
                and qualified(float_score, args.min_precision, args.max_clean_fp, args.min_dev_edits)
                and min(selected["edit_f0_5"], float_score["edit_f0_5"]) > shared["edit_f0_5"]):
            deployed_calibration.update({"selected": selected, "category_thresholds": family,
                                         "fp32_family_policy": float_score})
    threshold = deployed_calibration["selected"]["threshold"]
    disabled = deployed_calibration["disable_model_edits"]
    category_thresholds = deployed_calibration.get("category_thresholds", {})
    report["pytorch_calibration"] = report["calibration"]
    report["calibration"] = deployed_calibration
    report["quantized_development_inference"] = quantized_dev_inference
    report["fp32_development_inference"] = float_dev_inference
    report["development_by_origin"] = score_origins(rows["dev"], float_dev, threshold, disabled,
                                                    args.scorer, category_thresholds)
    report["pytorch_checkpoint_test"] = report["test"]
    report["pytorch_diagnostic_unconstrained_test"] = report["diagnostic_unconstrained_test"]
    report["pytorch_supported_tagged_subset"] = report.pop("supported_tagged_subset")
    exports = {}
    for filename in ["model.onnx", "model_quantized.onnx"]:
        session = quantized if filename == "model_quantized.onnx" else floating
        records, inference = collect_proposals(rows["test"], tokenizer, lambda feed: session.run(["logits"], feed)[0],
                                               labels, args.max_length, schema, args.inference_batch_size)
        if inference["inference_failures"]:
            raise ValueError("ONNX export failed complete-population inference validation")
        parity = {"max_logit_difference": 0., "argmax_matching": 0, "argmax_tokens": 0}
        if loaders["test"] is not None:
            for ids, mask, types, gold in loaders["test"]:
                feed = {key: value.numpy() for key, value in zip(input_names, [ids, mask, types])}
                onnx_logits = torch.from_numpy(session.run(["logits"], feed)[0])
                with torch.inference_mode(): pytorch_logits = model(input_ids=ids, attention_mask=mask, token_type_ids=types).logits
                valid = gold != -100
                parity["max_logit_difference"] = max(parity["max_logit_difference"], (onnx_logits - pytorch_logits).abs().max().item())
                parity["argmax_matching"] += (onnx_logits.argmax(-1) == pytorch_logits.argmax(-1))[valid].sum().item()
                parity["argmax_tokens"] += valid.sum().item()
        exports[filename] = {"metrics": evaluate_records(rows["test"], records, threshold, disabled,
                                                         scorer=args.scorer, category_thresholds=category_thresholds),
                             "by_origin": score_origins(rows["test"], records, threshold, disabled,
                                                        args.scorer, category_thresholds),
                             "diagnostic_metrics": evaluate_records(rows["test"], records,
                                deployed_calibration["best_unconstrained"]["threshold"], scorer=args.scorer),
                             "inference": inference, "max_logit_difference": parity["max_logit_difference"],
                             "argmax_agreement": parity["argmax_matching"] / parity["argmax_tokens"] if parity["argmax_tokens"] else None}
    report["exports"] = exports
    report["deployed_weights"] = "model.onnx"
    report["test"] = exports["model.onnx"]["metrics"]
    report["test_by_origin"] = exports["model.onnx"]["by_origin"]
    report["diagnostic_unconstrained_test"] = {**exports["model.onnx"]["diagnostic_metrics"],
        "deployment_policy": "Diagnostic only: shared threshold chosen on dev without safety constraints; never used to enable edits."}
    manifest = {"schema": 1, "name": f"gamma-eh-edit-v{schema}", "editSchema": schema, "maxPasses": 1,
                "model_license": base_license if report["publication_allowed"] else "training-or-base-terms-unverified",
                "publication_allowed": report["publication_allowed"], "base_model": args.base_model,
                "base_revision": args.base_revision, "parameters": report["parameters"],
                "confidenceThreshold": threshold, "disableModelEdits": disabled,
                "confidenceThresholds": category_thresholds, "maxSequenceLength": args.max_length,
                "experimental": True, "files": {}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "evaluation.json"}:
            manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.checkpoint / "receipt.json").write_text(json.dumps({
        "base_model": args.base_model, "base_revision": args.base_revision,
        "labels_sha256": report["labels_sha256"], "training_data_sha256": report["training_data_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"], "code_sha256": report["code_sha256"],
        "publication_allowed": report["publication_allowed"], "objective": report["objective"],
        "training_budget": report["training_budget"], "files": report["checkpoint_files"]}, indent=2) + "\n")
    print(json.dumps({"test": report["test"], "exports": exports}), flush=True)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, default=Path("data/generated"))
    result.add_argument("--output", type=Path, default=Path("models/browser"))
    result.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/tiny-edit"))
    result.add_argument("--epochs", type=int, default=8)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--inference-batch-size", type=int, default=16)
    result.add_argument("--max-length", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=5e-4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--objective", choices=["supervised", "anchored-reinforce"], default="supervised")
    result.add_argument("--rl-coefficient", type=float, default=.05,
                        help="Nonnegative finite REINFORCE weight; supervised weighted CE always remains active")
    result.add_argument("--initial-checkpoint", type=Path,
                        help="Local matching edit classifier warm start; experimental nonpublishable weights")
    result.add_argument("--base-model", default=BASE)
    result.add_argument("--base-revision", default=REVISION)
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--evaluation-dir", type=Path)
    result.add_argument("--min-precision", type=float, default=.95)
    result.add_argument("--max-clean-fp", type=float, default=.02)
    result.add_argument("--min-dev-edits", type=int, default=25,
                        help="Minimum decoded development predictions; operational evidence floor, not statistical certification")
    result.add_argument("--calibrate-categories", action="store_true")
    result.add_argument("--scorer", choices=["approximate", "errant"], default="approximate")
    result.add_argument("--device", choices=["cpu", "cuda"])
    return result


if __name__ == "__main__":
    main(parser().parse_args())
