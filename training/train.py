"""Fine-tune BERT-Tiny edit classifier; calibrate only on dev; export browser ONNX."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from calibration import choose_threshold

BASE = "google/bert_uncased_L-2_H-128_A-2"
REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"


def load_split(path, tokenizer, label_to_id, max_length):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError(f"{path.name} is empty. Provide separate reviewed development/test data for weakly supervised training.")
    encoded = tokenizer([r["tokens"] for r in rows], is_split_into_words=True,
                        padding="max_length", truncation=True, max_length=max_length,
                        return_tensors="pt")
    labels = torch.full_like(encoded["input_ids"], -100)
    for i, row in enumerate(rows):
        previous = None
        word_ids = encoded.word_ids(i)
        represented = set(w for w in word_ids if w is not None)
        if len(represented) != len(row["tokens"]):
            raise ValueError(f"Training sentence exceeds token budget: row {i}")
        for j, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous:
                labels[i, j] = label_to_id[row["tags"][word_id]]
            previous = word_id
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"], labels)


def metrics(logits, gold, threshold, disabled=False):
    probability = torch.softmax(logits, -1)
    confidence, prediction = probability.max(-1)
    prediction = torch.where(confidence >= threshold, prediction, 0)
    if disabled:
        prediction = torch.zeros_like(prediction)
    valid = gold != -100
    edits = valid & (prediction != 0)
    actual = valid & (gold != 0)
    tp = ((prediction == gold) & edits).sum().item()
    fp = (edits & (prediction != gold)).sum().item()
    fn = (actual & (prediction != gold)).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    clean = ~actual.any(-1)
    return {"threshold": threshold, "edit_precision": precision, "edit_recall": recall,
            "edit_f0_5": 1.25 * precision * recall / max(1e-12, .25 * precision + recall),
            "sentence_tag_accuracy": ((prediction == gold) | ~valid).all(-1).float().mean().item(),
            "clean_sentence_false_positive_rate": (edits.any(-1)[clean]).float().mean().item(),
            "true_positive_edits": tp, "false_positive_edits": fp, "missed_edits": fn,
            "sentences": gold.shape[0]}


def predict(model, loader, device):
    model.eval()
    predictions, gold = [], []
    with torch.inference_mode():
        for ids, mask, types, labels in loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           token_type_ids=types.to(device)).logits
            predictions.append(logits.cpu())
            gold.append(labels)
    return torch.cat(predictions), torch.cat(gold)


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def main(args):
    if args.epochs < 1 or args.batch_size < 1 or not 4 <= args.max_length <= 512 or args.min_dev_edits < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Training requires positive epochs, batch size, learning rate, and max-length >=4")
    if not math.isfinite(args.keep_weight) or args.keep_weight <= 0:
        raise ValueError("KEEP loss weight must be positive and finite")
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
    label_to_id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, use_fast=True, local_files_only=args.local_files_only)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id=label_to_id, attn_implementation="eager", local_files_only=args.local_files_only).to(device)
    data = {s: load_split(args.data / f"{s}.jsonl", tokenizer, label_to_id, args.max_length)
            for s in ["train", "dev", "test"]}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == "train",
                            generator=generator if s == "train" else None, num_workers=0)
               for s, d in data.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    weights = torch.ones(len(labels), device=device)
    weights[0] = args.keep_weight
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    best_f = -1
    history = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.
        for ids, mask, types, gold in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           token_type_ids=types.to(device)).logits
            loss = loss_fn(logits.reshape(-1, len(labels)), gold.to(device).reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item()
        dev_logits, dev_gold = predict(model, loaders["dev"], device)
        dev = metrics(dev_logits, dev_gold, .8)
        history.append({"epoch": epoch + 1, "loss": total_loss / len(loaders["train"]), "dev": dev})
        print(json.dumps(history[-1]), flush=True)
        if dev["edit_f0_5"] > best_f:
            best_f = dev["edit_f0_5"]
            model.save_pretrained(args.checkpoint)
            tokenizer.save_pretrained(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint, attn_implementation="eager").to(device)
    dev_logits, dev_gold = predict(model, loaders["dev"], device)
    thresholds = [metrics(dev_logits, dev_gold, t) for t in [.6, .7, .8, .85, .9, .95, .98]]
    calibration = choose_threshold(thresholds, metrics(dev_logits, dev_gold, 1., disabled=True), min_edits=args.min_dev_edits)
    chosen = calibration["selected"]
    disabled = calibration["disable_model_edits"]
    test_logits, test_gold = predict(model, loaders["test"], device)
    report = {"base_model": BASE, "base_revision": REVISION, "seed": args.seed,
              "device": device, "parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": round(time.monotonic() - started, 2), "epochs": history,
              "training_config": {"learning_rate": args.learning_rate, "keep_weight": args.keep_weight,
                                  "batch_size": args.batch_size, "max_length": args.max_length, "epochs": args.epochs},
              "calibration": calibration,
              "test": metrics(test_logits, test_gold, chosen["threshold"], disabled),
              "scope": dataset_manifest.get("evaluation_scope", "Synthetic in-distribution edit-tag benchmark; not real-world GEC quality."),
              "publication_allowed": dataset_manifest.get("publication_allowed", True)}
    args.output.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    wrapper = ExportModel(model)
    dummy = tokenizer("She have a book.", return_tensors="pt")
    inputs = tuple(dummy[k] for k in ["input_ids", "attention_mask", "token_type_ids"])
    dynamic_axes = {k: {0: "batch", 1: "sequence"} for k in ["input_ids", "attention_mask", "token_type_ids", "logits"]}
    torch.onnx.export(wrapper, inputs, str(args.output / "model.onnx"),
                      input_names=["input_ids", "attention_mask", "token_type_ids"],
                      output_names=["logits"], dynamic_axes=dynamic_axes,
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(args.output / "model.onnx"))
    quantize_dynamic(str(args.output / "model.onnx"), str(args.output / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    tokenizer.save_pretrained(args.output)
    shutil.copyfile(args.data / "labels.json", args.output / "labels.json")
    model.config.save_pretrained(args.output)
    shutil.copyfile(args.data / "manifest.json", args.output / "dataset-manifest.json")
    # Check both full-precision and quantized exports against the complete held-out split.
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    exports = {}
    for filename in ["model.onnx", "model_quantized.onnx"]:
        session = ort.InferenceSession(str(args.output / filename), sess_options=session_options,
                                       providers=["CPUExecutionProvider"])
        output = []
        for ids, mask, types, _ in loaders["test"]:
            feed = {k: value.numpy() for k, value in zip(
                ["input_ids", "attention_mask", "token_type_ids"], [ids, mask, types])}
            output.append(torch.from_numpy(session.run(None, feed)[0]))
        onnx_logits = torch.cat(output)
        exports[filename] = {"metrics": metrics(onnx_logits, test_gold, chosen["threshold"], disabled),
                             "max_logit_difference": (onnx_logits - test_logits).abs().max().item(),
                             "argmax_agreement": (onnx_logits.argmax(-1) == test_logits.argmax(-1))[test_gold != -100].float().mean().item()}
    report["exports"] = exports
    manifest = {"schema": 1, "name": args.name,
                "model_license": "Apache-2.0" if report["publication_allowed"] else "provider-terms-unverified",
                "publication_allowed": report["publication_allowed"],
                "base_model": BASE, "base_revision": REVISION, "parameters": report["parameters"],
                "confidenceThreshold": chosen["threshold"], "maxSequenceLength": args.max_length,
                "disableModelEdits": disabled,
                "experimental": True, "files": {}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "evaluation.json"}:
            manifest["files"][path.name] = {"bytes": path.stat().st_size,
                                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test": report["test"], "exports": exports}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/generated"))
    parser.add_argument("--output", type=Path, default=Path("models/browser"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/tiny-edit"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--keep-weight", type=float, default=.3)
    parser.add_argument("--min-dev-edits", type=int, default=25)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--name", default="gamma-eh-tiny-edit-v1")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
