"""Plan a bounded warmup and matched supervised/RL continuations; execute sequentially."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys

from pairs import evaluation_keys, hash_file, normalized
from rl_objective import checkpoint_hashes, validate_checkpoint_labels, validate_coefficient

ROOT = Path(__file__).resolve().parents[1]
BASE = "google/bert_uncased_L-2_H-128_A-2"
REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def plan(data, evaluation_dir, output, warmup_epochs=2, epochs=2, batch_size=64, max_length=96,
         learning_rate=.0002, seed=42, scorer="errant", device="cuda", rl_coefficient=.05,
         counterexamples=None, max_train_rows=4096, base_model=BASE, base_revision=REVISION):
    data, evaluation_dir, output = (Path(value).resolve() for value in (data, evaluation_dir, output))
    validate_coefficient(rl_coefficient)
    if not all(value >= 1 for value in (warmup_epochs, epochs, batch_size, max_train_rows)) or not 4 <= max_length <= 512:
        raise ValueError("Positive bounded schedules and context 4..512 are required")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or device not in {"cpu", "cuda"} or scorer not in {"approximate", "errant"}:
        raise ValueError("Invalid learning rate, device or scorer")
    if rl_coefficient == 0:
        raise ValueError("A matched RL comparison requires a positive RL coefficient")
    manifest = json.loads((data / "manifest.json").read_text())
    training = read_rows(data / "train.jsonl")
    if not 1 <= len(training) <= max_train_rows:
        raise ValueError("Training population exceeds the requested bounded row budget or is empty")
    labels = json.loads((data / "labels.json").read_text())
    if not labels or labels[0] != "KEEP" or len(set(labels)) != len(labels):
        raise ValueError("Expected unique ordered labels with KEEP at zero")
    files = [data / name for name in ("manifest.json", "labels.json", "train.jsonl", "dev.jsonl", "test.jsonl")]
    for filename, expected in manifest.get("license_files", {}).items():
        if Path(filename).name != filename or hash_file(data / filename) != expected:
            raise ValueError("Training license notice hash mismatch or unsafe filename")
        files.append(data / filename)
    for split in ("train", "dev", "test"):
        if hash_file(data / f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError("Frozen training dataset hash mismatch")
    if manifest.get("labels_sha256", hash_file(data / "labels.json")) != hash_file(data / "labels.json"):
        raise ValueError("Frozen label vocabulary hash mismatch")
    populations = {}
    for split in ("dev", "test"):
        path = evaluation_dir / f"{split}.jsonl"
        rows = read_rows(path)
        if not rows or not any(row.get("origin") == "ErAConD" for row in rows):
            raise ValueError("Independent human utterances are required in both evaluation splits")
        if any(row.get("origin") not in {"ErAConD", "ErAConD-human-reference-clean-control"}
               or not isinstance(row.get("source"), str) or not row.get("references")
               or not all(isinstance(ref, str) and ref.strip() for ref in row["references"]) for row in rows):
            raise ValueError("Evaluation requires all independent human references and explicit control provenance")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Evaluation IDs must be unique")
        expected = manifest.get("evaluation", {}).get(split, {}).get("sha256")
        if expected and hash_file(path) != expected:
            raise ValueError("Independent evaluation hash mismatch")
        populations[split] = rows
        files.append(path)
    if (evaluation_dir / "manifest.json").exists(): files.append(evaluation_dir / "manifest.json")
    heldout = evaluation_keys([row for rows in populations.values() for row in rows])
    if evaluation_keys(populations["dev"]) & evaluation_keys(populations["test"]):
        raise ValueError("Development and test sources/references overlap")
    evaluation_dialogs = {row.get("dialog_id") for rows in populations.values() for row in rows} - {None}
    if counterexamples:
        counterexamples = Path(counterexamples).resolve()
        diagnostic = json.loads(counterexamples.read_text())
        heldout |= evaluation_keys(diagnostic["cases"])
        files.append(counterexamples)
    for row in training:
        if any(normalized(value).casefold() in heldout for value in [row["source"], row["target"], *row.get("references", [])]):
            raise ValueError("Training overlaps an evaluation or diagnostic source/reference")
        if row.get("dialog_id") in evaluation_dialogs:
            raise ValueError("Training and evaluation dialogs overlap")
    code = [ROOT / "training" / name for name in ("compare_rl.py", "train.py", "rl_objective.py", "evaluate.py",
            "legacy_spelling.py", "edit_ops.py", "data.py", "prepare_pairs.py", "pairs.py")]
    code += [ROOT / "packages/engine/src" / name for name in ("dictionary.generated.ts", "spelling.ts")]
    if Path(base_model).is_dir():
        base_model = str(Path(base_model).resolve())
        files.extend(Path(base_model) / name for name in checkpoint_hashes(base_model))
    return {"schema": 1, "data": str(data), "evaluation_dir": str(evaluation_dir), "output": str(output),
            "input_hashes": {str(path): hash_file(path) for path in files},
            "code_hashes": {str(path): hash_file(path) for path in code},
            "warmup_epochs": warmup_epochs, "epochs": epochs, "batch_size": batch_size,
            "max_length": max_length, "learning_rate": learning_rate, "keep_weight": .3, "seed": seed,
            "scorer": scorer, "device": device, "rl_coefficient": rl_coefficient,
            "base_model": base_model, "base_revision": base_revision,
            "max_train_rows": max_train_rows, "raw_train_rows": len(training),
            "continuation_update_upper_bound": epochs * math.ceil(len(training) / batch_size),
            "counterexamples": str(counterexamples) if counterexamples else None,
            "evaluation_counts": {split: dict(Counter(row["origin"] for row in rows)) for split, rows in populations.items()},
            "arms": ["supervised", "anchored-reinforce"],
            "selection": "Development only; test and agent-authored counterexamples are confirmation diagnostics.",
            "resource_match": "Same retained warmup weights, labels, rows, seed, batch order, AdamW restart and optimizer update budget; RL sampling has its own RNG. Runtime/compute can differ.",
            "publication": "Warm-start weights remain experimental and nonpublishable; browser assets are not replaced."}


def verify_inputs(spec):
    for group in ("input_hashes", "code_hashes"):
        for name, expected in spec[group].items():
            if hash_file(Path(name)) != expected:
                raise ValueError("Frozen comparison input or code changed")


def command(spec, name, epochs, initial=None):
    directory = Path(spec["output"]) / name
    result = [sys.executable, str(ROOT / "training/train.py"), "--data", spec["data"],
              "--evaluation-dir", spec["evaluation_dir"], "--output", str(directory / "model"),
              "--checkpoint", str(directory / "checkpoint"), "--epochs", str(epochs),
              "--batch-size", str(spec["batch_size"]), "--max-length", str(spec["max_length"]),
              "--learning-rate", str(spec["learning_rate"]), "--seed", str(spec["seed"]),
              "--keep-weight", str(spec["keep_weight"]),
              "--base-model", spec["base_model"], "--base-revision", spec["base_revision"],
              "--device", spec["device"], "--scorer", spec["scorer"], "--local-files-only",
              "--objective", "supervised" if name == "warmup" else name,
              "--rl-coefficient", str(spec["rl_coefficient"])]
    if initial: result.extend(["--initial-checkpoint", str(initial)])
    return result


def completion_metadata(directory, create=False):
    """Freeze policy and metrics together after a successful stage, never overwrite."""
    directory = Path(directory)
    path = directory / "completion.json"
    current = {name: hash_file(directory / "model" / name) for name in ("manifest.json", "evaluation.json")}
    if create:
        with path.open("x") as stream:
            stream.write(json.dumps(current, indent=2) + "\n")
    elif not path.exists() or json.loads(path.read_text()) != current:
        raise ValueError("Completed stage policy or evaluation metadata changed or has no completion receipt")
    return current


def completed(spec, name, epochs, initial_hashes, newly_executed=False):
    directory = Path(spec["output"]) / name
    model, checkpoint = directory / "model", directory / "checkpoint"
    if not (model / "evaluation.json").exists(): return None
    if not newly_executed:
        completion_metadata(directory)
    report = json.loads((model / "evaluation.json").read_text())
    manifest = json.loads((model / "manifest.json").read_text())
    expected_schedule = {"epochs": epochs, "batch_size": spec["batch_size"],
                         "max_length": spec["max_length"], "learning_rate": spec["learning_rate"],
                         "keep_weight": spec["keep_weight"]}
    objective = "supervised" if name == "warmup" else name
    if (report["schedule"] != expected_schedule or report["seed"] != spec["seed"] or report["device"] != spec["device"]
            or report["objective"]["name"] != objective or report["objective"]["rl_coefficient"] != (
                spec["rl_coefficient"] if objective != "supervised" else 0.)
            or report["initialization"]["files"] != initial_hashes
            or report["dataset_manifest_sha256"] != spec["input_hashes"][str(Path(spec["data"]) / "manifest.json")]):
        raise ValueError("Existing comparison receipt differs from the frozen schedule/initialization")
    for source, expected in spec["code_hashes"].items():
        if Path(source).name not in {"compare_rl.py", "compare_judge_rl.py"} and report["code_sha256"].get(Path(source).name) != expected:
            raise ValueError("Existing trainer receipt used different frozen code")
    for filename, asset in manifest["files"].items():
        if Path(filename).name != filename or hash_file(model / filename) != asset["sha256"]:
            raise ValueError("Completed comparison model asset mismatch")
    if not {"model.onnx", "labels.json", "dataset-manifest.json"} <= manifest["files"].keys():
        raise ValueError("Completed comparison export is incomplete")
    for filename, expected in report["checkpoint_files"].items():
        if Path(filename).name != filename or hash_file(checkpoint / filename) != expected:
            raise ValueError("Completed warm-start checkpoint changed")
    if newly_executed:
        completion_metadata(directory, create=True)
    return report


def diagnose(spec, name, report):
    if not spec["counterexamples"]: return None
    from evaluate import collect_proposals, evaluate_records, onnx_predictor
    dataset = json.loads(Path(spec["counterexamples"]).read_text())
    rows = [dict(row, references=[row["target"]]) for row in dataset["cases"]]
    tokenizer, infer, labels, policy = onnx_predictor(Path(spec["output"]) / name / "model")
    records, inference = collect_proposals(rows, tokenizer, infer, labels, spec["max_length"], policy["editSchema"])
    if inference["inference_failures"]: raise ValueError("Counterexample inference failed")
    deployed = evaluate_records(rows, records, policy["confidenceThreshold"], policy["disableModelEdits"],
                               scorer=spec["scorer"], category_thresholds=policy.get("confidenceThresholds"))
    diagnostic = evaluate_records(rows, records, report["diagnostic_unconstrained_test"]["threshold"], scorer=spec["scorer"])
    return {"scope": dataset["scope"], "review_status": dataset["review_status"],
            "selection_use": False, "policy": deployed, "unconstrained": diagnostic, "inference": inference}


def run(spec):
    verify_inputs(spec)
    output = Path(spec["output"])
    if output.exists() and not (output / "plan.json").exists():
        raise ValueError("Existing comparison directory has no frozen plan; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != spec:
        raise ValueError("Comparison plan changed; choose a fresh directory")
    plan_path.write_text(json.dumps(spec, indent=2) + "\n")
    warmup = completed(spec, "warmup", spec["warmup_epochs"], None)
    if warmup is None:
        subprocess.run(command(spec, "warmup", spec["warmup_epochs"]), check=True)
        warmup = completed(spec, "warmup", spec["warmup_epochs"], None, newly_executed=True)
    initial = output / "warmup/checkpoint"
    initial_hashes = validate_checkpoint_labels(initial, json.loads((Path(spec["data"]) / "labels.json").read_text()))
    results = {}
    for name in spec["arms"]:
        verify_inputs(spec)
        if checkpoint_hashes(initial) != initial_hashes: raise ValueError("Shared initial checkpoint changed")
        report = completed(spec, name, spec["epochs"], initial_hashes)
        if report is None:
            subprocess.run(command(spec, name, spec["epochs"], initial), check=True)
            report = completed(spec, name, spec["epochs"], initial_hashes, newly_executed=True)
        results[name] = {"objective": report["objective"], "training_budget": report["training_budget"],
                         "training_seconds": report["training_seconds"], "publication_allowed": report["publication_allowed"],
                         "policy_qualified": report["calibration"]["constraints_met"],
                         "disable_model_edits": report["calibration"]["disable_model_edits"],
                         "development": report["calibration"]["selected"], "development_by_origin": report["development_by_origin"],
                         "test": report["test"], "test_by_origin": report["test_by_origin"],
                         "diagnostic_unconstrained_test": report["diagnostic_unconstrained_test"],
                         "counterexamples": diagnose(spec, name, report),
                         "evaluation_sha256": hash_file(output / name / "model/evaluation.json")}
    left, right = (results[name] for name in spec["arms"])
    for field in ("optimizer_updates", "examples_seen", "rows_per_epoch"):
        if left["training_budget"][field] != right["training_budget"][field]:
            raise ValueError("Matched continuation resource budgets differ")
    verify_inputs(spec)
    result = {"schema": 1, "plan_sha256": hash_file(plan_path), "initial_checkpoint_files": initial_hashes,
              "warmup_training_budget": warmup["training_budget"], "resource_match": spec["resource_match"],
              "scope": "Matched local neural experiment; no automatic winner, release, or browser-policy promotion.",
              "arms": results}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rules-rl"))
    parser.add_argument("--counterexamples", type=Path)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scorer", choices=("approximate", "errant"), default="errant")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rl-coefficient", type=float, default=.05)
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--base-model", default=BASE)
    parser.add_argument("--base-revision", default=REVISION)
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    execute = args.pop("execute")
    spec = plan(**args)
    print(json.dumps(run(spec) if execute else {"mode": "dry-run", **spec}, indent=2))
