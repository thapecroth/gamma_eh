"""Bounded matched supervised / calibrated LLM-judge RL experiment, sequential and local."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess

from compare_rl import (BASE, REVISION, ROOT, command as base_command, completed, diagnose,
                        plan as base_plan, read_rows, verify_inputs)
from judge import canonical, digest, endpoint, judge_spec, spec_hash, validate_calibration
from judge_objective import select_rows, subset_spec
from pairs import hash_file, normalized
from rl_objective import checkpoint_hashes, validate_checkpoint_labels


def plan(data, evaluation_dir, output, calibration, initial_checkpoint=None, fixture=ROOT / "data/judge-calibration.json",
         warmup_epochs=2, epochs=8, batch_size=64, max_length=96, learning_rate=.0002, seed=42,
         scorer="errant", device="cuda", rl_coefficient=.1, counterexamples=None, max_train_rows=4096,
         base_model=BASE, base_revision=REVISION, judge_model="gpt-6-luna", judge_provider="codex",
         judge_auth="subscription", judge_provider_header="X-CLIProxy-Provider", judge_auth_header="X-CLIProxy-Auth-Mode",
         rl_rows=128, rl_batch_size=16, temperature=1., judge_batch_size=32, max_requests=256, max_pairs=8192, timeout=60):
    spec = base_plan(data, evaluation_dir, output, warmup_epochs, epochs, batch_size, max_length, learning_rate,
                     seed, scorer, device, rl_coefficient, counterexamples, max_train_rows, base_model, base_revision)
    if not 1 <= rl_batch_size <= 16 or not 1 <= judge_batch_size <= 32 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid RL/judge batching or temperature")
    if not 1 <= max_requests <= 256 or not 1 <= max_pairs <= 8192 or not 1 <= timeout <= 60:
        raise ValueError("Judge request/pair/time limits exceed bounded budgets")
    data = Path(spec["data"])
    selected = select_rows(read_rows(data / "train.jsonl"), rl_rows, seed)
    worst_pairs = 2 * epochs * len(selected)
    worst_requests = epochs * sum(math.ceil(2 * len(selected[start:start + rl_batch_size]) / judge_batch_size)
                                  for start in range(0, len(selected), rl_batch_size))
    if worst_pairs > max_pairs or worst_requests > max_requests:
        raise ValueError("Worst-case sampled/baseline judgments exceed the finite budget")
    judge = judge_spec(judge_model, judge_provider, judge_auth, judge_provider_header, judge_auth_header)
    calibration, fixture = Path(calibration).resolve(), Path(fixture).resolve()
    calibration_receipt = validate_calibration(calibration, judge, fixture)
    for name, expected in calibration_receipt.get("code_sha256", {}).items():
        if name not in {"judge.py", "calibrate_judge.py"} or hash_file(ROOT / "training" / name) != expected:
            raise ValueError("Judge calibration executed different code")
    fixture_rows = json.loads(fixture.read_text())["cases"]
    reserved = {normalized(row[field]).casefold() for row in fixture_rows for field in ("source", "candidate")}
    for row in read_rows(data / "train.jsonl"):
        if any(normalized(value).casefold() in reserved for value in [row["source"], row["target"], *row.get("references", [])]):
            raise ValueError("Judge calibration source/candidate overlaps training")
    spec["input_hashes"].update({str(path): hash_file(path) for path in (calibration, fixture)})
    spec["input_hashes"][str(ROOT / "training/calibrate_judge.py")] = hash_file(ROOT / "training/calibrate_judge.py")
    for name in ("compare_judge_rl.py", "judge.py", "judge_objective.py"):
        path = ROOT / "training" / name
        spec["code_hashes"][str(path)] = hash_file(path)
    subset = subset_spec(selected)
    spec.update({"arms": ["supervised", "llm-judge-reinforce"], "judge_spec": judge,
                 "judge_spec_sha256": spec_hash(judge), "judge_subset": subset,
                 "judge_subset_sha256": digest(canonical(subset)), "judge_rl_rows": rl_rows,
                 "judge_rl_batch_size": rl_batch_size, "judge_temperature": temperature,
                 "judge_batch_size": judge_batch_size, "judge_max_requests": max_requests,
                 "judge_max_pairs": max_pairs, "judge_timeout": timeout,
                 "judge_endpoint": endpoint(os.environ.get("GAMMA_JUDGE_BASE_URL", "http://127.0.0.1:8317/v1")),
                 "judge_calibration": str(calibration), "judge_fixture": str(fixture),
                 "worst_case_judge_budget": {"pairs": worst_pairs, "requests": worst_requests},
                 "initial_checkpoint": None, "initial_checkpoint_files": None,
                 "resource_match": "Full identical CE data/order/seed and optimizer updates; RL adds bounded synthetic-only eval forwards and live judge work. Time/compute are measured, not equal.",
                 "selection": "Independent human development only; test, calibration fixture and counterexamples do not select model checkpoints/policy."})
    if initial_checkpoint:
        initial_checkpoint = Path(initial_checkpoint).resolve()
        hashes = validate_checkpoint_labels(initial_checkpoint, json.loads((data / "labels.json").read_text()))
        receipt = json.loads((initial_checkpoint / "receipt.json").read_text())
        if (receipt.get("objective", {}).get("name") != "supervised"
                or receipt.get("training_data_sha256") != hash_file(data / "train.jsonl")
                or receipt.get("labels_sha256") != hash_file(data / "labels.json")
                or receipt.get("dataset_manifest_sha256") != hash_file(data / "manifest.json")):
            raise ValueError("Existing supervised warmup checkpoint data/labels/provenance differ")
        for filename, expected in receipt["files"].items():
            if Path(filename).name != filename or hashes.get(filename) != expected:
                raise ValueError("Existing warmup checkpoint artifact hash mismatch")
        spec["initial_checkpoint"], spec["initial_checkpoint_files"] = str(initial_checkpoint), hashes
        spec["input_hashes"].update({str(initial_checkpoint / name): value for name, value in hashes.items()})
        spec["warmup_receipt"] = receipt
    return spec


def command(spec, name, initial):
    result = base_command(spec, name, spec["epochs"], initial)
    if name == "llm-judge-reinforce":
        judge = spec["judge_spec"]
        for flag, value in {
            "--judge-model": judge["model"], "--judge-provider": judge["expected_provider"],
            "--judge-auth": judge["expected_auth"], "--judge-provider-header": judge["provider_header"],
            "--judge-auth-header": judge["auth_header"], "--judge-calibration": spec["judge_calibration"],
            "--judge-calibration-fixture": spec["judge_fixture"], "--judge-subset-sha256": spec["judge_subset_sha256"],
            "--judge-cache": str(Path(spec["output"]) / name / "judge-cache"), "--judge-rl-rows": spec["judge_rl_rows"],
            "--judge-rl-batch-size": spec["judge_rl_batch_size"], "--judge-temperature": spec["judge_temperature"],
            "--judge-max-requests": spec["judge_max_requests"], "--judge-max-pairs": spec["judge_max_pairs"],
            "--judge-batch-size": spec["judge_batch_size"], "--judge-timeout": spec["judge_timeout"],
        }.items(): result.extend([flag, str(value)])
    return result


def verify_judge_report(spec, report):
    config = report["objective"]["judge"]
    if (config["spec_sha256"] != spec["judge_spec_sha256"] or config["subset_sha256"] != spec["judge_subset_sha256"]
            or config["temperature"] != spec["judge_temperature"] or config["rl_batch_size"] != spec["judge_rl_batch_size"]
            or config["calibration_sha256"] != spec["input_hashes"][spec["judge_calibration"]]):
        raise ValueError("Completed judge objective differs from the frozen plan")
    cache = Path(spec["output"]) / "llm-judge-reinforce/judge-cache"
    for name, expected in report["judge"]["ledgers"].items():
        if Path(name).name != name or hash_file(cache / name) != expected:
            raise ValueError("Completed judge cache/receipt changed")


def comparison_identity(result):
    """Exclude only fresh diagnostic wall time from immutable result checks."""
    identity = deepcopy(result)
    for arm in identity["arms"].values():
        if arm.get("counterexamples") is not None:
            arm["counterexamples"]["inference"].pop("inference_seconds", None)
    return identity


def save_comparison(path, result):
    if path.exists():
        previous = json.loads(path.read_text())
        if comparison_identity(previous) != comparison_identity(result):
            raise ValueError("Completed comparison metadata changed")
        return previous
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(spec):
    verify_inputs(spec)
    if endpoint(os.environ.get("GAMMA_JUDGE_BASE_URL", "http://127.0.0.1:8317/v1")) != spec["judge_endpoint"]:
        raise ValueError("Judge endpoint changed after planning")
    validate_calibration(spec["judge_calibration"], spec["judge_spec"], spec["judge_fixture"])
    output = Path(spec["output"])
    path = output / "plan.json"
    if output.exists() and not path.exists(): raise ValueError("Choose fresh output or an exact frozen resumable plan")
    output.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != spec: raise ValueError("Frozen judge experiment plan changed")
    if not path.exists(): path.write_text(json.dumps(spec, indent=2) + "\n")
    if spec["initial_checkpoint"]:
        initial = Path(spec["initial_checkpoint"])
        warmup_budget = spec["warmup_receipt"]["training_budget"]
    else:
        warmup = completed(spec, "warmup", spec["warmup_epochs"], None)
        if warmup is None:
            subprocess.run(base_command(spec, "warmup", spec["warmup_epochs"]), check=True)
            warmup = completed(spec, "warmup", spec["warmup_epochs"], None, newly_executed=True)
        initial, warmup_budget = output / "warmup/checkpoint", warmup["training_budget"]
    initial_hashes = validate_checkpoint_labels(initial, json.loads((Path(spec["data"]) / "labels.json").read_text()))
    if spec["initial_checkpoint_files"] and initial_hashes != spec["initial_checkpoint_files"]:
        raise ValueError("Shared initial checkpoint changed after planning")
    results = {}
    for name in spec["arms"]:
        verify_inputs(spec)
        if checkpoint_hashes(initial) != initial_hashes: raise ValueError("Shared initial checkpoint changed")
        report = completed(spec, name, spec["epochs"], initial_hashes)
        if report is None:
            subprocess.run(command(spec, name, initial), check=True)
            report = completed(spec, name, spec["epochs"], initial_hashes, newly_executed=True)
        if name == "llm-judge-reinforce": verify_judge_report(spec, report)
        results[name] = {"objective": report["objective"], "training_budget": report["training_budget"],
                         "training_seconds": report["training_seconds"], "publication_allowed": report["publication_allowed"],
                         "policy_qualified": report["calibration"]["constraints_met"],
                         "disable_model_edits": report["calibration"]["disable_model_edits"],
                         "development": report["calibration"]["selected"], "development_by_origin": report["development_by_origin"],
                         "test": report["test"], "test_by_origin": report["test_by_origin"],
                         "diagnostic_unconstrained_test": report["diagnostic_unconstrained_test"],
                         "judge_epochs": [epoch.get("judge") for epoch in report["epochs"]],
                         "judge": report.get("judge"), "counterexamples": diagnose(spec, name, report),
                         "evaluation_sha256": hash_file(output / name / "model/evaluation.json")}
    left, right = (results[name] for name in spec["arms"])
    for field in ("optimizer_updates", "examples_seen", "rows_per_epoch"):
        if left["training_budget"][field] != right["training_budget"][field]:
            raise ValueError("Matched CE continuation resource budgets differ")
    verify_inputs(spec)
    result = {"schema": 1, "plan_sha256": hash_file(path), "initial_checkpoint_files": initial_hashes,
              "warmup_training_budget": warmup_budget, "resource_match": spec["resource_match"],
              "judge_spec_sha256": spec["judge_spec_sha256"], "arms": results,
              "scope": "Matched local experiment; judge rewards are not independent accuracy, and no model is promoted automatically."}
    return save_comparison(output / "comparison.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/judge-rl"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--fixture", type=Path, default=ROOT / "data/judge-calibration.json")
    parser.add_argument("--counterexamples", type=Path)
    for name, default in (("warmup-epochs", 2), ("epochs", 8), ("batch-size", 64), ("max-length", 96),
                          ("seed", 42), ("max-train-rows", 4096), ("rl-rows", 128), ("rl-batch-size", 16),
                          ("judge-batch-size", 32), ("max-requests", 256), ("max-pairs", 8192), ("timeout", 60)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=.0002)
    parser.add_argument("--rl-coefficient", type=float, default=.1)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--scorer", choices=("approximate", "errant"), default="errant")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for name, default in (("base-model", BASE), ("base-revision", REVISION), ("judge-model", "gpt-6-luna"),
                          ("judge-provider", "codex"), ("judge-auth", "subscription"),
                          ("judge-provider-header", "X-CLIProxy-Provider"), ("judge-auth-header", "X-CLIProxy-Auth-Mode")):
        parser.add_argument("--" + name, default=default)
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    execute = args.pop("execute")
    spec = plan(**args)
    print(json.dumps(run(spec) if execute else {"mode": "dry-run", **spec}, indent=2))
