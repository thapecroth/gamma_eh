"""Sequential per-source and mixed-source research training with complete evaluation."""
import argparse
from contextlib import redirect_stdout
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from corpus_import import canonical, exclusions
from pairs import hash_file, normalized
from prepare_pairs import prepare

ROOT = Path(__file__).resolve().parent.parent


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def prepare_arm(inputs, output, evaluation, max_labels=4096):
    heldout = exclusions([evaluation])
    for path in inputs:
        for row in read_rows(path):
            if row.get("evaluation_only"):
                raise ValueError("Evaluation-only input cannot be trained")
            if any(normalized(canonical(row[field])).casefold() in heldout
                   or normalized(row[field]).casefold() in heldout for field in ("source", "target")):
                raise ValueError("Training source/reference overlaps evaluation")
    with output.with_suffix(".prepare.log").open("w") as log, redirect_stdout(log):
        manifest = prepare(inputs, output, allow_weak_train=True, schema=2,
                           max_labels=max_labels, allow_research=True)
    # Imported original train splits stay train-only. External human references
    # calibrate and evaluate complete populations, regardless of tag coverage.
    for split in ("dev", "test"):
        path = evaluation / (split + ".jsonl")
        rows = sum(1 for _ in read_rows(path))
        if not rows:
            raise ValueError("Empty independent evaluation population")
        shutil.copyfile(path, output / "evaluation" / path.name)
        manifest["evaluation"][split] = {"rows": rows, "sha256": hash_file(path)}
    manifest.update({"evaluation_ready": True,
                     "evaluation_scope": "Complete independent natural-reference populations; source integration pilot, not official benchmark scores.",
                     "evaluation_license_separate_from_training": True,
                     "training_purpose": "local-research",
                     "corpus_input_manifests": [{"corpus": json.loads(path.with_name("manifest.json").read_text())["corpus"],
                                                  "sha256": hash_file(path.with_name("manifest.json"))} for path in inputs]})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def public_summary(report):
    # Training reports contain aggregates, never sentences or prediction text.
    return {key: report[key] for key in ("parameters", "schedule", "training_seconds", "calibration",
                                         "test", "diagnostic_unconstrained_test", "supported_tagged_coverage",
                                         "inference", "scope", "publication_allowed", "toolchain",
                                         "code_sha256", "dataset_manifest_sha256", "evaluation_hashes")}


def selected_arms(corpora, combined, requested):
    available = list(corpora) + (["combined"] if combined else [])
    if requested and (len(set(requested)) != len(requested) or not set(requested) <= set(available)):
        raise ValueError("Selected arms must be unique and present in the corpus/combined plan")
    return [name for name in available if not requested or name in requested]


def plan(args):
    corpora = {}
    for item in args.corpus:
        name, separator, value = item.partition("=")
        if not separator or name in corpora or not name.replace("-", "").isalnum() or name == "combined":
            raise ValueError("Use unique corpus IDs as NAME=DIRECTORY")
        directory = Path(value).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / "candidates.jsonl"
        if manifest["corpus"] != name or hash_file(path) != manifest["sha256"]:
            raise ValueError("Corpus identity/hash mismatch")
        corpora[name] = {"path": str(path), "sha256": hash_file(path),
                         "manifest_sha256": hash_file(directory / "manifest.json")}
    if args.epochs < 1 or args.batch_size < 1 or not 4 <= args.max_length <= 512:
        raise ValueError("Invalid training schedule")
    return {"corpora": corpora,
            "evaluation": {split: hash_file(args.evaluation_dir / (split + ".jsonl")) for split in ("dev", "test")},
            "schedule": {"epochs": args.epochs, "batch_size": args.batch_size,
                         "max_length": args.max_length, "seed": args.seed,
                         "scorer": args.scorer, "max_labels": args.max_labels, "device": args.device},
            "toolchain": {"python": sys.version.split()[0], **{name: version(name) for name in
                          ["torch", "transformers", "onnx", "onnxruntime", "numpy"] +
                          (["errant", "spacy", "en-core-web-sm"] if args.scorer == "errant" else [])}},
            "combined": args.combined,
            "selected_arms": selected_arms(corpora, args.combined, getattr(args, "only_arm", [])),
            "code": {name: hash_file(Path(__file__).with_name(name)) for name in
                     ("corpus_training.py", "corpus_import.py", "prepare_pairs.py", "train.py", "evaluate.py",
                      "edit_ops.py", "data.py", "pairs.py", "legacy_spelling.py")}}


def verify_completed(arm, summary, receipt):
    model = arm / "model"
    if summary["plan_sha256"] != hash_file(receipt):
        raise ValueError("Completed arm belongs to another plan")
    if summary["model_manifest_sha256"] != hash_file(model / "manifest.json"):
        raise ValueError("Completed model manifest changed")
    if summary["evaluation_sha256"] != hash_file(model / "evaluation.json"):
        raise ValueError("Completed evaluation changed")
    manifest = json.loads((model / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = model / name
        if Path(name).name != name or not path.is_file() or path.stat().st_size != expected["bytes"] or hash_file(path) != expected["sha256"]:
            raise ValueError("Completed model asset missing or changed")


def run(args):
    run_plan = plan(args)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **run_plan}, indent=2))
        return
    args.output.resolve().relative_to(ROOT / "artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / "plan.json"
    if receipt.exists() and json.loads(receipt.read_text()) != run_plan:
        raise ValueError("Run identity changed; choose a fresh output directory")
    receipt.write_text(json.dumps(run_plan, indent=2) + "\n")
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip()
    common = (ROOT / common).resolve()
    with (common / "gamma-training.lock").open("a") as lock:
        print("Waiting for repository training lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = {name: [Path(value["path"])] for name, value in run_plan["corpora"].items()}
        if args.combined:
            inputs["combined"] = [Path(value["path"]) for value in run_plan["corpora"].values()]
        inputs = {name: paths for name, paths in inputs.items() if name in run_plan["selected_arms"]}
        results = {}
        for name, paths in inputs.items():
            arm = args.output / name
            finished = arm / "result.json"
            if finished.exists():
                results[name] = json.loads(finished.read_text())
                verify_completed(arm, results[name], receipt)
                print(name + ": completed arm retained", flush=True)
                continue
            if arm.exists():
                raise ValueError("Incomplete arm exists; retain it and choose a fresh run directory")
            arm.mkdir()
            print("Preparing " + name, flush=True)
            manifest = prepare_arm(paths, arm / "prepared", args.evaluation_dir, args.max_labels)
            command = [sys.executable, str(Path(__file__).with_name("train.py")),
                       "--data", str(arm / "prepared"), "--evaluation-dir", str(args.evaluation_dir),
                       "--output", str(arm / "model"), "--checkpoint", str(arm / "checkpoint"),
                       "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                       "--max-length", str(args.max_length), "--seed", str(args.seed),
                       "--inference-batch-size", "16", "--device", args.device,
                       "--scorer", args.scorer, "--local-files-only"]
            started = time.monotonic()
            print("Training " + name, flush=True)
            with (arm / "training.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Training failed for {name}; inspect its local training.log")
            report = json.loads((arm / "model/evaluation.json").read_text())
            summary = {"corpus": name, "status": "trained", "plan_sha256": hash_file(receipt),
                       "elapsed_seconds": round(time.monotonic() - started, 2),
                       "model_manifest_sha256": hash_file(arm / "model/manifest.json"),
                       "evaluation_sha256": hash_file(arm / "model/evaluation.json"),
                       "preparation": {key: manifest[key] for key in ("splits", "counts", "label_count", "publication_allowed", "licenses")},
                       "evaluation": public_summary(report)}
            finished.write_text(json.dumps(summary, indent=2) + "\n")
            results[name] = summary
            (args.output / "results.json").write_text(json.dumps({"plan": run_plan, "arms": results}, indent=2) + "\n")
            print(json.dumps({"corpus": name, "trained_rows": report["supported_tagged_coverage"]["train"]["accepted"],
                              "f0_5": report["test"]["edit_f0_5"],
                              "disabled": report["calibration"]["disable_model_edits"]}), flush=True)
        (args.output / "results.json").write_text(json.dumps({"plan": run_plan, "arms": results}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", action="append", required=True, help="NAME=IMPORTED_DIRECTORY")
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-labels", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scorer", choices=("approximate", "errant"), default="errant")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--combined", action="store_true")
    p.add_argument("--only-arm", action="append", default=[], help="Train selected arms only; all corpus inputs remain pinned")
    p.add_argument("--execute", action="store_true")
    run(p.parse_args())
