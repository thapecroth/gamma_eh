"""Publish aggregate-only receipts from verified, completed corpus pilots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from corpus_import import exclusions, keys
from corpus_training import public_summary, read_rows, verify_completed
from pairs import hash_file

SOURCES = ("wi", "fce", "nucle", "lang8", "clang8", "c4", "github-typo", "wiked")
METRICS = ("true_positive_edits", "false_positive_edits", "missed_edits",
           "predicted_edits", "reference_edits", "edit_precision", "edit_recall", "edit_f0_5",
           "sentences", "clean_sentences", "clean_sentences_changed",
           "clean_sentence_false_positive_rate", "inference_failures")
TRAINING_DEPENDENCIES = ("train.py", "evaluate.py", "legacy_spelling.py", "edit_ops.py",
                         "data.py", "prepare_pairs.py", "pairs.py")


def metrics(value):
    return {key: value[key] for key in METRICS}


def verify_report_identity(report, plan, prepared, prepared_path, arm):
    if report["evaluation_hashes"] != plan["evaluation"]:
        raise ValueError("Actual evaluation differs from the frozen training plan")
    if any(report["code_sha256"].get(key) != plan["code"][key] for key in TRAINING_DEPENDENCIES):
        raise ValueError("Actual training dependency differs from the frozen training plan")
    if report["device"] != plan["schedule"]["device"] or report["seed"] != plan["schedule"]["seed"]:
        raise ValueError("Actual device or seed differs from the frozen training plan")
    if any(report["schedule"][key] != plan["schedule"][key] for key in ("epochs", "batch_size", "max_length")):
        raise ValueError("Actual training schedule differs from the frozen training plan")
    if plan["schedule"]["scorer"] != "errant" or any(value.get("scorer") != "errant" for value in
            (report["test"], report["diagnostic_unconstrained_test"],
             report["calibration"]["selected"], report["calibration"]["best_unconstrained"])):
        raise ValueError("This publication path requires actual ERRANT metrics")
    if report["dataset_manifest_sha256"] != hash_file(prepared_path):
        raise ValueError("Actual dataset manifest differs from the packaged training receipt")
    expected = plan["corpora"].values() if arm == "combined" else [plan["corpora"][arm]]
    if sorted((row["path"], row["sha256"]) for row in prepared["input_files"]) != sorted(
            (row["path"], row["sha256"]) for row in expected):
        raise ValueError("Prepared corpus inputs differ from the frozen training plan")


def summarize(runs, reserved, output):
    entries, inputs, plans = {}, {}, []
    common = None
    for run in runs:
        receipt = run / "plan.json"
        results = json.loads((run / "results.json").read_text())
        plan = json.loads(receipt.read_text())
        comparable = {"schedule": {key: value for key, value in plan["schedule"].items() if key != "device"},
                      "code": {key: value for key, value in plan["code"].items() if key != "corpus_training.py"},
                      "toolchain": plan["toolchain"], "evaluation": plan["evaluation"]}
        if results["plan"] != plan or (common is not None and common != comparable):
            raise ValueError("Receipts have different data-processing, training, evaluation, or schedule identities")
        common = comparable
        selected = plan.get("selected_arms", list(plan["corpora"]) + (["combined"] if plan["combined"] else []))
        if not set(results["arms"]) <= set(selected) or not set(plan["corpora"]) <= set(SOURCES):
            raise ValueError("Unknown training arms")
        for name, entry in plan["corpora"].items():
            if name in inputs and inputs[name] != entry:
                raise ValueError("Corpus identity changed between receipts")
            inputs[name] = entry
        for name, summary in results["arms"].items():
            if name in entries:
                raise ValueError("Ambiguous duplicate completed arm")
            entries[name] = (run, receipt, summary)
        plans.append({"sha256": hash_file(receipt), "schedule": plan["schedule"], "code_sha256": plan["code"],
                      "selected_arms": selected, "completed_arms": list(results["arms"])})
    expected = set(inputs) | ({"combined"} if any("combined" in value["selected_arms"] for value in plans) else set())
    if set(entries) != expected:
        raise ValueError("Incomplete source/combined result collection")
    reserved_paths, reserved_hashes = [], {}
    for item in reserved:
        name, separator, directory = item.partition("=")
        if not separator or name in reserved_hashes or not name.replace("-", "").isalnum():
            raise ValueError("Use unique reserved IDs as NAME=DIRECTORY")
        paths = [Path(directory) / (split + ".jsonl") for split in ("dev", "test")
                 if (Path(directory) / (split + ".jsonl")).is_file()]
        if not paths:
            raise ValueError("Reserved evaluation population missing")
        reserved_paths.extend(paths)
        reserved_hashes[name] = {path.stem: hash_file(path) for path in paths}
    if reserved_hashes.get("jfleg") != common["evaluation"] or "test" not in reserved_hashes.get("cweb", {}):
        raise ValueError("Publish the frozen JFLEG and reserved CWEB evaluation identities")
    heldout = exclusions(reserved_paths)
    sources = {}
    for name in SOURCES:
        if name not in inputs:
            sources[name] = {"status": "access-pending" if name in {"nucle", "lang8", "clang8"} else "not-run",
                             "trained_rows": None}
            continue
        entry = inputs[name]
        path = Path(entry["path"])
        if hash_file(path) != entry["sha256"] or hash_file(path.with_name("manifest.json")) != entry["manifest_sha256"]:
            raise ValueError("Imported corpus no longer matches the training plan")
        for row in read_rows(path):
            if (keys(row["source"]) | keys(row["target"])) & heldout:
                raise ValueError("Training source/reference overlaps a reserved evaluation population")
        manifest = json.loads(path.with_name("manifest.json").read_text())
        sources[name] = {"status": "trained", "input_sha256": entry["sha256"],
                         "import_manifest_sha256": entry["manifest_sha256"],
                         "source_archive_sha256": manifest["source_archive_sha256"],
                         "import_counts": {key: manifest["counts"].get(key, 0) for key in
                             ("scanned", "valid", "selected_original_pairs", "derived_clean_controls",
                              "conflicting_source", "duplicate", "rejected:heldout_overlap")},
                         "publication_allowed": manifest["publication_allowed"]}
    arms, actual_code = {}, None
    for name, (run, receipt, summary) in entries.items():
        verify_completed(run / name, summary, receipt)
        plan = json.loads(receipt.read_text())
        report = json.loads((run / name / "model/evaluation.json").read_text())
        evaluation = public_summary(report)
        prepared_path = run / name / "model/dataset-manifest.json"
        prepared = json.loads(prepared_path.read_text())
        verify_report_identity(report, plan, prepared, prepared_path, name)
        if actual_code is not None and actual_code != report["code_sha256"]:
            raise ValueError("Actual training dependencies changed between completed arms")
        actual_code = report["code_sha256"]
        if evaluation != summary["evaluation"] or summary["preparation"] != {
                key: prepared[key] for key in ("splits", "counts", "label_count", "publication_allowed", "licenses")}:
            raise ValueError("Aggregate arm metrics do not match verified model receipts")
        if evaluation["test"]["sentences"] != 747 or evaluation["calibration"]["selected"]["sentences"] != 754:
            raise ValueError("Publication requires complete JFLEG regression populations")
        training = evaluation["supported_tagged_coverage"]["train"]
        arms[name] = {"trained_rows": training["accepted"], "tagged_context_coverage": training,
                      "device": report["device"], "training_plan_sha256": hash_file(receipt),
                      "prepared_splits": summary["preparation"]["splits"],
                      "preparation_counts": summary["preparation"]["counts"],
                      "label_count": summary["preparation"]["label_count"],
                      "parameters": evaluation["parameters"], "schedule": evaluation["schedule"],
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "training_seconds": evaluation["training_seconds"],
                      "model_manifest_sha256": summary["model_manifest_sha256"],
                      "evaluation_sha256": summary["evaluation_sha256"],
                      "publication_allowed": evaluation["publication_allowed"],
                      "quality_gate_passed": evaluation["calibration"]["constraints_met"],
                      "disable_model_edits": evaluation["calibration"]["disable_model_edits"],
                      "gate_constraints": evaluation["calibration"]["constraints"],
                      "development_diagnostic": metrics(evaluation["calibration"]["best_unconstrained"]),
                      "test_guarded": metrics(evaluation["test"]),
                      "test_diagnostic": metrics(evaluation["diagnostic_unconstrained_test"]),
                      "epoch_losses": [{"epoch": value["epoch"], "loss": value["loss"]} for value in report["epochs"]],
                      "onnx_exports": {key: {"argmax_agreement": value["argmax_agreement"],
                                             "max_logit_difference": value["max_logit_difference"],
                                             "inference_failures": value["inference"]["inference_failures"]}
                                       for key, value in report["exports"].items()}}
        if name in sources:
            sources[name]["trained_rows"] = training["accepted"]
    result = {"schema": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Bounded local research pilots; independent complete JFLEG regression evaluation. Not official GLEU, corpus ranking, or browser reliability proof.",
              "sampling": "Seeded selection from bounded prefixes; GitHub restricted to selected documentation repositories. Unequal training sizes; one seed; no confidence intervals.",
              "training_plans": plans, "schedule": common["schedule"],
              "toolchain": common["toolchain"], "training_code_sha256": common["code"],
              "actual_training_dependency_sha256": actual_code,
              "summary_code_sha256": hash_file(Path(__file__)), "evaluation_hashes": common["evaluation"],
              "reserved_evaluation_hashes": reserved_hashes, "reserved_keys": len(heldout),
              "final_training_heldout_overlap": 0,
              "reference_policy": "ERRANT token spans; best reference per sentence by F0.5. Dev selects checkpoint/threshold; test never selects them.",
              "sources": sources, "arms": arms,
              "shipped_model_changed": False, "weights_published": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {"output": str(output), "sha256": hash_file(output), "trained_arms": len(arms)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--reserved", action="append", required=True, help="NAME=RESERVED_EVALUATION_DIRECTORY")
    p.add_argument("--output", type=Path, default=Path("docs/training-corpora-results.json"))
    args = p.parse_args()
    print(json.dumps(summarize(args.run, args.reserved, args.output)))
