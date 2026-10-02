"""Aggregate frozen receipts, qualify candidates on dev, and retain disabled policies."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from calibration import choose_threshold
from pairs import hash_file


def read(path):
    return json.loads(path.read_text())


def validate_browser(report, rows, population_hash):
    if report.get("passed") is not True or report.get("input_sha256") != population_hash or len(report.get("rows", [])) != len(rows):
        raise ValueError("Failed or incomplete browser population")
    for expected, actual in zip(rows, report["rows"]):
        if actual.get("id") != expected["id"] or actual.get("source") != expected["source"]:
            raise ValueError("Browser population order/source mismatch")


def engine_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        digest.update((path.name + '\0').encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def score_receipt(path, population, model, expected_engine):
    report = read(path)
    if report.get("passed") is False or not report.get("scores"):
        raise ValueError("Missing or failed score receipt")
    browser_path = path.with_name(path.name.replace('-score.json', '-browser.json'))
    browser = read(browser_path)
    rows = [json.loads(line) for line in population.read_text().splitlines()]
    validate_browser(browser, rows, hash_file(population))
    if report.get("browser_report_sha256") != hash_file(browser_path) or report.get("input_sha256") != hash_file(population):
        raise ValueError("Score receipt population/browser mismatch")
    for receipt in [report, browser]:
        if receipt["model"]["model_sha256"] != hash_file(model / "model.onnx") or receipt["engine_source_sha256"] != expected_engine:
            raise ValueError("Score receipt model/engine mismatch")
    conditions = [report["scores"]["rules"], *report["scores"]["model_only"], *report["scores"]["full_engine"]]
    if any(condition.get("sentences") != len(rows) for condition in conditions):
        raise ValueError("Incomplete scored population")
    return {"sha256": hash_file(path), **{key: value for key, value in report.items() if key != "model"},
            "model": {key: value for key, value in report["model"].items() if key != "policy"}}


def run(directory, data, evaluation, output, include_integration=False):
    profiles = []
    executed = read(directory / "plan.json")
    current_engine = engine_hash(directory / "study-engine")
    old_engine = engine_hash(directory / "baseline-source/packages/engine/src")
    diagnostic_engine = engine_hash(directory / "diagnostic-source/packages/engine/src")
    default_model = Path(__file__).resolve().parent.parent / "models/browser"
    cases_path = data / "extra/challenge.json"
    cases = read(cases_path)["cases"]
    # Validate every input before changing any candidate activation manifest.
    receipts = {}
    for profile in executed["profiles"]:
        home = directory / profile["name"]
        model = home / "model"
        manifest = read(model / "manifest.json")
        for name, asset in manifest["files"].items():
            if Path(name).name != name or hash_file(model / name) != asset["sha256"]:
                raise ValueError("Model asset hash mismatch")
        after = score_receipt(home / "final-dev-score.json", evaluation / "dev.jsonl", model, current_engine)
        before = score_receipt(home / "dev-score.json", evaluation / "dev.jsonl", model, diagnostic_engine)
        challenge = read(home / "challenge-browser.json")
        validate_browser(challenge, cases, hash_file(cases_path))
        if challenge["model"]["model_sha256"] != hash_file(model / "model.onnx") or challenge["engine_source_sha256"] != diagnostic_engine:
            raise ValueError("Challenge model/engine mismatch")
        receipts[profile["name"]] = (before, after, challenge)
    comparison = {}
    for name, filename, split, source in [
            ("development_before", "baseline-original-dev", "dev", old_engine),
            ("development_after", "baseline-final-dev", "dev", current_engine),
            ("test_before", "baseline-source-test", "test", old_engine),
            ("test_after", "current-test", "test", current_engine)]:
        comparison[name] = score_receipt(directory / f"{filename}-score.json", evaluation / f"{split}.jsonl", default_model, source)
    final_challenge = read(directory.parent / "tuning-final/synthetic-challenge.json")
    if not final_challenge["infrastructure_passed"] or final_challenge["corpus_sha256"] != hash_file(cases_path) or final_challenge["engine_source_sha256"] != current_engine or final_challenge["model_sha256"] != hash_file(default_model / "model.onnx"):
        raise ValueError("Final challenge execution or identity mismatch")
    integration = None
    if include_integration:
        integrated_engine = engine_hash(Path(__file__).resolve().parent.parent / "packages/engine/src")
        parent_engine = engine_hash(directory / "main-source/packages/engine/src")
        integration = {"parent_main_commit": "310f10a3a19d314d87d32503dce73a2540e0ae9e",
            "test_before": score_receipt(directory / "integrated-before-test-score.json", evaluation / "test.jsonl", default_model, parent_engine),
            "test_after": score_receipt(directory / "integrated-after-test-score.json", evaluation / "test.jsonl", default_model, integrated_engine)}
    no_edit = {"threshold": 1., "edit_precision": 1., "edit_f0_5": 0., "true_positive_edits": 0,
               "false_positive_edits": 0, "clean_sentence_false_positive_rate": 0.}
    for profile in executed["profiles"]:
        home = directory / profile["name"]
        model = home / "model"
        training = read(model / "evaluation.json")
        before, dev, challenge = receipts[profile["name"]]
        policy = choose_threshold(dev["scores"]["model_only"], no_edit)
        # This receipt belongs to the newly calibrated guarded browser engine.
        policy.update(engine_source_sha256=dev["engine_source_sha256"], development_score_sha256=dev["sha256"])
        (home / "natural-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
        manifest = read(model / "manifest.json")
        original_manifest_hash = dev["model"]["manifest_sha256"]
        manifest.update(disableModelEdits=policy["disable_model_edits"], confidenceThreshold=policy["selected"]["threshold"],
                        naturalQualification={"constraints_met": policy["constraints_met"], "score_sha256": dev["sha256"],
                                              "engine_source_sha256": dev["engine_source_sha256"]})
        (model / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        challenge_scores = []
        for threshold in challenge["thresholds"]:
            predictions = [next(p["full_engine"] for p in row["predictions"] if p["threshold"] == threshold) for row in challenge["rows"]]
            challenge_scores.append({"threshold": threshold, "sentences": len(cases),
                "fully_corrected": sum(prediction == row["target"] for prediction, row in zip(predictions, cases) if not row["clean"]),
                "clean_changed": sum(prediction != row["source"] for prediction, row in zip(predictions, cases) if row["clean"])})
        training_summary = {key: value for key, value in training.items() if key not in {"epochs", "calibration", "exports"}}
        training_summary.update(evaluation_sha256=hash_file(model / "evaluation.json"),
            selected_epoch=max(training["epochs"], key=lambda epoch: epoch["dev"]["edit_f0_5"])["epoch"],
            synthetic_development=training["calibration"]["selected"],
            exports={name: {key: value for key, value in receipt.items() if key != "metrics"} for name, receipt in training["exports"].items()})
        profiles.append({**profile, "training": training_summary, "model_sha256": manifest["files"]["model.onnx"]["sha256"],
            "diagnostic_manifest_sha256": original_manifest_hash, "retained_manifest_sha256": hash_file(model / "manifest.json"),
            "natural_dev_before_decoder_fix": before,
            "natural_dev_after_decoder_fix": dev, "natural_calibration": {key: value for key, value in policy.items() if key != "candidates"},
            "challenge_before_decoder_fix": {"browser_report_sha256": hash_file(home / "challenge-browser.json"),
                "engine_source_sha256": challenge["engine_source_sha256"], "scope": "Inspected regression audit; not used to promote a candidate", "scores": challenge_scores}})
    qualified = [p["name"] for p in profiles if p["natural_calibration"]["constraints_met"]]
    result = {"schema": 1, "scope": "Controlled original-CC0 tuning study and general decoder safeguards; not general grammar accuracy.",
        "baseline_commit": "e0dfd1a7cfde119827228e7a8f4fd2835ed697f2", "executed_plan": executed,
        "dataset_manifest": read(data / "manifest.json"), "dataset_manifest_sha256": hash_file(data / "manifest.json"),
        "base_training_versions": {name: version(name) for name in ["torch", "transformers", "onnx", "onnxruntime", "numpy"]},
        "reference_dataset_manifest": read(evaluation / "manifest.json"), "profiles": profiles,
        "promotion": {"qualified_profiles": qualified,
            "weights_changed": False, "decision": "Qualified candidates need a separate activation review." if qualified else "Retain existing checkpoint; all four new weights fail natural development qualification. Ship only the general decoder safeguard."},
        "baseline_decoder_comparison": {**comparison,
            "challenge_after": {key: value for key, value in final_challenge.items() if key != "rows"}},
        "limitations": ["Only one training seed; all synthetic families remain narrow.",
            "Natural scoring is custom best-sentence-reference ERRANT, not official JFLEG GLEU.",
            "JFLEG clean identity can match one reference while another permits an edit.",
            "Rules/dictionary false corrections remain in full-engine metrics.",
            "The challenge was inspected; clean training controls share some challenge structures.",
            "No new checkpoint qualifies; high precision with fewer than 25 edits is insufficient evidence."]}
    if integration:
        result["integration_check"] = integration
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/tuning-v1"))
    parser.add_argument("--data", type=Path, default=Path("data/generated/tuning-v1"))
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/model-tuning-results.json"))
    parser.add_argument("--include-integration", action="store_true")
    args = parser.parse_args()
    result = run(args.directory, args.data, args.evaluation, args.output, args.include_integration)
    print(json.dumps(result["promotion"], indent=2))
