"""Prevent publication under a different dataset, implementation or scorer identity."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from pairs import hash_file
from summarize_corpora import TRAINING_DEPENDENCIES, verify_report_identity


def receipt(tmp_path):
    source = {"path": "source.jsonl", "sha256": "source-hash"}
    prepared = {"input_files": [source]}
    path = tmp_path / "dataset-manifest.json"
    path.write_text(json.dumps(prepared))
    schedule = {"device": "cpu", "seed": 42, "epochs": 4, "batch_size": 32,
                "max_length": 128, "scorer": "errant"}
    plan = {"evaluation": {"dev": "dev-hash", "test": "test-hash"},
            "code": {key: key + "-hash" for key in TRAINING_DEPENDENCIES},
            "schedule": schedule, "corpora": {"wi": source}}
    report = {"evaluation_hashes": deepcopy(plan["evaluation"]), "code_sha256": deepcopy(plan["code"]),
              "device": "cpu", "seed": 42, "schedule": deepcopy(schedule),
              "test": {"scorer": "errant"}, "diagnostic_unconstrained_test": {"scorer": "errant"},
              "calibration": {"selected": {"scorer": "errant"}, "best_unconstrained": {"scorer": "errant"}},
              "dataset_manifest_sha256": hash_file(path)}
    return report, plan, prepared, path


@pytest.mark.parametrize("arm", ["wi", "combined"])
def test_verified_individual_and_combined_identity(tmp_path, arm):
    verify_report_identity(*receipt(tmp_path), arm)


@pytest.mark.parametrize("change", ["evaluation", "dependency", "device", "seed", "schedule",
                                    "plan-scorer", "actual-scorer", "dataset", "input"])
def test_publication_rejects_actual_receipt_identity_drift(tmp_path, change):
    report, plan, prepared, path = receipt(tmp_path)
    if change == "evaluation":
        report["evaluation_hashes"]["test"] = "other-test"
    elif change == "dependency":
        report["code_sha256"]["evaluate.py"] = "other-implementation"
    elif change in {"device", "seed"}:
        report[change] = "other"
    elif change == "schedule":
        report["schedule"]["epochs"] = 5
    elif change == "plan-scorer":
        plan["schedule"]["scorer"] = "approximate"
    elif change == "actual-scorer":
        report["diagnostic_unconstrained_test"]["scorer"] = "approximate"
    elif change == "dataset":
        path.write_text("changed")
    elif change == "input":
        prepared["input_files"] = [{"path": "other-source", "sha256": "other-hash"}]
    with pytest.raises(ValueError):
        verify_report_identity(report, plan, prepared, path, "wi")
