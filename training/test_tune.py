import json
from pathlib import Path

import pytest

from pairs import hash_file
from tune import completed, plan, PROFILES


def test_plan_freezes_policy_device_and_data(tmp_path):
    for split in ["train", "dev", "test"]:
        (tmp_path / f"{split}.jsonl").write_text('{}\n')
    (tmp_path / "labels.json").write_text('["KEEP"]\n')
    (tmp_path / "manifest.json").write_text(json.dumps({"license": "CC0-1.0", "publication_allowed": True,
        "splits": {s: {"sha256": hash_file(tmp_path / f"{s}.jsonl")} for s in ["train", "dev", "test"]}}))
    spec = plan(tmp_path, device="cpu")
    assert spec["device"] == "cpu"
    assert spec["calibration_sha256"] == hash_file(Path(__file__).with_name("calibration.py"))
    assert plan(tmp_path, device="cuda") != spec
    (tmp_path / "train.jsonl").write_text('changed\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        plan(tmp_path)


def test_completed_receipt_validates_assets_and_profile(tmp_path):
    profile = PROFILES[0]
    spec = {"device": "cuda", "seed": 42, "batch_size": 128, "max_length": 64, "epochs": 8}
    assets = {}
    for name in ["model.onnx", "labels.json", "vocab.txt", "dataset-manifest.json"]:
        (tmp_path / name).write_text(name)
        assets[name] = {"sha256": hash_file(tmp_path / name)}
    spec.update(dataset_manifest_sha256=assets["dataset-manifest.json"]["sha256"], labels_sha256=assets["labels.json"]["sha256"])
    (tmp_path / "manifest.json").write_text(json.dumps({"files": assets}))
    receipt = {"device": "cuda", "seed": 42, "training_config": {"learning_rate": profile["learning_rate"],
        "keep_weight": profile["keep_weight"], "batch_size": 128, "max_length": 64, "epochs": 8}}
    (tmp_path / "evaluation.json").write_text(json.dumps(receipt))
    assert completed(tmp_path, profile, spec)
    (tmp_path / "model.onnx").write_text('corrupted')
    with pytest.raises(ValueError, match="asset mismatch"):
        completed(tmp_path, profile, spec)

