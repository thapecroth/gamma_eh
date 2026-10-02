import json
import shutil

import pytest

import calibrate_judge
from compare_judge_rl import command, plan
from judge import FLAGS, Judge, validate_calibration, judge_spec
from pairs import hash_file
from test_compare_rl import fixture_dataset
from test_judge import transport


def prepared(tmp_path, monkeypatch):
    data, evaluation = fixture_dataset(tmp_path)
    path = data / "train.jsonl"
    row = json.loads(path.read_text())
    row.update({"license": "CC0-1.0", "origin": "original-template-v1"})
    row.pop("dialog_id")
    path.write_text(json.dumps(row) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["splits"]["train"]["sha256"] = hash_file(path)
    (data / "manifest.json").write_text(json.dumps(manifest))
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"schema": 1, "license": "CC0-1.0", "origin": "agent-authored-judge-calibration-v1",
        "review_status": "unreviewed", "selection_use": False, "scope": "Diagnostic fixture.",
        "cases": [{"id": "one", "source": "The hedgehog rests beside a fern.",
                   "candidate": "The hedgehog rests beside a fern.", "expected": {name: True for name in FLAGS}}]}))
    monkeypatch.setattr(calibrate_judge, "Judge", lambda *args, **kwargs: Judge(*args, transport=transport, **kwargs))
    output = tmp_path / "calibration"
    assert calibrate_judge.run(fixture, output, execute=True)["passed"]
    return data, evaluation, fixture, output / "receipt.json"


def test_plan_freezes_judge_subset_and_matched_commands_without_execution(tmp_path, monkeypatch):
    data, evaluation, fixture, calibration = prepared(tmp_path, monkeypatch)
    output = tmp_path / "comparison"
    spec = plan(data, evaluation, output, calibration, fixture=fixture, device="cpu")
    assert not output.exists()
    assert spec["worst_case_judge_budget"] == {"pairs": 16, "requests": 8}
    initial = tmp_path / "initial"
    left, right = command(spec, "supervised", initial), command(spec, "llm-judge-reinforce", initial)
    for flag in ("--data", "--seed", "--epochs", "--batch-size", "--initial-checkpoint", "--learning-rate"):
        assert left[left.index(flag) + 1] == right[right.index(flag) + 1]
    assert "--judge-calibration" not in left and "--judge-calibration" in right
    assert "--judge-subset-sha256" in right


def test_calibration_rejects_truncated_verdicts_and_flipped_receipt(tmp_path, monkeypatch):
    _, _, fixture, calibration = prepared(tmp_path, monkeypatch)
    validate_calibration(calibration, judge_spec(), fixture)
    verdicts = calibration.parent / "round-1/verdicts.jsonl"
    verdicts.write_text("")
    with pytest.raises(ValueError, match="ledger changed"):
        validate_calibration(calibration, judge_spec(), fixture)


def test_calibration_rejects_copied_rounds_even_with_updated_hashes(tmp_path, monkeypatch):
    _, _, fixture, calibration = prepared(tmp_path, monkeypatch)
    receipt = json.loads(calibration.read_text())
    for name in receipt["round_receipts"][0]["ledgers"]:
        shutil.copyfile(calibration.parent / "round-1" / name, calibration.parent / "round-2" / name)
    receipt["round_receipts"][1] = receipt["round_receipts"][0]
    calibration.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="reused live request"):
        validate_calibration(calibration, judge_spec(), fixture)


def test_plan_rejects_oversized_live_budget_and_calibration_text_leakage(tmp_path, monkeypatch):
    data, evaluation, fixture, calibration = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="finite budget"):
        plan(data, evaluation, tmp_path / "output", calibration, fixture=fixture, max_pairs=1)
    path = data / "train.jsonl"
    row = json.loads(path.read_text())
    row["target"] = json.loads(fixture.read_text())["cases"][0]["candidate"]
    path.write_text(json.dumps(row) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["splits"]["train"]["sha256"] = hash_file(path)
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="calibration source/candidate overlaps"):
        plan(data, evaluation, tmp_path / "output", calibration, fixture=fixture)


def test_existing_warmup_requires_original_data_and_ordered_labels(tmp_path, monkeypatch):
    data, evaluation, fixture, calibration = prepared(tmp_path, monkeypatch)
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "config.json").write_text(json.dumps({"model_type": "bert", "id2label": {"0": "KEEP", "1": "REPLACE:has"},
        "label2id": {"KEEP": 0, "REPLACE:has": 1}}))
    (initial / "model.safetensors").write_bytes(b"fixture")
    receipt = {"objective": {"name": "supervised"}, "training_data_sha256": "wrong",
               "labels_sha256": hash_file(data / "labels.json"), "dataset_manifest_sha256": hash_file(data / "manifest.json"),
               "files": {name: hash_file(initial / name) for name in ("config.json", "model.safetensors")}}
    (initial / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="data/labels/provenance"):
        plan(data, evaluation, tmp_path / "output", calibration, fixture=fixture, initial_checkpoint=initial)
