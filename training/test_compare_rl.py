import json

import pytest

from compare_rl import command, completed, completion_metadata, plan, verify_inputs
from pairs import hash_file


def fixture_dataset(tmp_path):
    data, evaluation = tmp_path / "data", tmp_path / "evaluation"
    data.mkdir()
    evaluation.mkdir()
    (data / "train.jsonl").write_text(json.dumps({"source": "She have a lantern.", "target": "She has a lantern.",
                                                "dialog_id": "train-dialog"}) + "\n")
    for split in ("dev", "test"):
        (data / f"{split}.jsonl").write_text("")
        source = "We was waiting outside." if split == "dev" else "They has a small basket."
        target = "We were waiting outside." if split == "dev" else "They have a small basket."
        (evaluation / f"{split}.jsonl").write_text(json.dumps({"id": split, "source": source,
            "references": [target], "origin": "ErAConD", "dialog_id": split + "-dialog"}) + "\n")
    (data / "labels.json").write_text('["KEEP", "REPLACE:has"]\n')
    (data / "manifest.json").write_text(json.dumps({
        "splits": {split: {"sha256": hash_file(data / f"{split}.jsonl")} for split in ("train", "dev", "test")},
        "evaluation": {split: {"sha256": hash_file(evaluation / f"{split}.jsonl")} for split in ("dev", "test")},
        "labels_sha256": hash_file(data / "labels.json")}))
    return data, evaluation


def test_dry_plan_freezes_inputs_without_creating_outputs(tmp_path):
    data, evaluation = fixture_dataset(tmp_path)
    output = tmp_path / "experiment"
    spec = plan(data, evaluation, output, device="cpu")
    assert not output.exists()
    assert spec["arms"] == ["supervised", "anchored-reinforce"]
    assert spec["keep_weight"] == .3
    assert spec["continuation_update_upper_bound"] == 2
    verify_inputs(spec)
    (data / "train.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="Frozen comparison"):
        verify_inputs(spec)


def test_continuation_commands_match_checkpoint_schedule_and_seed(tmp_path):
    data, evaluation = fixture_dataset(tmp_path)
    spec = plan(data, evaluation, tmp_path / "experiment", device="cpu")
    initial = tmp_path / "initial"
    a = command(spec, "supervised", 2, initial)
    b = command(spec, "anchored-reinforce", 2, initial)
    for flag in ("--initial-checkpoint", "--data", "--evaluation-dir", "--seed", "--epochs", "--batch-size",
                 "--max-length", "--learning-rate", "--keep-weight", "--base-model", "--base-revision"):
        assert a[a.index(flag) + 1] == b[b.index(flag) + 1]
    assert a[a.index("--keep-weight") + 1] == "0.3"
    assert a[a.index("--objective") + 1] == "supervised"
    assert b[b.index("--objective") + 1] == "anchored-reinforce"


def test_plan_rejects_empty_or_over_budget_population_and_zero_rl(tmp_path):
    data, evaluation = fixture_dataset(tmp_path)
    for overrides in ({"max_train_rows": 0}, {"rl_coefficient": 0.}, {"learning_rate": float("nan")}):
        with pytest.raises(ValueError):
            plan(data, evaluation, tmp_path / "experiment", **overrides)
    (data / "train.jsonl").write_text("")
    with pytest.raises(ValueError, match="bounded row budget or is empty"):
        plan(data, evaluation, tmp_path / "experiment")


def test_plan_rejects_training_dialog_overlap(tmp_path):
    data, evaluation = fixture_dataset(tmp_path)
    path = data / "train.jsonl"
    row = json.loads(path.read_text())
    row["dialog_id"] = "dev-dialog"
    path.write_text(json.dumps(row) + "\n")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["sha256"] = hash_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dialogs overlap"):
        plan(data, evaluation, tmp_path / "experiment")


def test_plan_rejects_diagnostic_leakage_into_training(tmp_path):
    data, evaluation = fixture_dataset(tmp_path)
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(json.dumps({"cases": [{"source": "Other input.", "target": "She has a lantern."}]}))
    with pytest.raises(ValueError, match="diagnostic source/reference"):
        plan(data, evaluation, tmp_path / "experiment", counterexamples=diagnostic)


@pytest.mark.parametrize("filename", ["manifest.json", "evaluation.json"])
def test_resume_rejects_changed_deployment_policy_or_report_metrics(tmp_path, filename):
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text('{"confidenceThreshold": 0.95, "disableModelEdits": true}')
    (model / "evaluation.json").write_text('{"test": {"edit_precision": 0.2}}')
    with pytest.raises(ValueError, match="completion receipt"):
        completion_metadata(tmp_path)
    completion_metadata(tmp_path, create=True)
    completion_metadata(tmp_path)
    with pytest.raises(FileExistsError):
        completion_metadata(tmp_path, create=True)
    path = model / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="metadata changed"):
        completion_metadata(tmp_path)


def test_completed_stage_rejects_a_different_keep_loss_weight(tmp_path):
    data, evaluation = fixture_dataset(tmp_path)
    spec = plan(data, evaluation, tmp_path / "experiment", device="cpu")
    directory = tmp_path / "experiment/supervised"
    model = directory / "model"
    model.mkdir(parents=True)
    report = {"schedule": {"epochs": spec["epochs"], "batch_size": spec["batch_size"],
                          "max_length": spec["max_length"], "learning_rate": spec["learning_rate"], "keep_weight": 1.},
              "seed": spec["seed"], "device": spec["device"], "objective": {"name": "supervised", "rl_coefficient": 0.},
              "initialization": {"files": None}, "dataset_manifest_sha256": hash_file(data / "manifest.json")}
    (model / "manifest.json").write_text("{}")
    (model / "evaluation.json").write_text(json.dumps(report))
    completion_metadata(directory, create=True)
    with pytest.raises(ValueError, match="frozen schedule"):
        completed(spec, "supervised", spec["epochs"], None)
