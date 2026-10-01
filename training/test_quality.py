"""Pure-stdlib quality workflow boundaries; these also run without training dependencies."""
import json
from pathlib import Path
from argparse import Namespace

import pytest

from data import edit_tags, tokens
from edit_ops import decode_word, reconstruct
from evaluate import (apply_proposals, choose_calibration, choose_joint_calibration, decode_proposal,
                      evaluate_records, score_predictions)
from experiments import build, common_training_population, select_rows
from import_jfleg import materialize
from pairs import evaluation_keys, group_id
from prepare_pairs import prepare


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def words(text):
    from edit_ops import TOKEN_RE
    return [{"text": match.group(), "start": match.start(), "end": match.end()} for match in TOKEN_RE.finditer(text)]


def test_shared_decoder_fixture_and_invalid_payloads():
    fixture = json.loads(Path(__file__).with_name("edit-parity.json").read_text())
    for case in fixture["cases"]:
        assert decode_word(case["word"], case["tag"], 2) == case["tokens"]
    for case in fixture["invalid"]:
        with pytest.raises(ValueError): decode_word(case["word"], case["tag"], 2)
    with pytest.raises(ValueError): decode_word("word", "CASE:UPPER", 1)


@pytest.mark.parametrize("source,target", [
    ("Book.", "A book."), ("He study here.", "He studies here."),
    ("We need book.", "We need a small book."), ("hello world.", "Hello, world!"),
    ("She reads.", "The careful teacher reads."), ("HELLO.", "hello."),
    ("We go home.", "We will be going home."),
])
def test_v2_alignment_roundtrip(source, target):
    source_words, tags = edit_tags(source, target, 2)
    assert reconstruct(source_words, tags, 2) == target
    assert source_words == tokens(source)


def test_v1_tags_remain_default_and_v2_payloads_are_bounded():
    assert edit_tags("He work here.", "He works here.")[1][1] == "REPLACE:works"
    assert edit_tags("He work here.", "He works here.", 2)[1][1] == "SUFFIX:ADD_S"
    with pytest.raises(ValueError): edit_tags("Book.", "One two three four five book.", 2)
    with pytest.raises(ValueError): reconstruct(["word"], [], 2)


def test_prepare_retains_unsupported_reviewed_targets_and_prevents_leakage(tmp_path):
    target = next(f"The careful teacher reads for task {index}." for index in range(1000)
                  if int(group_id(f"The careful teacher reads for task {index}.")[:8], 16) % 100 >= 90)
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [
        {"source": "She reads.", "target": target, "category": "mixed", "license": "CC0-1.0", "review_status": "human-reviewed"},
        {"source": target.replace("reads", "read"), "target": target, "category": "agreement", "license": "CC0-1.0", "review_status": "unreviewed"},
        {"source": "He have a pen.", "target": "He has a pen.", "category": "agreement", "license": "CC0-1.0", "review_status": "unreviewed"},
    ])
    output = tmp_path / "prepared"
    manifest = prepare([source], output, allow_weak_train=True)
    assert manifest["evaluation"]["test"]["rows"] == 1
    assert manifest["splits"]["test"]["accepted"] == 0
    assert manifest["coverage_by_category"]["test"]["mixed"]["unrepresentable"] == 1
    assert manifest["counts"]["weak_heldout_group_dropped"] == 1
    population = json.loads((output / "evaluation/test.jsonl").read_text())
    assert population["source"] == "She reads."
    assert population["references"] == [target]
    assert target not in (output / "train.jsonl").read_text()


def test_prepare_keeps_heldout_rows_outside_train_vocabulary(tmp_path):
    target = next(f"They scrutinize project {index}." for index in range(1000)
                  if int(group_id(f"They scrutinize project {index}.")[:8], 16) % 100 >= 90)
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [
        {"source": target.replace("scrutinize", "scrutinise"), "target": target, "category": "spelling", "license": "CC0-1.0", "review_status": "human-reviewed"},
        {"source": "He have a pen.", "target": "He has a pen.", "category": "agreement", "license": "CC0-1.0", "review_status": "unreviewed"},
    ])
    manifest = prepare([source], tmp_path / "prepared", allow_weak_train=True, schema=2)
    assert manifest["evaluation"]["test"]["rows"] == 1
    assert manifest["splits"]["test"]["unsupported_tag_examples"] == 1
    assert manifest["splits"]["test"]["accepted"] == 0


def test_complete_multireference_scoring_and_failed_identity_output():
    rows = [{"source": "She have a book.", "references": ["She has a book.", "She owns a book."]},
            {"source": "Book.", "references": ["A carefully illustrated book."]},
            {"source": "We watched her duck.", "references": ["We watched her duck."]}]
    predictions = ["She owns a book.", "Book.", "We watched her duck."]
    report = score_predictions(rows, predictions, failures=1)
    assert report["sentences"] == 3
    assert report["true_positive_edits"] == 1
    assert report["missed_edits"] == 1
    assert report["clean_sentence_false_positive_rate"] == 0
    assert report["inference_failures"] == 1
    assert "not ERRANT" in report["scoring_note"]
    with pytest.raises(ValueError): score_predictions(rows, predictions[:-1])
    with pytest.raises(ValueError): score_predictions([{"source": "A book.", "references": []}], ["A book."])


def test_missing_clean_controls_and_unsafe_thresholds_trigger_explicit_no_edit():
    rows = [{"source": "She have a book.", "references": ["She has a book."]}]
    candidate = {**score_predictions(rows, ["She has a book."]), "threshold": .8}
    no_edit = {**score_predictions(rows, ["She have a book."]), "threshold": 1., "disable_model_edits": True}
    calibrated = choose_calibration([candidate], no_edit)
    assert candidate["clean_sentence_false_positive_rate"] is None
    assert calibrated["disable_model_edits"] is True
    assert calibrated["constraints_met"] is False
    assert calibrated["best_unconstrained"]["edit_f0_5"] == 1.
    candidate["clean_sentence_false_positive_rate"] = 0
    assert choose_calibration([candidate], no_edit)["constraints_met"] is True
    candidate["inference_failures"] = 1
    assert choose_calibration([candidate], no_edit)["constraints_met"] is False


def test_quantization_drift_cannot_enable_a_policy_that_fails_one_export():
    base = {"predicted_edits": 10, "edit_precision": .99, "clean_sentence_false_positive_rate": 0,
            "edit_f0_5": .5, "threshold": .8, "inference_failures": 0}
    no_edit = {**base, "threshold": 1., "predicted_edits": 0, "edit_f0_5": 0}
    assert choose_joint_calibration([base], [{**base, "edit_precision": .5}], no_edit)["disable_model_edits"] is True
    int8 = [base, {**base, "threshold": .95, "edit_f0_5": .4}]
    fp32 = [{**base, "edit_precision": .5}, {**base, "threshold": .95, "edit_f0_5": .3}]
    shared = choose_joint_calibration(int8, fp32, no_edit)
    assert shared["selected"]["threshold"] == .95
    assert shared["constraints_met"] is True


def test_deletion_and_article_guards_match_browser_contract():
    text = "Hello, world."
    proposal = decode_proposal(text, words(text), 1, "DELETE", .99, 2)
    assert apply_proposals(text, [proposal], .8) == "Hello world."
    text = "Hello,world."
    proposal = decode_proposal(text, words(text), 1, "DELETE", .99, 2)
    assert apply_proposals(text, [proposal], .8) == "Hello world."
    text = "We read books."
    assert decode_proposal(text, words(text), 2, "DELETE", .999, 2) is None
    text = "I have the book."
    assert decode_proposal(text, words(text), 2, "DELETE", .979, 2) is None
    assert decode_proposal(text, words(text), 2, "DELETE", .99, 2) is not None
    assert decode_proposal(text, words(text), 2, "DELETE", .99, 1) is None
    text = "She had had a book."
    assert decode_proposal(text, words(text), 1, "DELETE", .999, 2) is None
    text = "I have a university."
    assert decode_proposal(text, words(text), 2, "REPLACE_EXACT:an", .999, 2) is None
    text = "I have a hour."
    assert decode_proposal(text, words(text), 2, "REPLACE_EXACT:an", .999, 2) is not None


def test_overlapping_proposals_and_disabled_model_never_apply_twice():
    text = "hello world."
    proposed = [{"start": 0, "end": 5, "replacement": "Hello", "confidence": .99, "category": "case"},
                {"start": 0, "end": 0, "replacement": "Well, ", "confidence": .99, "category": "prepend"}]
    assert apply_proposals(text, proposed, .8, disabled=True) == text
    assert apply_proposals(text, proposed, .8) == "Well, hello world."
    assert apply_proposals(text, proposed[:1], .8, category_thresholds={"case": 1.}) == text
    assert apply_proposals(text, proposed[:1], 1., category_thresholds={"case": .8}) == text


def test_jfleg_import_preserves_every_reference_and_refuses_wrong_population(tmp_path):
    def fetch(name, revision):
        assert len(revision) == 40
        split, filename = name.split("/")
        count = 754 if split == "dev" else 747
        label = filename.rsplit(".", 1)[1]
        return "".join(f"{label} example {index}.\n" for index in range(count)).encode()
    manifest = materialize(tmp_path / "evaluation", fetch=fetch)
    assert manifest["training_allowed"] is False
    assert manifest["license"] == "CC-BY-NC-SA-4.0"
    row = json.loads((tmp_path / "evaluation/dev.jsonl").read_text().splitlines()[0])
    assert len(row["references"]) == 4
    assert row["review_status"] == "human-reference-benchmark"
    with pytest.raises(ValueError, match="already exists"): materialize(tmp_path / "evaluation", fetch=fetch)
    with pytest.raises(ValueError, match="population"):
        materialize(tmp_path / "short", fetch=lambda *_: b"One row.\n")


def test_experiment_selector_removes_conflicts_and_eval_only_data(tmp_path):
    source = tmp_path / "rows.jsonl"
    base = {"license": "CC0-1.0", "category": "mixed"}
    write_rows(source, [{**base, "source": "He have a book.", "target": "He has a book."},
                       {**base, "source": "He have a book.", "target": "He had a book."},
                       {**base, "source": "She have a pen.", "target": "She has a pen.", "evaluation_only": True},
                       {**base, "source": "They has a pen.", "target": "They have a pen."}])
    selected, counts = select_rows(source, 10, 10, 42, set(), set())
    assert len(selected) == 1
    assert selected[0]["source"] == "They has a pen."
    assert counts["conflicting_source"] == 1
    assert counts["invalid"] == 1


def test_cross_role_evaluation_strings_are_excluded_from_either_training_field(tmp_path):
    heldout = [{"source": "She have a pen.", "references": ["She has a pen.", "She owns a pen."]}]
    keys = evaluation_keys(heldout)
    path = tmp_path / "rows.jsonl"
    write_rows(path, [
        {"source": "She owns a pen.", "target": "She owns one pen.", "license": "CC0-1.0"},
        {"source": "She have pen.", "target": "She have a pen.", "license": "CC0-1.0"},
        {"source": "They has books.", "target": "They have books.", "license": "CC0-1.0"},
    ])
    selected, counts = select_rows(path, 10, 10, 42, keys, {group_id(value) for value in keys})
    assert counts["heldout_overlap"] == 2
    assert len(selected) == 1


def test_controlled_experiments_cap_equal_raw_counts_and_keep_eval_hashes(tmp_path):
    template, evaluation = tmp_path / "template", tmp_path / "evaluation"
    template.mkdir(); evaluation.mkdir()
    train = [{"source": f"He have a book for task {index}.", "target": f"He has a book for task {index}."} for index in range(8)]
    write_rows(template / "train.jsonl", train)
    for split in ["dev", "test"]:
        write_rows(template / f"{split}.jsonl", [{"source": f"We have a {split} book.", "target": f"We have a {split} book."}])
        write_rows(evaluation / f"{split}.jsonl", [{"source": f"She has a {split} pen.", "references": [f"She has a {split} pen."]}])
    weak = tmp_path / "weak.jsonl"
    write_rows(weak, [{"source": f"They has a pen for task {index}.", "target": f"They have a pen for task {index}.",
                      "license": "CC0-1.0", "category": "agreement", "review_status": "unreviewed"} for index in range(2)])
    args = Namespace(output=tmp_path / "run", rows=8, weak=[weak], max_scanned=20, epochs=2,
                     batch_size=2, evaluation_dir=evaluation, template=template, seed=42,
                     max_labels=100, learning_rate=.0005, local_files_only=True, device="cpu")
    plan = build(args)
    assert plan["effective_raw_rows_per_arm"] == 4
    assert len(plan["arms"]) == 4
    assert {value["raw_train_rows"] for value in plan["dataset_manifests"].values()} == {4}
    for arm in plan["arms"]:
        assert arm["command"][arm["command"].index("--evaluation-dir") + 1] == str(evaluation.resolve())
    with pytest.raises(ValueError, match="already exists"): build(args)


def test_common_budget_equalizes_supported_rows_without_trimming_evaluation(tmp_path):
    manifests = {}
    for name, lengths in [("template", [3, 3, 3, 3]), ("mixed", [3, 7, 3])]:
        directory = tmp_path / name
        directory.mkdir()
        records = [{"pair_id": f"{name}-{index}", "tokens": ["word"] * length,
                    "source": "text", "target": "text"} for index, length in enumerate(lengths)]
        write_rows(directory / "train.jsonl", records)
        manifests[name] = {"splits": {"train": {"accepted": len(records)}}}
        (directory / "dev.jsonl").write_text("Retained full evaluation sentinel\n")
    result = common_training_population(tmp_path, manifests, lambda words: len(words) + 2, 6, 42)
    assert result["rows_per_arm"] == 2
    assert result["coverage"]["mixed"]["token_budget_rejected"] == 1
    for name in manifests:
        assert len((tmp_path / name / "train.jsonl").read_text().splitlines()) == 2
        assert (tmp_path / name / "dev.jsonl").read_text() == "Retained full evaluation sentinel\n"
