import json
import pytest

from assemble_pilot import assemble
from data import edit_tags
from prepare_pairs import prepare


def write_rows(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def tagged(source, target):
    words, tags = edit_tags(source, target)
    return {"source": source, "target": target, "tokens": words, "tags": tags}


def test_unverified_teacher_requires_explicit_local_only_flags(tmp_path):
    source = tmp_path / "candidates.jsonl"
    teacher = {"source": "She have a book.", "target": "She has a book.", "category": "agreement",
               "license": "provider-terms-unverified", "origin": "llm-teacher",
               "review_status": "human-reviewed"}
    write_rows(source, [teacher])
    default = prepare([source], tmp_path / "default", allow_weak_train=True)
    assert default["splits"]["train"]["accepted"] == 0
    local = prepare([source], tmp_path / "local", allow_weak_train=True, allow_unverified_teacher_terms=True)
    assert local["publication_allowed"] is False
    assert local["evaluation_ready"] is False
    row = json.loads((tmp_path / "local/train.jsonl").read_text())
    assert row["license"] == "provider-terms-unverified"
    assert row["review_status"] == "weak-supervision"
    no_weak = prepare([source], tmp_path / "no-weak", allow_unverified_teacher_terms=True)
    assert no_weak["splits"]["train"]["accepted"] == 0
    teacher["origin"] = "other-corpus"
    write_rows(source, [teacher])
    other = prepare([source], tmp_path / "other", allow_weak_train=True, allow_unverified_teacher_terms=True)
    assert other["splits"]["train"]["accepted"] == 0


def fixture_corpus(path):
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"origin": "original-template-v1", "license": "CC0-1.0"}))
    write_rows(path / "train.jsonl", [tagged("She have a book.", "She has a book.")])
    write_rows(path / "dev.jsonl", [tagged("We have an apple.", "We have an apple.")])
    write_rows(path / "test.jsonl", [tagged("They have a pen.", "They have a pen.")])


def test_pilot_keeps_teacher_train_only_and_checks_normalized_leakage(tmp_path):
    base, weak, output = [tmp_path / name for name in ["base", "weak", "pilot"]]
    fixture_corpus(base)
    weak.mkdir()
    (weak / "manifest.json").write_text(json.dumps({"publication_allowed": False, "licenses": {"provider-terms-unverified": 4}}))
    write_rows(weak / "train.jsonl", [
        tagged("We has an apple.", "We have an apple."),
        tagged("They has a pen.", "They have a pen."),
        tagged("SHE have a book.", "She has a book."),
        tagged("He have a ticket.", "He has a ticket."),
    ])
    manifest = assemble(base, weak, output)
    assert manifest["publication_allowed"] is False
    assert manifest["counts"]["weak_train_added"] == 1
    assert manifest["counts"]["heldout_overlap_dropped"] == 2
    assert manifest["counts"]["duplicate_or_conflicting_source_dropped"] == 1
    assert manifest["splits"]["train"]["rows"] == 2
    assert (output / "dev.jsonl").read_bytes() == (base / "dev.jsonl").read_bytes()
    assert (output / "test.jsonl").read_bytes() == (base / "test.jsonl").read_bytes()
    assert json.loads((output / "labels.json").read_text()) == ["KEEP", "REPLACE:has"]
    with pytest.raises(ValueError, match="already exists"): assemble(base, weak, output)


def test_pilot_rejects_untrusted_evaluation_origin(tmp_path):
    base, weak = tmp_path / "base", tmp_path / "weak"
    fixture_corpus(base)
    weak.mkdir()
    (weak / "manifest.json").write_text('{"licenses": {}}')
    (base / "manifest.json").write_text('{"origin": "llm-teacher", "license": "CC0-1.0"}')
    with pytest.raises(ValueError, match="original CC0"): assemble(base, weak, tmp_path / "pilot")


def test_pilot_excludes_every_reference_and_cross_column_evaluation_match(tmp_path):
    base, weak, output = [tmp_path / name for name in ["base", "weak", "pilot"]]
    fixture_corpus(base)
    evaluation = base / "evaluation"
    evaluation.mkdir()
    for split in ["dev", "test"]:
        write_rows(evaluation / f"{split}.jsonl", [
            {"source": f"Extra {split} source.", "references": [f"Extra {split} target.", f"Other {split} reference."]}])
    weak.mkdir()
    (weak / "manifest.json").write_text('{"publication_allowed": false, "licenses": {}}')
    write_rows(weak / "train.jsonl", [
        tagged("EXTRA DEV TARGET.", "Other sentence."),
        tagged("Another sentence.", "Extra test source."),
        tagged("Other test reference.", "Safe sentence."),
        tagged("He have a ticket.", "He has a ticket."),
    ])
    manifest = assemble(base, weak, output)
    assert manifest["counts"]["heldout_overlap_dropped"] == 3
    assert manifest["counts"]["weak_train_added"] == 1
    for split in ["dev", "test"]:
        assert (output / "evaluation" / f"{split}.jsonl").read_bytes() == (evaluation / f"{split}.jsonl").read_bytes()


def test_pilot_can_bound_template_training_without_evaluation_vocab_leakage(tmp_path):
    base, weak = tmp_path / "base", tmp_path / "weak"
    fixture_corpus(base)
    write_rows(base / "train.jsonl", [tagged("She have a book.", "She has a book."),
                                       tagged("He are ready.", "He is ready.")])
    weak.mkdir()
    (weak / "manifest.json").write_text('{"publication_allowed": false, "licenses": {}}')
    write_rows(weak / "train.jsonl", [])
    manifest = assemble(base, weak, tmp_path / "pilot", max_template_train_rows=1)
    assert manifest["counts"]["template_train_selected"] == 1
    assert manifest["splits"]["train"]["rows"] == 1
    assert "REPLACE:is" not in json.loads((tmp_path / "pilot/labels.json").read_text())
    with pytest.raises(ValueError, match="nonnegative"):
        assemble(base, weak, tmp_path / "invalid", max_template_train_rows=-1)
