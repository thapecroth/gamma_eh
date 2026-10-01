import json

import pytest

from data import render, tokens
from prepare_pairs import reconstruct
from synthetic_challenge import (CHALLENGE_ADJECTIVES, CHALLENGE_NOUNS, CHALLENGE_TAILS,
                                 TRAIN_ADJECTIVES, TRAIN_NOUNS, TRAIN_TAILS,
                                 build, canonical, challenge_examples, example, mutation)


def apply_gold(row):
    value = row["source"].encode("utf-16-le")
    for edit in reversed(row["goldEdits"]):
        start, end = edit["start"] * 2, edit["end"] * 2
        assert value[start:end].decode("utf-16-le") == edit["original"]
        value = value[:start] + edit["replacement"].encode("utf-16-le") + value[end:]
    return value.decode("utf-16-le")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    output = tmp_path_factory.mktemp("harder-synthetic")
    manifest = build(output)
    return output, manifest


def test_explicit_gold_edits_preserve_utf16_and_line_endings():
    clean = "😀. The mice have a telescope.\r\nWe received an invitation."
    row = example(clean, "unicode", [mutation(clean, "have", "has"), mutation(clean, "received", "recieved")])
    assert row["goldEdits"][0]["start"] == clean.index("have") + 1
    assert "\r\n" in row["source"]
    assert apply_gold(row) == clean
    with pytest.raises(ValueError, match="Overlapping"):
        example("Hello.", "bad", [(0, 3, "x"), (2, 4, "y")])


def test_challenge_gold_is_consistent_and_covers_hard_families(corpus):
    output, manifest = corpus
    rows = json.loads((output / "challenge.json").read_text())["cases"]
    assert len(rows) == 512
    assert sum(row["clean"] for row in rows) == 272
    assert len(manifest["challenge"]["families"]) == 16
    assert all(count == 32 for count in manifest["challenge"]["families"].values())
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert apply_gold(row) == row["target"]
        assert row["clean"] == (row["source"] == row["target"])
        assert bool(row["goldEdits"]) != row["clean"]
    assert any("who helped" in row["source"] for row in rows)
    assert any("`teh = 1`" in row["source"] for row in rows)
    assert any("item 35" in row["source"] for row in rows)


def test_training_splits_and_challenge_have_no_normalized_text_leakage(corpus):
    output, _ = corpus
    seen_text, seen_groups = set(), set()
    for split in ["train", "dev", "test"]:
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        text = {canonical(row[key]) for row in rows for key in ["source", "target"]}
        groups = {row["clean_group"] for row in rows}
        assert not text & seen_text
        assert not groups & seen_groups
        seen_text |= text
        seen_groups |= groups
        for row in rows:
            assert reconstruct(row["tokens"], row["tags"]) == render(tokens(row["target"]))
    challenge = list(challenge_examples())
    assert not seen_text & {canonical(row[key]) for row in challenge for key in ["source", "target"]}
    assert not seen_groups & {row["clean_group"] for row in challenge}


def test_reserved_vocabulary_and_structures_are_not_training_inputs(corpus):
    assert not set(sum(TRAIN_NOUNS, ())) & set(sum(CHALLENGE_NOUNS, ()))
    assert not set(TRAIN_ADJECTIVES) & set(CHALLENGE_ADJECTIVES)
    assert not set(TRAIN_TAILS) & set(CHALLENGE_TAILS)
    output, _ = corpus
    train = (output / "train.jsonl").read_text()
    for reserved in ["who helped", "We learned that", "should have", "`teh = 1`", "item 35"]:
        assert reserved not in train


def test_augmentation_truthfully_remains_generated_train_only(corpus):
    output, manifest = corpus
    rows = [json.loads(line) for line in (output / "augmentation-candidates.jsonl").read_text().splitlines()]
    assert len(rows) == manifest["splits"]["train"]["rows"] == 9980
    assert sum(row["clean"] for row in rows) * 2 == len(rows)
    assert all(row["review_status"] == "synthetic-generated" and row["license"] == "CC0-1.0" for row in rows)
    assert {row["id"] for row in rows} == {json.loads(line)["id"] for line in (output / "train.jsonl").read_text().splitlines()}
    train_tags = {tag for row in rows for tag in row["tags"]}
    assert set(json.loads((output / "labels.json").read_text())) == train_tags


def test_rebuilding_is_byte_deterministic(corpus, tmp_path):
    output, manifest = corpus
    assert build(tmp_path) == manifest
    for name in ["train.jsonl", "dev.jsonl", "test.jsonl", "labels.json", "manifest.json", "challenge.json", "augmentation-candidates.jsonl"]:
        assert (tmp_path / name).read_bytes() == (output / name).read_bytes()
