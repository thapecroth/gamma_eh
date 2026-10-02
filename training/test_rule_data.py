"""Provenance, leakage, reconstruction and ambiguous-corruption boundaries."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from data import edit_tags
from edit_ops import reconstruct
from import_eracond import dialog_split, key, materialize, unpack, MAX_TAR_BYTES, MAX_MEMBERS
from pairs import hash_file, evaluation_keys
from prepare_pairs import prepare
from rule_data import build, corrupt, seed_pairs


def archive(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as target:
        for name, text in files.items():
            raw = text.encode()
            member = tarfile.TarInfo("fixture/" + name)
            member.size = len(raw)
            target.addfile(member, io.BytesIO(raw))
    return stream.getvalue()


def fixture_archive(mismatch=False):
    files = {"LICENSE": "MIT License\nCopyright (c) 2022 YUAN Xun\n"}
    names = {}
    for split, subject in (("train", "She"), ("dev", "He"), ("test", "They")):
        dialog = next(f"data/1/{index}" for index in range(1000)
                      if dialog_split(f"data/1/{index}", 42) == split)
        names[split] = dialog
        prefix, number = dialog.rsplit("/", 1)
        source = f"{subject} have the notebook.\nWe share the same phrase.\n"
        target = f"{subject} {'have' if subject == 'They' else 'has'} the notebook.\nWe share the same phrase.\n"
        files[f"{prefix}/orig/{number}.txt"] = source
        files[f"{prefix}/corr/{number}.txt"] = target + ("extra\n" if mismatch and split == "train" else "")
    return archive(files), names


def import_fixture(path):
    blob, names = fixture_archive()
    result = materialize(path, fetch=lambda: blob, expected_sha256=hashlib.sha256(blob).hexdigest())
    return result, names


def test_human_import_splits_dialogs_and_removes_both_sides_of_leakage(tmp_path):
    manifest, names = import_fixture(tmp_path / "human")
    assert manifest["counts"]["original_utterances"] == 6
    assert manifest["splits"]["train"]["rows"] == 1
    populations = []
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in (tmp_path / "human" / f"{split}.jsonl").read_text().splitlines()]
        assert all(row["dialog_id"] == names[split] for row in rows)
        assert all(row["review_status"] == "human-reviewed" for row in rows)
        assert all("same phrase" not in row["source"] for row in rows)
        populations.append(evaluation_keys(rows))
    assert not populations[0] & (populations[1] | populations[2])
    assert not populations[1] & populations[2]


def test_import_rejects_hash_alignment_and_unsafe_archive(tmp_path):
    blob, _ = fixture_archive()
    with pytest.raises(ValueError, match="hash mismatch"):
        materialize(tmp_path / "bad-hash", fetch=lambda: blob)
    blob, _ = fixture_archive(mismatch=True)
    with pytest.raises(ValueError, match="Misaligned"):
        materialize(tmp_path / "bad-align", fetch=lambda: blob, expected_sha256=hashlib.sha256(blob).hexdigest())
    with pytest.raises(ValueError, match="Invalid archive member"):
        unpack(archive({"../escape": "unsafe"}))


def test_archive_limits_include_tar_headers_and_zero_byte_members():
    import gzip
    with pytest.raises(ValueError, match="decompressed"):
        unpack(gzip.compress(b"\0" * (MAX_TAR_BYTES + 1)))
    with pytest.raises(ValueError, match="member limit"):
        unpack(archive({f"empty-{index}": "" for index in range(MAX_MEMBERS + 1)}))


@pytest.mark.parametrize("text", [
    "I read the book yesterday.", "She read the note.", "Does she have a notebook?",
    "I recommend that he have more time.", "She had had enough time.",
    "We knew that that answer was correct.", "An hour passed.", "A university opened.",
    'The quote is "She have time".', "Use `receive()` here.",
    "The label says 'receive the package'.", "The label says ‘receive the package’.",
])
def test_corruption_does_not_invent_context_dependent_errors(text):
    assert all(category != "agreement" for _, category, _ in corrupt(text))
    assert all(rule not in {"article-swap", "tense-change", "duplicate-word"} for _, _, rule in corrupt(text))
    if '"' in text or '`' in text or "The label says" in text:
        assert corrupt(text) == []


def test_contractions_are_not_mistaken_for_quoted_text():
    assert any(category == "spelling" for _, category, _ in corrupt("We're writing the necessary message."))


def test_rule_pairs_reconstruct_and_keep_real_provenance():
    seeds = [{"id": "sample", "source": "She has the necessary information.", "family": "sample",
              "origin": "agent-authored-rule-seeds-v1", "review_status": "unreviewed", "license": "CC0-1.0"}]
    rows = list(seed_pairs(seeds))
    assert any(row["source"] == "She have the necessary information." for row in rows)
    assert any(row["source"] == row["target"] for row in rows)
    for row in rows:
        words, tags = edit_tags(row["source"], row["target"], 2)
        assert reconstruct(words, tags, 2) == row["target"]
        assert row["review_status"] == "unreviewed"
    with pytest.raises(ValueError, match="unreviewed"):
        list(seed_pairs([{**seeds[0], "review_status": "human-reviewed"}]))


@pytest.mark.parametrize("license_id,publication_allowed", [("MIT", True), ("CC-BY-NC-SA-4.0", False)])
def test_explicit_train_only_preserves_reviewed_training_without_fabricated_eval(tmp_path, license_id, publication_allowed):
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"source": "She have time.", "target": "She has time.",
                                  "category": "agreement", "license": license_id, "review_status": "human-reviewed"}) + "\n")
    manifest = prepare([source], tmp_path / "prepared", schema=2, train_only=True, allow_research=True)
    assert manifest["splits"]["train"]["accepted"] == 1
    assert not manifest["evaluation_ready"]
    assert manifest["evaluation"]["dev"]["rows"] == 0
    assert manifest["publication_allowed"] is publication_allowed
    assert manifest["training_purpose"] == "local-research"
    row = json.loads((tmp_path / "prepared/train.jsonl").read_text())
    assert row["review_status"] == "human-reviewed"


def test_mixture_is_deterministic_and_excludes_diagnostics_and_human_refs(tmp_path):
    human = tmp_path / "human"
    import_fixture(human)
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(json.dumps({"id": "seed", "source": "We have the necessary address.", "family": "messages",
                                 "origin": "agent-authored-rule-seeds-v1", "review_status": "unreviewed", "license": "CC0-1.0"}) + "\n")
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(json.dumps({"review_status": "unreviewed", "cases": [
        {"source": "I have the secret diagnostic notebook.", "target": "I have the secret diagnostic notebook."}]}))
    templates = tmp_path / "templates"
    templates.mkdir()
    rows = [{"source": "I have the secret diagnostic notebook.", "target": "I have the secret diagnostic notebook.",
             "origin": "synthetic", "tokens": ["I", "have"], "tags": ["KEEP", "KEEP"]},
            {"source": "They share a story.", "target": "They share a story.",
             "origin": "synthetic", "tokens": ["They", "share"], "tags": ["KEEP", "KEEP"]}]
    (templates / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (templates / "manifest.json").write_text(json.dumps({"license": "CC0-1.0", "origin": "original-template-v1",
        "splits": {"train": {"sha256": hash_file(templates / "train.jsonl")}}}))
    results = [build(tmp_path / name, human, seeds, diagnostics, max_rows=32, template_data=templates)
               for name in ("one", "two")]
    paths = [tmp_path / name / "prepared" for name in ("one", "two")]
    assert hash_file(paths[0] / "train.jsonl") == hash_file(paths[1] / "train.jsonl")
    assert results[0]["mixture_counts"]["heldout_overlap_removed"] == 1
    assert (paths[0] / "ERACOND-LICENSE").read_text() == (human / "LICENSE").read_text()
    assert results[0]["license_files"]["ERACOND-LICENSE"] == hash_file(paths[0] / "ERACOND-LICENSE")
    train_rows = [json.loads(line) for line in (paths[0] / "train.jsonl").read_text().splitlines()]
    heldout = evaluation_keys([row for split in ("dev", "test")
                              for row in [json.loads(line) for line in (paths[0] / "evaluation" / f"{split}.jsonl").read_text().splitlines()]])
    assert not any(key(row[field]) in heldout for row in train_rows for field in ("source", "target"))
    assert any(row["origin"] == "ErAConD" for row in train_rows)
    assert any(row["origin"] == "rule-corruption-v1" for row in train_rows)
    assert all(row["review_status"] == "weak-supervision" for row in train_rows if row["origin"] == "rule-corruption-v1")


def test_reserved_keys_include_target_and_every_reference():
    assert evaluation_keys([{"source": "Original source.", "target": "Canonical target.",
                             "references": ["Alternate correction."]}]) == {
        "original source.", "canonical target.", "alternate correction."}


def test_bundled_agent_fixtures_remain_unreviewed_and_disjoint():
    root = Path(__file__).resolve().parents[1]
    seeds = [json.loads(line) for line in (root / "data/rule-seeds.jsonl").read_text().splitlines()]
    diagnostics = json.loads((root / "data/counterexamples.json").read_text())
    assert len(seeds) == len(diagnostics["cases"]) == 100
    assert diagnostics["review_status"] == "unreviewed"
    assert all(seed["review_status"] == "unreviewed" for seed in seeds)
    assert not {key(row["source"]) for row in seeds} & evaluation_keys(diagnostics["cases"])
