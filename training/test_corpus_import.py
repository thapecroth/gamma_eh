import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from corpus_import import archive_m2, canonical, dolt_rows, exclusions, materialize, m2_pairs, repository_slug, wdiff_pair
from prepare_pairs import prepare
from pairs import hash_file
from corpus_training import selected_arms, verify_completed


def spec(license_id="LicenseRef-Cambridge-WI"):
    return {"id": "wi", "format": "m2", "license": license_id,
            "license_scope": "Noncommercial research; no corpus redistribution.",
            "release_eligible": False, "source_url": "https://example.org/corpus",
            "supervision": "human-annotated", "training_split": "train"}


def test_m2_annotators_are_separate_and_noops_are_clean():
    value = "S They is here .\nA 1 2|||R:VERB:SVA|||are|||-REQUIRED-|||NONE|||0\nA -1 -1|||noop|||-NONE-|||REQUIRED|||NONE|||1\n"
    rows = list(m2_pairs(io.StringIO(value)))
    assert [r["target"] for r in rows] == ["They are here .", "They is here ."]
    assert canonical(rows[0]["target"]) == "They are here."


def test_insertion_replacement_same_boundary_and_unresolved_annotations():
    value = "S Cat sleep .\nA 0 0|||M:DET|||The|||-REQUIRED-|||NONE|||0\nA 0 1|||R:NOUN|||cat|||-REQUIRED-|||NONE|||0\n"
    assert list(m2_pairs(io.StringIO(value)))[0]["target"] == "The cat sleep ."
    unknown = "S Cat sleep .\nA 0 1|||UNK|||-NONE-|||-REQUIRED-|||NONE|||0\n"
    assert list(m2_pairs(io.StringIO(unknown))) == [{"reject_reason": "unresolved_annotation"}]


@pytest.mark.parametrize("edit", ["A 0 9|||R:NOUN|||dog|||-REQUIRED-|||NONE|||0", "A 0 0|||noop|||-NONE-|||REQUIRED|||0"])
def test_invalid_m2_coordinates_fail(edit):
    with pytest.raises(ValueError):
        list(m2_pairs(io.StringIO("S A cat sleeps .\n" + edit)))


def test_import_keeps_benchmarks_out_under_both_spacing_forms(tmp_path):
    benchmark = tmp_path / "eval.jsonl"
    benchmark.write_text(json.dumps({"source": "They is here .", "references": ["They are here ."]}) + "\n")
    heldout = exclusions([benchmark])
    rows = [{"source": "They is here.", "target": "They are here."},
            {"source": "He have a small book.", "target": "He has a small book."}]
    result = materialize(spec(), iter(rows), tmp_path / "out", heldout, 10, 100)
    assert result["counts"]["rejected:heldout_overlap"] == 1
    assert result["counts"]["selected_original_pairs"] == 1
    assert "They" not in (tmp_path / "out/candidates.jsonl").read_text()


def test_import_is_deterministic_and_discards_conflicting_sources(tmp_path):
    rows = [{"source": "He have a small book.", "target": "He has a small book."},
            {"source": "He have a small book.", "target": "He owns a small book."},
            {"source": "They works near the park.", "target": "They work near the park."}]
    a = materialize(spec(), iter(rows), tmp_path / "a", set(), 10, 100)
    b = materialize(spec(), iter(rows), tmp_path / "b", set(), 10, 100)
    assert a["sha256"] == b["sha256"]
    assert a["counts"]["conflicting_source"] == 1
    assert a["counts"]["selected_original_pairs"] == 1


def test_restricted_terms_require_explicit_research_and_survive_preparation(tmp_path):
    out = tmp_path / "source"
    materialize(spec(), iter([{"source": "He have a small book.", "target": "He has a small book."}]), out, set(), 10, 100)
    rejected = prepare([out / "candidates.jsonl"], tmp_path / "rejected", allow_weak_train=True, schema=2)
    assert rejected["splits"]["train"]["accepted"] == 0
    accepted = prepare([out / "candidates.jsonl"], tmp_path / "accepted", allow_weak_train=True,
                       schema=2, allow_research=True)
    assert accepted["splits"]["train"]["accepted"] > 0
    assert accepted["publication_allowed"] is False
    row = json.loads((tmp_path / "accepted/train.jsonl").read_text().splitlines()[0])
    assert row["license"] == "LicenseRef-Cambridge-WI"
    assert row["supervision"] == "human-annotated"
    assert row["source_split"] == "train"
    assert row["license_scope"] == spec()["license_scope"]


def test_source_release_restriction_blocks_permissive_base_license(tmp_path):
    out = tmp_path / "source"
    materialize(spec("CC-BY-4.0"), iter([{"source": "He have a small book.", "target": "He has a small book."}]), out, set(), 10, 100)
    result = prepare([out / "candidates.jsonl"], tmp_path / "prepared", allow_weak_train=True, schema=2)
    assert result["publication_allowed"] is False


def test_evaluation_only_record_never_enters_training(tmp_path):
    source = tmp_path / "pairs.jsonl"
    source.write_text(json.dumps({"source": "He have a small book.", "target": "He has a small book.",
                                  "category": "mixed", "license": "CC0-1.0", "evaluation_only": True}) + "\n")
    result = prepare([source], tmp_path / "out", allow_weak_train=True)
    assert result["counts"]["rejected:evaluation_only"] == 1
    assert result["splits"]["train"]["accepted"] == 0


def test_no_rows_is_a_failure_not_a_successful_import(tmp_path):
    with pytest.raises(ValueError, match="No eligible"):
        materialize(spec(), iter([]), tmp_path / "out", set(), 10, 100)


def test_wiked_reconstruction_uses_authors_inline_diff_format():
    assert wdiff_pair("There [-is-] {+are+} two books.") == ("There is two books.", "There are two books.")
    with pytest.raises(ValueError):
        wdiff_pair("There [-is a truncated record.")


def test_standard_spacy_contractions_are_restored_for_training():
    assert canonical("I ca n't believe he 's here .") == "I can't believe he's here."


def test_completed_run_cannot_hide_changed_export_behind_unchanged_manifest(tmp_path):
    model = tmp_path / "arm/model"
    model.mkdir(parents=True)
    receipt = tmp_path / "plan.json"
    receipt.write_text("{}")
    asset = model / "model.onnx"
    asset.write_bytes(b"model")
    (model / "evaluation.json").write_text("{}")
    (model / "manifest.json").write_text(json.dumps({"files": {"model.onnx": {"bytes": 5, "sha256": hash_file(asset)}}}))
    summary = {"plan_sha256": hash_file(receipt), "model_manifest_sha256": hash_file(model / "manifest.json"),
               "evaluation_sha256": hash_file(model / "evaluation.json")}
    verify_completed(tmp_path / "arm", summary, receipt)
    asset.write_bytes(b"other")
    with pytest.raises(ValueError, match="asset"):
        verify_completed(tmp_path / "arm", summary, receipt)


def test_bounded_import_never_reads_after_its_scheduled_prefix(tmp_path):
    def rows():
        yield {"source": "He have a small book.", "target": "He has a small book."}
        raise EOFError("unread tail of bounded gzip prefix")
    result = materialize(spec(), rows(), tmp_path / "out", set(), 1, 1)
    assert result["counts"]["scanned"] == 1


def test_real_gzip_tar_stream_does_not_require_seekable_members(tmp_path):
    content = b"S He have a small book .\nA 1 2|||R:VERB:SVA|||has|||-REQUIRED-|||NONE|||0\n"
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("corpus/train.m2")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    rows = list(archive_m2(path, ["corpus/train.m2"]))
    assert rows[0]["target"] == "He has a small book ."


def test_author_mirror_repository_urls_preserve_correct_commit_provenance():
    assert repository_slug("https://github.com/facebook/react") == "facebook/react"
    assert repository_slug("https://github.com/facebook/react.git") == "facebook/react"
    with pytest.raises(ValueError):
        repository_slug("https://example.org/facebook/react")


def test_dolt_repository_queries_share_one_total_bound(monkeypatch):
    from urllib.parse import parse_qs, urlsplit
    queries = []
    def request(url, timeout):
        query = parse_qs(urlsplit(url).query)["q"][0]
        queries.append(query)
        repo = "a/one" if "github.com/a/one" in query else "b/two"
        return io.BytesIO(json.dumps({"query_execution_status": "Success", "rows": [
            {"repo": "https://github.com/" + repo, "commit_hash": "a" * 40}]}).encode())
    monkeypatch.setattr("corpus_import.urlopen", request)
    rows = list(dolt_rows({"revision": "a" * 32, "repositories": ["c/three", "b/two", "a/one"],
                           "api_url": "https://example.org/query"}, 2))
    assert [row["repo"] for row in rows] == ["a/one", "b/two"]
    assert len(queries) == 2
    assert "LIMIT 2 OFFSET 0" in queries[0]
    assert "LIMIT 1 OFFSET 0" in queries[1]


def test_device_recovery_can_select_only_unfinished_arms():
    assert selected_arms({"wi": {}, "github-typo": {}, "wiked": {}}, True,
                         ["github-typo", "wiked", "combined"]) == ["github-typo", "wiked", "combined"]
    assert selected_arms({"wi": {}}, False, []) == ["wi"]
    with pytest.raises(ValueError):
        selected_arms({"wi": {}}, False, ["combined"])
    with pytest.raises(ValueError):
        selected_arms({"wi": {}}, False, ["wi", "wi"])
