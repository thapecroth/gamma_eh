"""Offline, focused recipe/transport/ledger invariants; no live provider calls."""
import json
from pathlib import Path
import sqlite3

import pytest

import verified_teacher as pipeline
from teacher_recipes import (derive_variant, guard_coverage, lexical_family,
                             load_registry, mutation_candidates, synthetic_split)


def args_for(tmp_path, recipe="agreement_subject", **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    if recipe:
        registry["recipes"] = [row for row in registry["recipes"] if row["id"] == recipe]
    registry_path = tmp_path / "recipes.json"
    registry_path.write_text(json.dumps(registry))
    args = pipeline.parser().parse_args(["--output", str(tmp_path / "run"), "--recipes", str(registry_path),
                                         "--families", "1", "--batch-size", "1", "--retries", "0", "--execute"])
    for key, value in overrides.items(): setattr(args, key, value)
    return args


class FixtureClient:
    def __init__(self, text="She is ready for the workshop.", change=None):
        self.text, self.change, self.calls = text, change, []
        self.targets = {}

    def payload(self, stage, user):
        return {"stage": stage, "user": user}

    def request_once(self, stage, user):
        self.calls.append((stage, user))
        if stage == "generate":
            value = {"contexts": [{"id": row["id"], "text": self.text} for row in user["contexts"]]}
        elif stage == "repair":
            value = {"repairs": [{"id": row["id"], "corrected": self.text,
                                   "source_is_grammatical": row["source"] == self.text} for row in user["sources"]]}
        else:
            value = {"reviews": [{"id": row["id"], **{field: row["source"] != row["target"]
                       if field in {"source_has_error", "correction_is_necessary"} else True
                       for field in pipeline.CRITIC_FIELDS}} for row in user["pairs"]]}
        if self.change: self.change(stage, value)
        return {"content": json.dumps(value), "usage": {"total_tokens": 10, "untrusted": "discard"},
                "attribution": {"provider": "claude", "auth_mode": "api-key", "account_id": "discard"},
                "billing_unknown": False, "error": None}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def all_admitted(args):
    return [row for path in [args.output / "candidates.jsonl", args.output / "diagnostic/dev.jsonl",
                             args.output / "diagnostic/test.jsonl"] for row in read_jsonl(path)]


def test_registry_examples_have_real_single_mutations_and_reverse_roundtrip():
    for recipe in load_registry()["recipes"]:
        for text in recipe["examples"]:
            variant = derive_variant(text, recipe, "fixture")
            if recipe["mutator"] == "identity":
                assert variant is None
                continue
            assert variant is not None, (recipe["id"], text)
            mutation = variant["mutation"]
            assert variant["source"] != text
            assert variant["source"][:mutation["start"]] + mutation["clean"] + variant["source"][mutation["source_end"]:] == text


@pytest.mark.parametrize("value", [[], 3, None, {"schema": 1, "recipes": [{"id": []}]}])
def test_registry_invalid_shapes_have_fixed_value_errors(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="recipe_registry"):
        load_registry(path)


def test_guard_coverage_reports_broad_rows_spacing_and_protected_source():
    assert guard_coverage("Does they need a ticket?", "Do they need a ticket?")["guarded"]
    assert guard_coverage("They have ate lunch.", "They have eaten lunch.")["guarded"]
    assert guard_coverage("We have many information.", "We have much information.")["guarded"]
    broad = guard_coverage("The box of cables are here.", "The box of cables is here.")
    assert broad["alignment"] and not broad["guarded"]
    assert not guard_coverage("She  are ready.", "She  is ready.")["alignment"]
    protected = guard_coverage("They rely of `the map`.", "They rely on `the map`.")
    assert not protected["guarded"]


def test_ambiguous_contexts_abstain_and_utf16_offsets_are_explicit():
    assert mutation_candidates("She read the note yesterday.", "agreement") == []
    assert mutation_candidates("She has apple juice.", "article_missing") == []
    assert mutation_candidates("We bought two ticket machines.", "numeric_count") == []
    assert mutation_candidates("We had much time.", "mass_quantifier") == []
    recipe = next(row for row in load_registry()["recipes"] if row["id"] == "numeric_count")
    text = "Note 🐾: we bought two tickets."
    variant = derive_variant(text, recipe, "fixture")
    mutation = variant["mutation"]
    assert mutation["clean_span_utf16"][0] == mutation["start"] + 1
    assert mutation["source_span_utf16"][1] == mutation["source_end"] + 1


def test_family_splits_are_stable_for_case_punctuation_and_numerals():
    family = lexical_family("Mara bought 2 tickets.")
    assert family == lexical_family("mara bought 3 tickets!")
    assert synthetic_split(family) in {"train", "dev", "test"}


def test_dry_run_large_plan_is_count_only_and_writes_nothing(tmp_path):
    args = args_for(tmp_path, recipe=None, execute=False, families=1_000_000)
    report = pipeline.run(args)
    assert report["generation_jobs"] == 1_000_000
    assert not args.output.exists()


@pytest.mark.parametrize("url", ["https://api.z.ai/v1", "http://localhost:8317/v1", "http://127.0.0.1:8317/v1?key=secret",
                                  "http://user:secret@127.0.0.1:8317/v1"])
def test_remote_and_alternative_endpoints_are_denied(url):
    with pytest.raises(ValueError): pipeline.local_endpoint(url)


def test_redirects_are_denied():
    with pytest.raises(pipeline.PipelineError, match="redirect_denied"):
        pipeline.NoRedirect().redirect_request(None, None, None, None, None, "https://elsewhere.test")


def test_blind_input_does_not_include_target_recipe_or_generator_metadata():
    row = {"row_id": "fixture", "source": "They have ate lunch.", "target": "They have eaten lunch.",
           "register": "informal", "context": "a fictional team chat", "recipe_id": "perfect_participle",
           "mutation": {"answer": "eaten"}, "generator_explanation": "expected answer"}
    user = pipeline.repair_input([row])
    assert set(user["sources"][0]) == {"id", "source", "register", "context"}
    assert "eaten" not in json.dumps(user)
    assert "fixture" not in user["sources"][0]["id"]


def test_repair_requests_never_expose_siblings_answers_or_class_ids(tmp_path):
    args = args_for(tmp_path)
    client = FixtureClient()
    pipeline.run(args, client, sleeper=lambda _: None)
    repairs = [user["sources"] for stage, user in client.calls if stage == "repair"]
    assert len(repairs) == 2
    assert len(repairs[0]) == len(repairs[1]) == 1
    assert repairs[1][0]["source"] != client.text
    assert client.text not in {row["source"] for row in repairs[1]}
    for bucket in repairs:
        for row in bucket:
            assert ":clean" not in row["id"] and ":variant" not in row["id"] and "family" not in row["id"]


def test_informal_clean_identity_and_original_screenshot_stay_local(tmp_path):
    # The real screenshot regression is a local fixture only, never a live prompt.
    phrase = "my cat is hungry."
    assert phrase not in " ".join(pipeline.PROMPTS.values())
    assert phrase not in json.dumps(load_registry())
    args = args_for(tmp_path, recipe="clean_general")
    client = FixtureClient(phrase)
    pipeline.run(args, client, sleeper=lambda _: None)
    assert all_admitted(args)[0]["source"] == phrase
    def capitalize(stage, value):
        if stage == "repair": value["repairs"][0]["corrected"] = "My cat is hungry."
    args2 = args_for(tmp_path / "second", recipe="clean_general")
    pipeline.run(args2, FixtureClient(phrase, capitalize), sleeper=lambda _: None)
    assert not all_admitted(args2)
    assert read_jsonl(args2.output / "quarantine.jsonl")[0]["reason"] == "clean_identity_changed"


def test_false_critic_boolean_quarantines_error_and_retains_clean_sibling(tmp_path):
    def reject(stage, value):
        if stage == "critic":
            for row in value["reviews"]:
                if row["id"].endswith(":variant"): row["meaning_preserved"] = False
    args = args_for(tmp_path)
    pipeline.run(args, FixtureClient(change=reject), sleeper=lambda _: None)
    assert len(all_admitted(args)) == 1
    assert all_admitted(args)[0]["category"] == "clean"
    assert read_jsonl(args.output / "quarantine.jsonl")[0]["reason"] == "critic_semantic_disagreement"


def test_row_schema_failure_is_local_and_boolean_strings_are_rejected(tmp_path):
    def malformed(stage, value):
        if stage == "repair":
            for row in value["repairs"]:
                if row["source_is_grammatical"] is False: row["source_is_grammatical"] = "false"
    args = args_for(tmp_path)
    pipeline.run(args, FixtureClient(change=malformed), sleeper=lambda _: None)
    assert len(all_admitted(args)) == 1
    assert read_jsonl(args.output / "quarantine.jsonl")[0]["reason"] == "repair_row_schema"


def test_id_join_disagreement_is_not_positional_or_silently_filtered(tmp_path):
    def wrong_id(stage, value):
        if stage == "repair": value["repairs"][0]["id"] += "-unrequested"
    args = args_for(tmp_path)
    pipeline.run(args, FixtureClient(change=wrong_id), sleeper=lambda _: None)
    assert not all_admitted(args)
    quarantine = read_jsonl(args.output / "quarantine.jsonl")
    assert {row["reason"] for row in quarantine} == {"repair_stage_stage_id_join"}
    assert all("source" in row for row in quarantine)


def test_completed_calls_and_immutable_shards_are_not_replayed_on_resume(tmp_path):
    args = args_for(tmp_path)
    client = FixtureClient()
    first = pipeline.run(args, client, sleeper=lambda _: None)
    shards = {path.name: path.read_bytes() for path in (args.output / "shards").iterdir()}
    second = pipeline.run(args, client, sleeper=lambda _: None)
    assert len(client.calls) == 4
    assert first["row_counts"] == second["row_counts"]
    assert shards == {path.name: path.read_bytes() for path in (args.output / "shards").iterdir()}


def test_crash_after_checkpointed_response_resumes_with_zero_replay_at_retries_zero(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    client = FixtureClient()
    original = pipeline.decode_stage
    def crash(stage, content, ids):
        if stage == "generate": raise KeyboardInterrupt
        return original(stage, content, ids)
    monkeypatch.setattr(pipeline, "decode_stage", crash)
    with pytest.raises(KeyboardInterrupt): pipeline.run(args, client, sleeper=lambda _: None)
    assert len(client.calls) == 1
    monkeypatch.setattr(pipeline, "decode_stage", original)
    pipeline.run(args, client, sleeper=lambda _: None)
    assert [stage for stage, _ in client.calls] == ["generate", "repair", "repair", "critic"]


def test_exclusion_all_fields_and_resume_hashes_are_immutable(tmp_path):
    text = "She is ready for the workshop."
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text(json.dumps({"source": "Other source.", "target": text, "references": ["Other reference."]}) + "\n")
    args = args_for(tmp_path, exclude=[excluded])
    client = FixtureClient(text)
    pipeline.run(args, client, sleeper=lambda _: None)
    assert len(client.calls) == 1
    assert not all_admitted(args)
    assert text not in json.dumps(client.calls[0][1])
    excluded.write_text(json.dumps({"source": "Changed input."}) + "\n")
    with pytest.raises(ValueError, match="immutable_run_inputs_changed"):
        pipeline.run(args, client, sleeper=lambda _: None)


def test_persistent_budget_stops_without_an_extra_call(tmp_path):
    args = args_for(tmp_path, max_requests=1)
    client = FixtureClient()
    first = pipeline.run(args, client, sleeper=lambda _: None)
    second = pipeline.run(args, client, sleeper=lambda _: None)
    assert first["stop_reason"] == second["stop_reason"] == "request_or_token_ceiling"
    assert len(client.calls) == 1
    assert first["request_count"] == 1


def test_fatal_route_failure_stops_all_jobs_and_preserves_pending_state(tmp_path):
    class BadRoute(FixtureClient):
        def request_once(self, stage, user):
            self.calls.append((stage, user))
            return {"error": "route_model_mismatch", "content": None, "usage": {"total_tokens": 8},
                    "attribution": {"provider": "claude", "auth_mode": "api-key"}, "billing_unknown": False}
    args = args_for(tmp_path, families=10)
    client = BadRoute()
    with pytest.raises(pipeline.PipelineError, match="route_model_mismatch"):
        pipeline.run(args, client, sleeper=lambda _: None)
    report = json.loads((args.output / "manifest.json").read_text())
    assert len(client.calls) == 1
    assert report["stop_reason"] == "route_model_mismatch"
    assert report["status_counts"] == {"pending": 1}


def test_global_conflict_revokes_effective_review_status_but_preserves_receipt(tmp_path):
    args = args_for(tmp_path)
    pipeline.run(args, FixtureClient(), sleeper=lambda _: None)
    db = sqlite3.connect(args.output / "ledger.sqlite3")
    db.execute("UPDATE rows SET status='quarantined',reason='conflicting_source' WHERE status='screened'")
    db.commit()
    db.close()
    args.snapshot_only = True
    pipeline.run(args, FixtureClient(), sleeper=lambda _: None)
    assert not all_admitted(args)
    rows = read_jsonl(args.output / "quarantine.jsonl")
    assert all(row["review_status"] == "quarantined" and row["screening_status"] == "screened" for row in rows)


def test_secrets_and_account_ids_are_not_saved(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setenv(args.api_key_env, "secret-key-should-not-be-saved")
    monkeypatch.setenv(args.attribution_token_env, "secret-monitor-should-not-be-saved")
    pipeline.run(args, FixtureClient(), sleeper=lambda _: None)
    for path in args.output.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert b"secret-key-should-not-be-saved" not in content
            assert b"secret-monitor-should-not-be-saved" not in content
            assert b"account_id" not in content
            assert b"untrusted" not in content
