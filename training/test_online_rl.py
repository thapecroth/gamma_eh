"""Exercise generated-data admission and crash/restart behavior without live calls."""
import json
from pathlib import Path
import shutil
import uuid

import pytest

from judge import FLAGS, canonical, digest, judge_spec, ledger_read
from judge_objective import eligible, ONLINE_ORIGIN
from online_data import filter_pairs, generate, generation_spec, prepare_cycle, rows, screen_pairs
import online_rl as loop
from pairs import hash_file, normalized
from rl_objective import checkpoint_hashes
from test_compare_judge_rl import prepared
from test_judge import transport as judge_transport


def generated_transport():
    calls = []
    def transport(payload):
        if "requests" not in json.loads(payload["input"]): return judge_transport(payload)
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["strict"] is True
        calls.append(payload)
        values = []
        for request in json.loads(payload["input"])["requests"]:
            # Different contexts on subsequent cycles; unsupported edits are intentional.
            tail = f"a parcel beside shelf {len(calls)}{request['id']}."
            source, target = "She have " + tail, "She has " + tail
            if request["category"] == "clean": source = target
            elif request["category"] != "agreement": target = "She owns " + tail
            values.append({"id": request["id"], "source": source, "target": target,
                           "category": request["category"]})
        return ({"id": "resp-" + uuid.uuid4().hex, "model": payload["model"], "status": "completed",
                 "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                 "output": [{"type": "message", "content": [{"type": "output_text",
                     "text": json.dumps({"pairs": values})}]}]},
                {"X-CLIProxy-Provider": "codex", "X-CLIProxy-Auth-Mode": "subscription"})
    return transport, calls


def fake_trainer(scores=None, support=25):
    scores = scores or {"supervised": .6, "judge-rl": .7}
    calls = []
    def train(arguments):
        calls.append(arguments)
        value = lambda flag: arguments[arguments.index(flag) + 1]
        data, evaluation = Path(value("--data")), Path(value("--evaluation-dir"))
        output, checkpoint = Path(value("--output")), Path(value("--checkpoint"))
        output.mkdir(); checkpoint.mkdir()
        labels = json.loads((data / "labels.json").read_text())
        (checkpoint / "config.json").write_text(json.dumps({"model_type": "bert",
            "id2label": {str(i): label for i, label in enumerate(labels)},
            "label2id": {label: i for i, label in enumerate(labels)}}))
        (checkpoint / "model.safetensors").write_bytes(b"fixture" + output.parent.name.encode())
        score = scores.get(output.parent.name, .5)
        exported = {"split": "dev", "inference": {"inference_failures": 0},
                    "metrics": {"predicted_edits": support, "edit_precision": .96,
                                "clean_sentence_false_positive_rate": .01},
                    "by_origin": {"ErAConD": {"edit_f0_5": score}}}
        exported["diagnostic_by_origin"] = {"ErAConD": {"edit_f0_5": score}}
        initial = Path(value("--initial-checkpoint")) if "--initial-checkpoint" in arguments else None
        report = {"evaluation_mode": "development-only", "test_status": "deferred",
                  "exports": {name: exported for name in ("model.onnx", "model_quantized.onnx")},
                  "training_budget": {"optimizer_updates": 2, "examples_seen": len(rows(data / "train.jsonl")),
                      "rows_per_epoch": len(rows(data / "train.jsonl")), "optimizer_state": "restarted",
                      "selected_checkpoint_epoch": 1 if output.parent.name == "supervised" else 2},
                  "initialization": {"files": checkpoint_hashes(initial) if initial else {}},
                  "training_data_sha256": hash_file(data / "train.jsonl"),
                  "labels_sha256": hash_file(data / "labels.json"),
                  "evaluation_hashes": {split: hash_file(evaluation / f"{split}.jsonl") for split in ("dev", "test")}}
        (output / "evaluation.json").write_text(json.dumps(report))
    return train, calls


@pytest.fixture
def configured(tmp_path, monkeypatch):
    data, evaluation, fixture, calibration = prepared(tmp_path, monkeypatch)
    output = loop.ROOT / "artifacts" / ("test-online-" + uuid.uuid4().hex)
    args = loop.parser().parse_args(["--data", str(data), "--evaluation-dir", str(evaluation),
        "--fixture", str(fixture), "--calibration", str(calibration), "--output", str(output),
        "--new-pairs", "4", "--generation-batch-size", "4", "--rl-rows", "4", "--replay-rows", "4",
        "--epochs", "1", "--warmup-epochs", "1", "--device", "cpu", "--execute"])
    args.counterexamples = None
    try: yield args
    finally:
        if output.exists(): shutil.rmtree(output)


def test_generation_cache_and_response_identity(tmp_path):
    transport, calls = generated_transport()
    spec = generation_spec(judge_spec(), 4, 4, 42)
    result = generate(tmp_path, spec, transport)
    assert len(result) == 4 and all(eligible(row) for row in result)
    assert generate(tmp_path, spec, transport) == result and len(calls) == 1
    saved = tmp_path / "batch-0000.json"
    value = json.loads(saved.read_text()); value["pairs"][0]["target"] = "Corrupted text."
    value["pairs_sha256"] = digest(canonical(value["pairs"]))
    saved.write_text(canonical(value))
    with pytest.raises(ValueError, match="batch identity"): generate(tmp_path, spec, transport)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["duplicate-id", "category", "extra", "provider", "credential"])
def test_bad_generation_charges_attempt_and_retains_no_content(tmp_path, monkeypatch, change):
    monkeypatch.setenv("GAMMA_JUDGE_API_KEY", "fixture-private-token")
    transport, calls = generated_transport()
    def broken(payload):
        body, headers = transport(payload)
        values = json.loads(body["output"][0]["content"][0]["text"])
        if change == "duplicate-id": values["pairs"][1]["id"] = values["pairs"][0]["id"]
        elif change == "category": values["pairs"][0]["category"] = "clean"
        elif change == "extra": values["pairs"][0]["extra"] = True
        elif change == "provider": headers["X-CLIProxy-Provider"] = "openrouter"
        else: values["pairs"][0]["source"] = "fixture-private-token"
        body["output"][0]["content"][0]["text"] = json.dumps(values)
        return body, headers
    spec = generation_spec(judge_spec(), 4, 4, 42)
    with pytest.raises(ValueError) as error: generate(tmp_path, spec, broken)
    assert "fixture-private-token" not in str(error.value)
    assert len(ledger_read(tmp_path / "requests.jsonl")[0]) == 1
    assert not list(tmp_path.glob("batch-*.json"))
    with pytest.raises(ValueError, match="no completed batch"): generate(tmp_path, spec, transport)
    assert len(calls) == 1


def test_fixed_label_holdout_duplicate_and_wordpiece_filters(tmp_path):
    transport, _ = generated_transport()
    candidates = generate(tmp_path, generation_spec(judge_spec(), 4, 4, 42), transport)
    selected, rejected = filter_pairs(candidates, {"KEEP", "REPLACE:has"}, 1, set(), set())
    assert len(selected) == 2 and rejected == {"unsupported_edit_or_context": 2}
    keys = {normalized(candidates[0]["target"]).casefold()}
    selected, rejected = filter_pairs(candidates, {"KEEP", "REPLACE:has"}, 1, keys,
                                      {normalized(candidates[3]["source"]).casefold()})
    assert not selected and rejected["heldout_overlap"] == rejected["duplicate_source"] == 1
    selected, rejected = filter_pairs([candidates[0]], {"KEEP", "REPLACE:has"}, 1, set(), set(),
                                      lambda *_args, **_kwargs: {"input_ids": list(range(100))})
    assert not selected and rejected == {"unsupported_edit_or_context": 1}


def test_screening_requires_every_boolean_and_manifest_records_weak_license(tmp_path, configured):
    transport, _ = generated_transport()
    candidates = generate(tmp_path / "generation", generation_spec(judge_spec(), 4, 4, 42), transport)
    selected, _ = filter_pairs(candidates, {"KEEP", "REPLACE:has"}, 1, set(), set())
    def reject(payload):
        body, headers = judge_transport(payload)
        text = json.loads(body["output"][0]["content"][0]["text"])
        text["verdicts"][0]["flags"][1] = False
        body["output"][0]["content"][0]["text"] = json.dumps(text)
        return body, headers
    admitted, stats = screen_pairs(tmp_path / "screening", selected, judge_spec(), reject)
    assert len(admitted) == 1 and stats["rejected"] == 1
    manifest = prepare_cycle(configured.data, tmp_path / "prepared", admitted)
    assert manifest["publication_allowed"] is False
    assert manifest["licenses"]["provider-terms-unverified"] == 1
    assert manifest["training_origins"][ONLINE_ORIGIN] == 1
    assert manifest["source_provenance"][-1]["rows"] == 1


@pytest.mark.parametrize("field", ["origin", "license", "machine_generated_original", "human_reviewed",
                                  "publication_allowed", "generation_model", "generation_request_id", "generator_spec_sha256"])
def test_online_reward_admission_needs_complete_provenance(tmp_path, field):
    transport, _ = generated_transport()
    row = generate(tmp_path, generation_spec(judge_spec(), 4, 4, 42), transport)[0]
    row.pop(field)
    assert not eligible(row)


def test_continuations_use_same_data_and_accept_qualified_best(configured):
    transport, _ = generated_transport(); trainer, calls = fake_trainer()
    result = loop.run(configured, transport, trainer)
    assert result["state"]["accepted"]["checkpoint"].endswith("judge-rl/checkpoint")
    assert len(calls) == 3
    for flag in ("--data", "--initial-checkpoint", "--epochs", "--seed", "--batch-size"):
        assert calls[1][calls[1].index(flag) + 1] == calls[2][calls[2].index(flag) + 1]
    assert "--development-only" in calls[1] and "--development-only" in calls[2]
    assert result["test_inference"] is False and result["publication_allowed"] is False


@pytest.mark.parametrize("scores,support,accepted", [({"supervised": .7, "judge-rl": .6}, 25, "supervised/checkpoint"),
    ({"supervised": .4, "judge-rl": .3}, 25, "bootstrap-0000/checkpoint"),
    ({"supervised": .9, "judge-rl": .95}, 0, "bootstrap-0000/checkpoint")])
def test_supervised_win_inferior_or_zero_edit_candidates(configured, scores, support, accepted):
    transport, _ = generated_transport(); trainer, _ = fake_trainer(scores, support)
    assert loop.run(configured, transport, trainer)["state"]["accepted"]["checkpoint"].endswith(accepted)


def test_unqualified_research_improvement_continues_next_cycle_without_release_acceptance(configured):
    transport, _ = generated_transport(); trainer, calls = fake_trainer(support=0)
    first = loop.run(configured, transport, trainer)
    assert first["state"]["accepted"]["checkpoint"].endswith("bootstrap-0000/checkpoint")
    cursor = first["state"]["training"]["checkpoint"]
    assert cursor.endswith("judge-rl/checkpoint")
    second = loop.run(configured, transport, trainer)
    assert second["state"]["training"]["checkpoint"] == cursor  # Equal scores retain prior weights.
    assert calls[3][calls[3].index("--initial-checkpoint") + 1] == cursor
    assert calls[4][calls[4].index("--initial-checkpoint") + 1] == cursor


def test_cumulative_reservations_survive_restart_and_ledger_cleanup_is_rejected(configured):
    configured.max_total_requests = 3  # One generation, one screening, one RL request upper bound.
    transport, calls = generated_transport(); trainer, _ = fake_trainer()
    first = loop.run(configured, transport, trainer)
    assert first["state"]["requests_reserved"] == 3
    second = loop.run(configured, transport, trainer)
    assert second["stop_reason"] == "request-budget" and len(calls) == 1
    ledger = configured.output / "cycles/cycle-000000/generation/requests.jsonl"
    ledger.unlink()
    with pytest.raises(ValueError, match="ledger changed or disappeared"): loop.run(configured, transport, trainer)


def test_interrupted_bootstrap_uses_fresh_numbered_attempt(configured):
    trainer, calls = fake_trainer(); transport, _ = generated_transport()
    def interrupt(_arguments): raise loop.StopRequested()
    assert loop.run(configured, transport, interrupt)["status"] == "interrupted-partial-retained"
    result = loop.run(configured, transport, trainer)
    assert (configured.output / "bootstrap-0000").exists()
    assert calls[0][calls[0].index("--checkpoint") + 1].endswith("bootstrap-0001/checkpoint")
    assert result["state"]["next_cycle"] == 1


def test_partial_cycle_reservation_not_refunded_on_resume(configured):
    configured.max_total_requests = 6
    configured.cycles = 2
    trainer, _ = fake_trainer(); transport, calls = generated_transport()
    def interrupt(payload):
        result = transport(payload)
        (configured.output / "STOP").touch()
        return result
    assert loop.run(configured, interrupt, trainer)["status"] == "interrupted-partial-retained"
    (configured.output / "STOP").unlink()
    result = loop.run(configured, transport, trainer)
    assert result["state"]["requests_reserved"] == 6 and len(calls) == 2
    assert json.loads((configured.output / "cycles/cycle-000000/abandonment.json").read_text())["status"] == "abandoned-partial"
    assert result["state"]["next_cycle"] == 2
    assert loop.run(configured, transport, trainer)["stop_reason"] == "request-budget"


def test_stop_file_prevents_bootstrap_or_new_calls(configured):
    configured.output.mkdir(parents=True); (configured.output / "STOP").touch()
    trainer, calls = fake_trainer(); transport, generated = generated_transport()
    assert loop.run(configured, transport, trainer)["status"] == "interrupted-partial-retained"
    assert not calls and not generated


def test_resume_rejects_result_or_checkpoint_tampering(configured):
    transport, _ = generated_transport(); trainer, _ = fake_trainer()
    result = loop.run(configured, transport, trainer)
    checkpoint = Path(result["state"]["accepted"]["checkpoint"]) / "model.safetensors"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Accepted checkpoint changed"): loop.run(configured, transport, trainer)


def test_cycle_failure_retains_incumbent_and_stops_at_limit(configured):
    configured.max_failures = 1
    transport, _ = generated_transport(); trainer, _ = fake_trainer()
    def failing(arguments):
        if "--initial-checkpoint" in arguments: raise ValueError("fixture continuation failure")
        trainer(arguments)
    result = loop.run(configured, transport, failing)
    assert result["stop_reason"] == "failure-limit"
    assert result["state"]["accepted"]["checkpoint"].endswith("bootstrap-0000/checkpoint")
    assert sum(record["rows"] for record in result["state"]["replay"]) == 2
    resumed = loop.run(configured, transport, trainer)
    assert sum(record["rows"] for record in resumed["state"]["replay"]) == 4
    data = configured.output / "cycles/cycle-000001/prepared/train.jsonl"
    assert len(rows(data)) == 5  # Anchor plus screened data retained from both attempts.


def test_smaller_judge_batches_have_matching_conservative_request_caps(configured):
    configured.new_pairs = configured.rl_rows = 32
    configured.replay_rows = 32
    configured.generation_batch_size = 4
    configured.judge_batch_size = 8
    spec = loop.plan(configured)
    assert spec["rl_max_requests"] == 8 and spec["worst_cycle_requests"] == 20
    arguments = loop.command(spec, configured.data, configured.output / "stage", None,
                             "llm-judge-reinforce", 42, 1, configured.data / "train.jsonl")
    assert arguments[arguments.index("--judge-batch-size") + 1] == "8"


def test_calibration_overlap_rejected_before_bootstrap_or_remote_calls(configured):
    path = configured.data / "train.jsonl"
    row = rows(path)[0]
    row["source"] = json.loads(configured.fixture.read_text())["cases"][0]["source"]
    path.write_text(canonical(row) + "\n")
    manifest_path = configured.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["sha256"] = hash_file(path)
    manifest_path.write_text(canonical(manifest))
    trainer, calls = fake_trainer(); transport, generated = generated_transport()
    with pytest.raises(ValueError, match="calibration source/candidate overlaps"):
        loop.run(configured, transport, trainer)
    assert not configured.output.exists() and not calls and not generated
