import json
import uuid
from pathlib import Path
from urllib.error import URLError

import pytest

import judge as module
from calibrate_judge import load_fixture, measure
from judge import (FLAGS, Judge, endpoint, flags_reward, judge_spec, parse_verdicts,
                   response_verdicts, spec_hash, validate_calibration)


def flags(**changes):
    return {name: changes.get(name, True) for name in FLAGS}


def transport(payload):
    rows = json.loads(payload["input"])["pairs"]
    return ({"id": "resp-fixture-" + uuid.uuid4().hex, "model": payload["model"], "status": "completed",
             "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
             "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({
                 "verdicts": [{"id": row["id"], "flags": [flags()[name] for name in FLAGS]} for row in rows]})}]}]},
            {"X-CLIProxy-Provider": "codex", "X-CLIProxy-Auth-Mode": "subscription"})


def test_strict_flags_ids_and_reward_gates():
    assert flags_reward(flags()) == 1.
    assert flags_reward(flags(errors_resolved=False)) == .8
    assert flags_reward(flags(meaning_preserved=False)) == 0.
    assert flags_reward(flags(protected_text_preserved=False)) == 0.
    for value in (flags(meaning_preserved=1), {**flags(), "extra": True}, {"meaning_preserved": True}):
        with pytest.raises(ValueError): flags_reward(value)
    for values in ([], [{"id": "b", "flags": flags()}], [{"id": "a", "flags": flags()}] * 2):
        with pytest.raises(ValueError): parse_verdicts(json.dumps({"verdicts": values}), {"a"})
    with pytest.raises(ValueError, match="Duplicate JSON"):
        parse_verdicts('{"verdicts":[],"verdicts":[]}', set())
    ordered = [True, False, True, True, True]
    assert parse_verdicts(json.dumps({"verdicts": [{"id": "0", "flags": ordered}]}), {"0"})["0"] == flags(errors_resolved=False)
    for invalid in (flags(), ordered[:-1], ordered + [True], [True, 0, True, True, True]):
        with pytest.raises(ValueError, match="ordered booleans"):
            parse_verdicts(json.dumps({"verdicts": [{"id": "0", "flags": invalid}]}), {"0"})


def test_response_attribution_model_and_completed_status_fail_closed():
    payload = {"model": "gpt-6-luna", "input": '{"pairs":[{"id":"a"}]}' }
    body, headers = transport(payload)
    response_verdicts(body, headers, judge_spec(), {"a"})
    for changed in ({}, {**headers, "X-CLIProxy-Provider": "openrouter"}, {**headers, "X-CLIProxy-Auth-Mode": "api-key"}):
        with pytest.raises(ValueError, match="attribution"):
            response_verdicts(body, changed, judge_spec(), {"a"})
    for changed in ({**body, "status": "incomplete"}, {**body, "model": "different-model"}, {**body, "usage": {}}):
        with pytest.raises(ValueError): response_verdicts(changed, headers, judge_spec(), {"a"})


def test_cache_deduplicates_hashes_only_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMMA_JUDGE_API_KEY", "fixture-secret-key")
    monkeypatch.setenv("GAMMA_JUDGE_ATTRIBUTION_TOKEN", "fixture-secret-monitor")
    pairs = [("She have a scarf.", "She has a scarf.")] * 2
    with Judge(tmp_path, transport=transport) as client:
        results = client.judge(pairs)
        assert len(results) == 2 and client.stats()["requests"] == 1
        assert client.stats()["cache_hits"] == 1
        client.judge(pairs)
        assert client.stats()["requests"] == 1
    for path in tmp_path.iterdir():
        if path.is_file():
            text = path.read_text()
            assert "She have" not in text and "She has" not in text and "fixture-secret" not in text
    with Judge(tmp_path, transport=lambda payload: pytest.fail("cache must avoid transport")) as cached:
        assert cached.rewards(pairs) == [1., 1.]
    verdict = tmp_path / "verdicts.jsonl"
    value = json.loads(verdict.read_text())
    value["flags"]["meaning_preserved"] = False
    verdict.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="hash chain"):
        Judge(tmp_path, transport=transport)


def test_changed_spec_and_budget_exhaustion_reject_before_transport(tmp_path):
    with Judge(tmp_path, max_requests=1, max_pairs=1, transport=transport) as client:
        client.judge([("I saw a heron.", "I saw a heron.")])
        with pytest.raises(ValueError, match="budget exhausted"):
            client.judge([("We saw a heron.", "We saw a heron.")])
    with pytest.raises(ValueError, match="spec/settings changed"):
        Judge(tmp_path, spec=judge_spec(model="other-model"), transport=transport)


def test_malformed_batch_awards_no_partial_rewards_and_consumes_attempt_budget(tmp_path):
    def broken(payload):
        body, headers = transport(payload)
        body["output"][0]["content"][0]["text"] = "private provider error body"
        return body, headers
    with Judge(tmp_path, transport=broken, max_requests=1) as client:
        with pytest.raises(ValueError) as error:
            client.judge([("The robin sings.", "The robin sings.")])
        assert "private provider" not in str(error.value)
        assert client.stats()["requests"] == 1 and client.stats()["cached_pairs"] == 0
        assert not (tmp_path / "verdicts.jsonl").exists()


def test_endpoint_redirect_and_network_fail_closed(monkeypatch):
    assert endpoint("http://127.0.0.1:8317/v1").endswith("/v1/responses")
    for url in ("http://remote.test/v1", "https://key:secret@remote.test/v1", "https://remote.test/v1?key=secret"):
        with pytest.raises(ValueError): endpoint(url)
    assert module.NoRedirect().redirect_request(None, None, 302, "", {}, "https://remote.test") is None
    class Broken:
        def open(self, *unused, **kwargs): raise URLError("private upstream secret")
    monkeypatch.setattr(module, "build_opener", lambda *unused: Broken())
    with pytest.raises(ValueError) as error:
        module.live_response("http://127.0.0.1/v1/responses", {}, "fixture-key", 1)
    assert "private upstream" not in str(error.value)


def test_calibration_fixture_metrics_and_numeric_gate(tmp_path):
    _, rows = load_fixture(Path(__file__).resolve().parents[1] / "data/judge-calibration.json")
    assert len(rows) == 25
    rounds = [[{"flags": row["expected"]} for row in rows] for _ in range(2)]
    metrics = measure(rows, rounds)
    assert metrics["passed"] and metrics["criteria_agreement_by_flag"][0]["errors_resolved"] == 1.
    unsafe = next(index for index, row in enumerate(rows) if not row["expected"]["meaning_preserved"])
    rounds[0][unsafe] = {"flags": flags()}
    assert not measure(rows, rounds)["passed"]
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"passed": True, "judge_spec_sha256": spec_hash(judge_spec()),
        "criteria_agreement": [.99, .94], "repeat_agreement": 1., "unsafe_high_reward_count": 0,
        "critical_false_positive_flags": 0}))
    with pytest.raises(ValueError, match="numeric agreement"):
        validate_calibration(receipt, judge_spec())
