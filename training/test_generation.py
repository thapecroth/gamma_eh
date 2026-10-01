from argparse import Namespace
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

import pytest

import generate_llm
from generate_llm import endpoint, generate, plan_jobs
from import_c4 import materialize
from merge_teacher import merge
from pairs import group_id, validate_pair
from prepare_pairs import prepare


def args_for(path, **overrides):
    values = dict(base_url="http://localhost:8000/v1", provider="openai-compatible", model="test-teacher",
                  pairs=4, batch_size=2, concurrency=2, seed=42, max_tokens=1024,
                  retries=1, timeout=10, api_key_env="GAMMA_TEST_KEY", output=path, execute=True)
    values.update(overrides)
    return Namespace(**values)


def test_plan_is_deterministic_and_bounds_partial_batch():
    a = plan_jobs(81, 20, 42)
    assert a == plan_jobs(81, 20, 42)
    assert sum(job["count"] for job in a) == 81
    assert a[-1]["count"] == 1
    assert a[3]["category"] == "clean"
    assert len({job["nonce"] for job in a}) == len(a)


def test_validator_rejects_secret_pattern_wrong_clean_and_schema():
    for row in [{"source": "Contact me at person@example.test", "target": "Hello there."},
                {"source": "She have a book.", "target": "She has a book.", "category": "clean"},
                {"source": "Same text.", "target": "Same text.", "category": "agreement"},
                {"source": 3, "target": "A book."}]:
        with pytest.raises(ValueError): validate_pair(row)


def test_endpoint_rejects_embedded_keys_and_plaintext_remote():
    assert endpoint("http://localhost:8000/v1", "openai-compatible").endswith('/v1/chat/completions')
    assert endpoint("https://api.example.test/v1", "anthropic").endswith('/v1/messages')
    for value in ["http://remote.example.test/v1", "https://user:secret@example.test/v1", "https://api.example.test/v1?key=secret"]:
        with pytest.raises(ValueError): endpoint(value, "openai-compatible")


def test_optional_effort_json_mode_and_bounded_output_validation_retry(monkeypatch):
    calls = []
    class Response:
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, _): return json.dumps(self.value).encode()
    def request(value, **_):
        calls.append(json.loads(value.data))
        pairs = [{"source": "She have a book.", "target": "She has a book.", "category": "agreement"}]
        content = json.dumps({"pairs": [] if len(calls) == 1 else pairs})
        return Response({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})
    monkeypatch.setattr(generate_llm, "urlopen", request)
    monkeypatch.setattr(generate_llm.time, "sleep", lambda *_: None)
    settings = {"provider": "openai-compatible", "model": "test", "endpoint": "http://localhost/v1/chat/completions",
                "max_tokens": 200, "retries": 1, "timeout": 1, "reasoning_effort": "low", "json_mode": True}
    rows, _ = generate_llm.request_teacher({"count": 1, "category": "agreement", "domain": "test", "nonce": "test"}, settings, "test", "")
    assert len(calls) == 2
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert len(rows) == 1


def test_optional_generation_flags_isolate_resume_fingerprints(tmp_path, monkeypatch):
    def teacher(job, *_):
        return [{"source": f"She have book {i}.", "target": f"She has book {i}.", "category": job["category"]} for i in range(2)], {}
    monkeypatch.setattr(generate_llm, "request_teacher", teacher)
    generate(args_for(tmp_path, pairs=2))
    config = json.loads((tmp_path / "manifest.json").read_text())["config"]
    assert "reasoning_effort" not in config and "json_mode" not in config
    with pytest.raises(ValueError, match="settings changed"):
        generate(args_for(tmp_path, pairs=2, reasoning_effort="low", json_mode=True))


def test_generation_resumes_without_duplicate_completed_requests(tmp_path, monkeypatch):
    calls = []
    def teacher(job, settings, prompt, key):
        calls.append(job["id"])
        assert "preserve" in prompt.lower()
        rows = [{"source": f"She have a book for task {job['id']} example {i}.",
                 "target": f"She has a book for task {job['id']} example {i}.",
                 "category": job["category"]} for i in range(job["count"])]
        return rows, {"prompt_tokens": 10, "untrusted": "do-not-store"}
    monkeypatch.setattr(generate_llm, "request_teacher", teacher)
    monkeypatch.setenv("GAMMA_TEST_KEY", "test-only-key-never-save")
    args = args_for(tmp_path)
    generate(args)
    first = (tmp_path / "candidates.jsonl").read_bytes()
    generate(args)
    assert len(calls) == 2
    assert (tmp_path / "candidates.jsonl").read_bytes() == first
    assert len(first.splitlines()) == 4
    for path in tmp_path.iterdir():
        if path.is_file():
            assert b"test-only-key-never-save" not in path.read_bytes()
            assert b"do-not-store" not in path.read_bytes()
    with pytest.raises(ValueError, match="settings changed"):
        generate(args_for(tmp_path, model="another-teacher"))


def test_failed_jobs_are_retryable(tmp_path, monkeypatch):
    def fail(*_): raise RuntimeError("private provider body must not be stored")
    monkeypatch.setattr(generate_llm, "request_teacher", fail)
    with pytest.raises(RuntimeError): generate(args_for(tmp_path, pairs=2))
    manifest = (tmp_path / "manifest.json").read_text()
    assert "private provider" not in manifest
    assert json.loads(manifest)["status_counts"] == {"failed": 1}
    def success(job, *_):
        return [{"source": f"She have book {i}.", "target": f"She has book {i}.", "category": job["category"]} for i in range(2)], {}
    monkeypatch.setattr(generate_llm, "request_teacher", success)
    generate(args_for(tmp_path, pairs=2))
    assert json.loads((tmp_path / "manifest.json").read_text())["status_counts"] == {"done": 1}


def test_only_one_generator_can_use_a_run_directory(tmp_path):
    path = tmp_path / "run-lock.sqlite3"
    first = generate_llm.acquire_run_lock(path)
    try:
        with pytest.raises(RuntimeError, match="Another generator"):
            generate_llm.acquire_run_lock(path)
    finally:
        first.close()
    next_run = generate_llm.acquire_run_lock(path)
    next_run.close()


def test_interrupted_generation_keeps_a_snapshot_and_offline_export_never_calls_teacher(tmp_path, monkeypatch):
    calls = []
    def teacher(job, *_):
        calls.append(job["id"])
        if job["id"] == 1: raise KeyboardInterrupt()
        return [{"source": f"She have book {job['id']} example {i}.",
                 "target": f"She has book {job['id']} example {i}.",
                 "category": job["category"]} for i in range(job["count"])], {"total_tokens": 10}
    monkeypatch.setattr(generate_llm, "request_teacher", teacher)
    with pytest.raises(KeyboardInterrupt):
        generate(args_for(tmp_path, concurrency=1, snapshot_every=1))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["accepted_candidates"] == 2
    assert manifest["unfinished_requests"] == 1
    assert manifest["publication_allowed"] is False
    first = (tmp_path / "candidates.jsonl").read_bytes()
    generate(args_for(tmp_path, concurrency=1, execute=False, snapshot_only=True))
    assert calls == [0, 1]
    assert (tmp_path / "candidates.jsonl").read_bytes() == first
    exported = json.loads((tmp_path / "manifest.json").read_text())
    assert exported["status_counts"] == {"done": 1, "running": 1}
    assert exported["completed_request_usage"] == {"total_tokens": 10}


def test_teacher_merge_removes_conflicts_and_verifies_provenance(tmp_path, monkeypatch):
    def teacher(job, settings, *_):
        seed = settings["seed"]
        return [{"source": "He have a book.", "target": "He has a book." if seed == 42 else "He had a book.",
                 "category": job["category"]},
                {"source": f"She have book {seed}.", "target": f"She has book {seed}.", "category": job["category"]}], {}
    monkeypatch.setattr(generate_llm, "request_teacher", teacher)
    a, b = tmp_path / "a", tmp_path / "b"
    generate(args_for(a, pairs=2, seed=42))
    generate(args_for(b, pairs=2, seed=43))
    report = merge([a, b], tmp_path / "merged")
    assert report["counts"]["accepted"] == 2
    assert report["counts"]["conflicting_sources_removed"] == 1
    assert report["publication_allowed"] is False
    merge([b, a], tmp_path / "reversed")
    assert (tmp_path / "merged/candidates.jsonl").read_bytes() == (tmp_path / "reversed/candidates.jsonl").read_bytes()
    assert (tmp_path / "merged/manifest.json").read_bytes() == (tmp_path / "reversed/manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"): merge([a], tmp_path / "merged")
    original = (a / "manifest.json").read_text()
    tampered_manifest = json.loads(original)
    tampered_manifest["fingerprint"] = "0" * 64
    (a / "manifest.json").write_text(json.dumps(tampered_manifest))
    with pytest.raises(ValueError, match="fingerprint mismatch"): merge([a], tmp_path / "bad-config")
    (a / "manifest.json").write_text(original)
    with (a / "candidates.jsonl").open("a") as stream: stream.write('{}\n')
    with pytest.raises(ValueError, match="hash mismatch"): merge([a], tmp_path / "tampered")


def test_import_is_bounded_and_preserves_attribution(tmp_path):
    rows = [{"input": f"She have a book for project {i}.", "output": f"She has a book for project {i}."} for i in range(10)]
    manifest = materialize(rows, tmp_path, limit=3, max_scanned=5)
    assert manifest["counts"]["accepted"] == 3
    assert manifest["counts"]["scanned"] == 3
    assert "Stahlberg" in manifest["attribution"]
    records = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert all(row["review_status"] == "unreviewed" for row in records)
    with pytest.raises(ValueError, match="already exists"): materialize(rows, tmp_path, 3, 5)


def test_import_closes_early_stopped_reader(tmp_path):
    closed = []
    def rows():
        try:
            for i in range(100):
                yield {'input': f'She have book {i}.', 'output': f'She has book {i}.'}
        finally:
            closed.append(True)
    materialize(rows(), tmp_path, limit=2, max_scanned=10)
    assert closed == [True]


def test_weak_pairs_never_leak_reviewed_heldout_targets(tmp_path):
    target = next(f"She has a book for project {i}." for i in range(1000)
                  if int(group_id(f"She has a book for project {i}.")[:8], 16) % 100 >= 90)
    inputs = tmp_path / "input.jsonl"
    records = [
        {"source": target.replace("has", "have"), "target": target, "category": "agreement", "review_status": "human-reviewed", "license": "CC0-1.0"},
        {"source": target.replace("a book", "an book"), "target": target, "category": "articles", "review_status": "unreviewed", "license": "CC0-1.0"},
        {"source": "She have a pen.", "target": "She has a pen.", "category": "agreement", "review_status": "unreviewed", "license": "CC0-1.0"},
        {"source": "Hello , friend.", "target": "Hello, friend.", "category": "punctuation", "review_status": "unreviewed", "license": "CC0-1.0"},
    ]
    inputs.write_text("".join(json.dumps(row) + "\n" for row in records))
    manifest = prepare([inputs], tmp_path / "prepared", allow_weak_train=True)
    assert manifest["counts"]["weak_heldout_group_dropped"] == 1
    assert manifest["counts"]["rejected"] == 1
    train = (tmp_path / "prepared/train.jsonl").read_text()
    assert target not in train
    assert "weak-supervision" in train
    assert "human-reviewed" in (tmp_path / "prepared/test.jsonl").read_text()


def test_actual_http_adapter_retries_rate_limit_and_parses_both_protocols(monkeypatch):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, body))
            if len(calls) == 1:
                self.send_response(429)
                self.send_header('Retry-After', '0')
                self.end_headers()
                return
            content = json.dumps({'pairs': [{'source': 'She have a book.', 'target': 'She has a book.', 'category': 'agreement'}]})
            if self.path.endswith('/messages'):
                assert self.headers['anthropic-version'] == '2023-06-01'
                response = {'content': [{'type': 'text', 'text': content}], 'stop_reason': 'end_turn', 'usage': {'output_tokens': 20}}
            else:
                assert self.headers['Authorization'] == 'Bearer test-only-key'
                response = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 20}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        job = {'count': 1, 'category': 'agreement', 'domain': 'work email', 'nonce': 'test-1'}
        for provider in ['openai-compatible', 'anthropic']:
            settings = {'provider': provider, 'model': 'test-teacher', 'max_tokens': 100,
                        'endpoint': endpoint(f'http://127.0.0.1:{server.server_port}/v1', provider), 'retries': 1, 'timeout': 5}
            rows, usage = generate_llm.request_teacher(job, settings, 'A test prompt.', 'test-only-key')
            assert rows[0]['target'] == 'She has a book.'
            assert usage
        assert len(calls) == 3
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_teacher_merge_metadata_ties_are_canonical(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(generate_llm, "request_teacher", lambda job, settings, *_: (
        [{"source": "She have a pen.", "target": "She has a pen.", "category": job["category"]}], {}))
    a, b = tmp_path / "a", tmp_path / "b"
    generate(args_for(a, pairs=1))
    shutil.copytree(a, b)
    row = json.loads((b / "candidates.jsonl").read_text())
    row["job_id"] = 7
    (b / "candidates.jsonl").write_text(json.dumps(row) + "\n")
    manifest = json.loads((b / "manifest.json").read_text())
    manifest["candidates_sha256"] = generate_llm.hash_file(b / "candidates.jsonl")
    (b / "manifest.json").write_text(json.dumps(manifest))
    merge([a, b], tmp_path / "forward")
    merge([b, a], tmp_path / "backward")
    for name in ["candidates.jsonl", "manifest.json"]:
        assert (tmp_path / "forward" / name).read_bytes() == (tmp_path / "backward" / name).read_bytes()


def test_rejected_legacy_resume_cannot_save_the_wrong_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_llm, "request_teacher", lambda job, settings, *_: (
        [{"source": "She have a pen.", "target": "She has a pen.", "category": job["category"]}], {}))
    run = tmp_path / "run"
    generate(args_for(run, pairs=1))
    original = (run / "prompt.txt").read_text()
    (run / "prompt.txt").unlink()
    changed = tmp_path / "changed-prompt.txt"
    changed.write_text(original + "Different prompt.\n")
    wrong = args_for(run, pairs=1)
    wrong.prompt_file = changed
    with pytest.raises(ValueError, match="settings changed"): generate(wrong)
    assert not (run / "prompt.txt").exists()
    generate(args_for(run, pairs=1))
    assert (run / "prompt.txt").read_text() == original


def test_nested_cached_token_usage_is_numeric_and_resumable():
    from collections import Counter
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
             "prompt_tokens_details": {"cached_tokens": 8},
             "completion_tokens_details": {"reasoning_tokens": 5},
             "input_tokens": -1, "output_tokens": True, "metadata": "ignored"}
    totals = Counter()
    for _ in range(2): totals.update(generate_llm.token_usage_counts(usage))
    assert totals == {"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60,
                      "prompt_tokens_details.cached_tokens": 16, "completion_tokens_details.reasoning_tokens": 10}
    assert generate_llm.token_usage_counts(None) == {}
