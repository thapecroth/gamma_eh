"""Resumable teacher generation; defaults to a dry run and never stores credentials."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pairs import hash_file, normalized, validate_pair

PROMPT_PATH = Path(__file__).parent / "prompts/teacher.txt"
DOMAINS = ["work email", "team chat", "personal message", "travel planning", "school essay",
           "customer support", "technical documentation", "product review", "job application",
           "meeting notes", "recipe discussion", "community forum", "sports discussion",
           "science explanation", "creative writing", "event invitation"]
ERRORS = ["agreement", "articles", "prepositions", "verb-tense", "spelling", "punctuation", "word-order"]


class TeacherError(RuntimeError):
    """A stable, safe error reason; never contains a provider response."""
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def iter_jobs(total, batch_size, seed):
    rng = random.Random(seed)
    for i in range(math.ceil(total / batch_size)):
        count = min(batch_size, total - i * batch_size)
        category = "clean" if i % 4 == 3 else ERRORS[(i - i // 4) % len(ERRORS)]
        yield {"id": i, "count": count, "category": category,
                     "domain": DOMAINS[rng.randrange(len(DOMAINS))],
                     "nonce": f"{seed}-{i}-{rng.getrandbits(64):016x}"}


def plan_jobs(total, batch_size, seed):
    return list(iter_jobs(total, batch_size, seed))


def endpoint(base_url, provider):
    parsed = urlparse(base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain credentials, query parameters, or fragments")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Endpoint must be an HTTP(S) URL")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use HTTPS for a remote teacher endpoint")
    return base_url.rstrip("/") + ("/messages" if provider == "anthropic" else "/chat/completions")


def request_teacher(job, settings, system_prompt, api_key):
    user = json.dumps({"count": job["count"], "category": job["category"],
                       "domain": job["domain"], "diversity_nonce": job["nonce"]})
    if settings["provider"] == "anthropic":
        payload = {"model": settings["model"], "system": system_prompt,
                   "messages": [{"role": "user", "content": user}], "max_tokens": settings["max_tokens"]}
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    else:
        payload = {"model": settings["model"], "messages": [{"role": "system", "content": system_prompt},
                                                               {"role": "user", "content": user}],
                   settings.get("max_token_field", "max_tokens"): settings["max_tokens"]}
        if settings.get("reasoning_effort"):
            payload["reasoning_effort"] = settings["reasoning_effort"]
        if settings.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}
        # No provider-specific JSON mode/temperature assumption. Some reasoning
        # gateways reject these parameters; structural validation still applies.
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    headers["Content-Type"] = "application/json"
    request = Request(settings["endpoint"], data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(settings["retries"] + 1):
        try:
            with urlopen(request, timeout=settings["timeout"]) as response:
                raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise TeacherError("output_size_limit")
            result = json.loads(raw)
            if settings["provider"] == "anthropic":
                text = "".join(block.get("text", "") for block in result["content"] if block.get("type") == "text")
                if result.get("stop_reason") == "max_tokens": raise TeacherError("output_truncated")
            else:
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length": raise TeacherError("output_truncated")
                text = choice["message"]["content"]
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("pairs"), list): raise TeacherError("output_schema")
            if len(parsed["pairs"]) != job["count"]: raise TeacherError("output_pair_count")
            for candidate in parsed["pairs"]:
                try:
                    validate_pair(candidate, expected_category=job["category"])
                except ValueError as error:
                    # validate_pair reasons are fixed enums; never include row contents.
                    raise TeacherError("pair_" + str(error)) from None
            return parsed["pairs"], result.get("usage", {})
        except HTTPError as error:
            # Never log response bodies/headers; providers may echo sensitive content.
            retry = error.code in {408, 429, 500, 502, 503, 504}
            delay = error.headers.get("Retry-After", "")
            if not retry or attempt == settings["retries"]: raise TeacherError("http_" + str(error.code)) from None
            time.sleep(min(60, float(delay) if delay.isdigit() else 2 ** attempt + random.random()))
        except (URLError, TimeoutError):
            if attempt == settings["retries"]: raise TeacherError("connection_failed") from None
            time.sleep(min(60, 2 ** attempt + random.random()))
        except (TeacherError, ValueError, KeyError, TypeError, IndexError) as error:
            # Retry incomplete JSON, truncation, and invalid rows within the same
            # bounded attempt budget as transport failures.
            if attempt == settings["retries"]:
                reason = error.reason if isinstance(error, TeacherError) else "output_validation"
                raise TeacherError(reason) from None
            time.sleep(min(60, 2 ** attempt + random.random()))


def open_ledger(path, fingerprint):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, usage TEXT, error TEXT);
      CREATE TABLE IF NOT EXISTS pairs (id TEXT PRIMARY KEY, source TEXT NOT NULL, target TEXT NOT NULL, category TEXT NOT NULL, clean_group TEXT NOT NULL, job_id INTEGER NOT NULL, source_key TEXT NOT NULL UNIQUE);
    """)
    existing = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
    if existing and existing[0] != fingerprint:
        db.close()
        raise ValueError("Run settings changed. Choose a new output directory instead of mixing runs.")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
    db.commit()
    return db


def acquire_run_lock(path):
    lock = sqlite3.connect(path, timeout=0)
    try:
        lock.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError:
        lock.close()
        raise RuntimeError("Another generator is using this output directory") from None
    return lock


def materialize_snapshot(db, output, config, fingerprint, rejected):
    """Export committed rows without calling the provider, including incomplete runs."""
    partial = output / 'candidates.partial.jsonl'
    with partial.open("w") as stream:
        for pair, source, target, category, group, job_id in db.execute(
                "SELECT id,source,target,category,clean_group,job_id FROM pairs ORDER BY id"):
            row = {"pair_id": pair, "source": source, "target": target, "category": category,
                   "clean_group": group, "job_id": job_id, "review_status": "unreviewed",
                   "origin": "llm-teacher", "model": config["model"],
                   "license": "provider-terms-unverified", "prompt_sha256": config["prompt_sha256"]}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.replace(output / 'candidates.jsonl')
    statuses = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
    planned = math.ceil(config["pairs"] / config["batch_size"])
    usage = Counter()
    for (value,) in db.execute("SELECT usage FROM jobs WHERE status='done' AND usage IS NOT NULL"):
        usage.update(json.loads(value))
    manifest = {"schema": 1, "config": config, "fingerprint": fingerprint,
                "status_counts": statuses, "planned_requests": planned,
                "unfinished_requests": planned - statuses.get("done", 0),
                "accepted_candidates": db.execute("SELECT count(*) FROM pairs").fetchone()[0],
                "candidates_sha256": hash_file(output / 'candidates.jsonl'),
                "rejections_this_invocation": dict(rejected),
                "completed_request_usage": dict(usage),
                "usage_scope": "Successful requests only; failed/interrupted attempts may also consume quota.",
                "quality": "Weak labels; grammar correctness not independently verified",
                "publication_allowed": False,
                "dataset_license": "Provider terms must be checked before publication"}
    temporary = output / 'manifest.partial.json'
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / 'manifest.json')
    return manifest


def generate(args):
    if not 1 <= args.concurrency <= 16 or not 1 <= args.batch_size <= 50 or args.pairs < 1:
        raise ValueError("Use positive pairs, batch size1..50, and concurrency1..16")
    if args.retries < 0 or args.timeout <= 0 or args.max_tokens < 1 or getattr(args, "snapshot_every", 10) < 1:
        raise ValueError("Retries must be nonnegative; timeout and max-tokens must be positive")
    prompt = (getattr(args, "prompt_file", None) or PROMPT_PATH).read_text()
    config = {"provider": args.provider, "endpoint": endpoint(args.base_url, args.provider),
              "model": args.model, "pairs": args.pairs, "batch_size": args.batch_size,
              "seed": args.seed, "max_tokens": args.max_tokens, "retries": args.retries,
              "timeout": args.timeout, "max_token_field": getattr(args, "max_token_field", "max_tokens"),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    # Absent options do not alter fingerprints of existing resumable ledgers.
    if getattr(args, "reasoning_effort", None):
        if args.provider != "openai-compatible": raise ValueError("reasoning-effort requires OpenAI-compatible provider")
        config["reasoning_effort"] = args.reasoning_effort
    if getattr(args, "json_mode", False):
        if args.provider != "openai-compatible": raise ValueError("json-mode requires OpenAI-compatible provider")
        config["json_mode"] = True
    planned_count = math.ceil(args.pairs / args.batch_size)
    jobs = iter_jobs(args.pairs, args.batch_size, args.seed)
    snapshot_only = getattr(args, "snapshot_only", False)
    print(json.dumps({"mode": "snapshot" if snapshot_only else "execute" if args.execute else "dry-run", "planned_requests": planned_count,
                      "requested_pairs": args.pairs, "max_output_tokens": planned_count * args.max_tokens,
                      "output": str(args.output), "concurrency": args.concurrency}), flush=True)
    if not args.execute and not snapshot_only:
        return
    if snapshot_only and not (args.output / 'ledger.sqlite3').is_file():
        raise ValueError("No existing ledger to snapshot")
    key = os.environ.get(args.api_key_env, "")
    if not snapshot_only and not key and urlparse(config["endpoint"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(f"Set {args.api_key_env} in the environment")
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    lock = acquire_run_lock(args.output / "run-lock.sqlite3")
    try:
        db = open_ledger(args.output / "ledger.sqlite3", fingerprint)
    except Exception:
        lock.close()
        raise
    rejected = Counter()
    try:
        saved_prompt = args.output / "prompt.txt"
        if saved_prompt.exists() and saved_prompt.read_text() != prompt:
            raise ValueError("Run prompt changed; use its saved prompt or a new output directory")
        if not saved_prompt.exists(): saved_prompt.write_text(prompt)
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('config', ?)", (json.dumps(config, sort_keys=True),))
        db.commit()
        if snapshot_only:
            manifest = materialize_snapshot(db, args.output, config, fingerprint, rejected)
            print(json.dumps({"snapshot": True, "accepted_candidates": manifest["accepted_candidates"],
                              "unfinished_requests": manifest["unfinished_requests"]}), flush=True)
            return
        waiting = iter(j for j in jobs if db.execute("SELECT status FROM jobs WHERE id=?", (j["id"],)).fetchone() != ('done',))
        # Keep only concurrency futures in flight instead of allocating millions.
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            def submit_one():
                job = next(waiting, None)
                if job is None: return False
                db.execute("INSERT INTO jobs(id,status,attempts) VALUES (?,'running',1) ON CONFLICT(id) DO UPDATE SET status='running', attempts=attempts+1", (job["id"],))
                db.commit()
                futures[executor.submit(request_teacher, job, config, prompt, key)] = job
                return True
            for _ in range(args.concurrency): submit_one()
            completed = 0
            while futures:
                future = next(as_completed(futures))
                job = futures.pop(future)
                try:
                    candidates, usage = future.result()
                    for candidate in candidates:
                        try:
                            row = validate_pair(candidate, expected_category=job["category"])
                            source_key = hashlib.sha256(normalized(row["source"]).casefold().encode()).hexdigest()
                            inserted = db.execute("INSERT OR IGNORE INTO pairs VALUES (?,?,?,?,?,?,?)", (row["pair_id"], row["source"], row["target"], row["category"], row["clean_group"], job["id"], source_key)).rowcount
                            if not inserted: rejected["duplicate_or_conflicting_source"] += 1
                        except ValueError as error: rejected[str(error)] += 1
                    safe_usage = {name: value for name, value in usage.items() if isinstance(value, int) and value >= 0}
                    db.execute("UPDATE jobs SET status='done',usage=?,error=NULL WHERE id=?", (json.dumps(safe_usage), job["id"]))
                except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
                    # Store a fixed error label only: no provider bodies or credentials.
                    db.rollback()
                    reason = error.reason if isinstance(error, TeacherError) else 'request_or_output_validation_failed'
                    db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (reason, job["id"]))
                db.commit()
                completed += 1
                if completed % getattr(args, "snapshot_every", 10) == 0:
                    materialize_snapshot(db, args.output, config, fingerprint, rejected)
                print(json.dumps({"completed_job": job["id"],
                                  "status": db.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()[0],
                                  "accepted_so_far": db.execute("SELECT count(*) FROM pairs").fetchone()[0]}), flush=True)
                submit_one()
        manifest = materialize_snapshot(db, args.output, config, fingerprint, rejected)
        statuses = manifest["status_counts"]
        print(json.dumps({"statuses": statuses, "accepted_candidates": manifest["accepted_candidates"], "rejected": dict(rejected)}), flush=True)
        if statuses.get("failed", 0): raise RuntimeError("Some teacher jobs failed; rerun the identical command to retry unfinished jobs")
    finally:
        db.close()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL"), required=not os.environ.get("TEACHER_BASE_URL"))
    parser.add_argument("--provider", choices=["openai-compatible", "anthropic"], default="openai-compatible")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, help="Use a saved run prompt when resuming after the default prompt changes")
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--pairs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-token-field", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"])
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/teacher/run-001"))
    parser.add_argument("--snapshot-every", type=int, default=10, help="Materialize committed rows every N completed jobs")
    parser.add_argument("--snapshot-only", action="store_true", help="Export an existing ledger without any provider requests; settings must match")
    parser.add_argument("--execute", action="store_true", help="Actually call the teacher; without this flag only print the plan")
    generate(parser.parse_args())
