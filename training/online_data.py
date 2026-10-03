"""Original machine-generated, screened weak pairs for a continual local RL loop."""
from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import shutil

from data import edit_tags
from edit_ops import reconstruct
from generate_llm import DOMAINS, ERRORS
from judge import (Judge, JudgeTransportError, canonical, digest, endpoint, ledger_append, ledger_read,
                   live_response, response_text, strict_json)
from judge_objective import ONLINE_ORIGIN, eligible
from pairs import hash_file, normalized, validate_pair

PROMPT = """Generate entirely original fictional English correction pairs.
Return only JSON {"pairs":[{"id":"requested ID","source":"text","target":"text","category":"requested category"}]}.
Return each requested ID exactly once with exactly these keys. Sources should resemble natural
writing in the requested domain, with varied vocabulary, structures, lengths, and situations.
Use 3 to 50 words, ASCII text, and no quotes, code, contact details, URLs, credentials, real
personal facts, copyrighted text, or copied examples. Do not include the diversity nonce.
For clean requests, return an already grammatical source and an EXACTLY identical target.
Otherwise introduce one unmistakable error of the requested category and supply its minimal
correction. Do not change facts, intent, negation, valid dialect, register, or style.
Do not replace an already valid tense or arbitrarily insert/delete an article. Avoid ambiguous
constructions. Keep all unrelated source characters unchanged. No explanations or extra keys."""

PAIR_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pairs"],
    "properties": {"pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["id", "source", "target", "category"], "properties": {
            "id": {"type": "string"}, "source": {"type": "string"}, "target": {"type": "string"},
            "category": {"type": "string", "enum": [*ERRORS, "clean"]}}}}}}


def rows(path):
    return [strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_rows(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        for value in values: stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def generation_spec(route, count, batch_size, seed):
    if route["model"] != "gpt-6-luna" or route["expected_provider"] != "codex" or route["expected_auth"] != "subscription":
        raise ValueError("Online generation requires the verified Codex subscription Luna route")
    if not 4 <= count <= 256 or not 1 <= batch_size <= 16:
        raise ValueError("Generation requires 4..256 pairs and batches of 1..16")
    return {"schema": 1, "route": route, "pairs": count, "batch_size": batch_size,
            "seed": seed, "prompt_sha256": digest(PROMPT), "publication_allowed": False,
            "format_sha256": digest(canonical(PAIR_SCHEMA)),
            "endpoint": endpoint(os.environ.get("GAMMA_JUDGE_BASE_URL", "http://127.0.0.1:8317/v1"))}


def generate(directory, spec, transport=None, before_request=None):
    """Charge attempts durably; checkpoint each parsed batch before requesting another."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "spec.json"
    if path.exists() and strict_json(path.read_text()) != spec:
        raise ValueError("Generation plan changed")
    if not path.exists(): atomic_json(path, spec)
    identity = digest(canonical(spec))
    requests, chain = ledger_read(directory / "requests.jsonl")
    responses, response_chain = ledger_read(directory / "responses.jsonl")
    completed = {row["job"]: row for row in responses}
    if len(completed) != len(responses): raise ValueError("Duplicate generation response identity")
    budget = math.ceil(spec["pairs"] / spec["batch_size"])
    if len(requests) > budget or any(row.get("spec_sha256") != identity for row in requests):
        raise ValueError("Generation request budget or identity changed")
    rng = random.Random(spec["seed"])
    result = []
    for job in range(budget):
        count = min(spec["batch_size"], spec["pairs"] - job * spec["batch_size"])
        requested = [{"id": str(job * spec["batch_size"] + index),
                      "category": "clean" if (job * spec["batch_size"] + index) % 4 == 3
                      else ERRORS[(job * spec["batch_size"] + index) % len(ERRORS)],
                      "domain": rng.choice(DOMAINS)} for index in range(count)]
        nonce = f"{spec['seed']}:{job}:{rng.getrandbits(64):016x}"
        batch_path = directory / f"batch-{job:04d}.json"
        if batch_path.exists():
            saved = strict_json(batch_path.read_text())
            if (saved["spec_sha256"] != identity or saved["pairs_sha256"] != digest(canonical(saved["pairs"]))
                    or job >= len(requests) or saved["payload_sha256"] != requests[job]["payload_sha256"]
                    or job not in completed or completed[job]["batch_sha256"] != hash_file(batch_path)):
                raise ValueError("Generation batch identity changed")
            result.extend(saved["pairs"])
            continue
        if job < len(requests):
            raise ValueError("Generation attempt has no completed batch; retain this cycle and start a new one")
        payload = {"model": spec["route"]["model"], "instructions": PROMPT,
                   "input": canonical({"requests": requested, "diversity_nonce": nonce}),
                   "reasoning": {"effort": "none"}, "max_output_tokens": 8192,
                   "text": {"format": {"type": "json_schema", "name": "original_correction_pairs",
                                       "strict": True, "schema": PAIR_SCHEMA}}}
        payload_hash = digest(canonical(payload))
        if before_request: before_request()
        chain = ledger_append(directory / "requests.jsonl", {"job": job, "spec_sha256": identity,
                              "payload_sha256": payload_hash, "pairs": count}, chain)
        key = os.environ.get("GAMMA_JUDGE_API_KEY", "")
        if transport is None and not key: raise ValueError("GAMMA_JUDGE_API_KEY is required")
        try:
            body, headers = transport(payload) if transport else live_response(spec["endpoint"], payload, key, 60)
            text, receipt = response_text(body, headers, spec["route"])
            secrets = [value for value in (key, os.environ.get("GAMMA_JUDGE_ATTRIBUTION_TOKEN", "")) if value]
            if any(value in text for value in secrets): raise ValueError("Credential echo rejected")
            parsed = strict_json(text)
            if not isinstance(parsed, dict) or set(parsed) != {"pairs"} or not isinstance(parsed["pairs"], list):
                raise ValueError("Generated pair schema rejected")
            expected = {row["id"]: row for row in requested}
            accepted = []
            for raw in parsed["pairs"]:
                if not isinstance(raw, dict) or set(raw) != {"id", "source", "target", "category"} or raw.get("id") not in expected:
                    raise ValueError("Generated pair IDs rejected")
                request = expected.pop(raw["id"])
                pair = validate_pair(raw, request["category"])
                pair.update({"origin": ONLINE_ORIGIN, "license": "provider-terms-unverified",
                             "review_status": "unreviewed", "human_reviewed": False,
                             "machine_generated_original": True, "publication_allowed": False,
                             "generation_model": spec["route"]["model"], "generation_request_id": receipt["request_id"],
                             "generator_spec_sha256": identity, "domain": request["domain"]})
                if not eligible(pair) or not eligible({**pair, "source": pair["target"]}):
                    raise ValueError("Generated source/target content rejected")
                accepted.append(pair)
            if expected: raise ValueError("Generated pair IDs missing")
        except JudgeTransportError:
            raise
        except Exception as error:
            reason = str(error) if type(error) is ValueError else type(error).__name__
            raise ValueError(f"Generation rejected: {reason}; no provider content retained") from None
        atomic_json(batch_path, {"spec_sha256": identity, "payload_sha256": payload_hash,
                                "pairs_sha256": digest(canonical(accepted)), "pairs": accepted,
                                "response": receipt})
        response_chain = ledger_append(directory / "responses.jsonl", {"job": job, "spec_sha256": identity,
                                        "batch_sha256": hash_file(batch_path), "response": receipt}, response_chain)
        result.extend(accepted)
    write_rows(directory / "candidates.jsonl", result)
    return result


def filter_pairs(candidates, labels, schema, excluded, seen, tokenizer=None, max_length=96):
    accepted, counts = [], Counter()
    for raw in candidates:
        source = normalized(raw["source"]).casefold()
        target = normalized(raw["target"]).casefold()
        if source in excluded or target in excluded:
            counts["heldout_overlap"] += 1
            continue
        if source in seen:
            counts["duplicate_source"] += 1
            continue
        try:
            words, tags = edit_tags(raw["source"], raw["target"], schema)
            if any(tag not in labels for tag in tags): raise ValueError("unsupported_label")
            if raw["source"] != raw["target"] and normalized(reconstruct(words, tags, schema)) != normalized(raw["target"]):
                raise ValueError("unsupported_alignment")
            if len(words) > 60: raise ValueError("context_budget")
            if tokenizer is not None and len(tokenizer(words, is_split_into_words=True)["input_ids"]) > max_length:
                raise ValueError("wordpiece_context_budget")
        except ValueError:
            counts["unsupported_edit_or_context"] += 1
            continue
        seen.add(source)
        accepted.append({**raw, "tokens": words, "tags": tags})
    return accepted, dict(counts)


def screen_pairs(directory, candidates, route, transport=None, before_request=None, batch_size=8):
    if not candidates: return [], {"screened": 0, "rejected": 0}
    with Judge(directory, route, batch_size=batch_size, max_requests=32, max_pairs=256,
               timeout=60, transport=transport, before_request=before_request) as judge:
        verdicts = judge.judge([(row["source"], row["target"]) for row in candidates])
        accepted = [{**row, "admission_spec_sha256": judge.spec_sha256,
                     "admission_pair_id": verdict["id"], "admission_flags": verdict["flags"]}
                    for row, verdict in zip(candidates, verdicts) if all(verdict["flags"].values())]
        return accepted, {"screened": len(candidates), "rejected": len(candidates) - len(accepted), **judge.stats()}


def prepare_cycle(anchor, output, replay):
    """Keep anchor rows/labels/notices fixed; add only screened, representable replay."""
    anchor, output = Path(anchor), Path(output)
    if output.exists(): raise ValueError("Cycle dataset already exists")
    output.mkdir()
    manifest = deepcopy(strict_json((anchor / "manifest.json").read_text()))
    manifest.update({"publication_allowed": False, "online_weak_rows": len(replay),
                     "online_scope": "Original generated, machine-screened weak training only; fixed human evaluation."})
    manifest["online_terms"] = "provider-terms-unverified; not qualified for weight redistribution"
    manifest["anchor_provenance"] = {"path": str(anchor), "manifest_sha256": hash_file(anchor / "manifest.json")}
    manifest.setdefault("source_provenance", []).append({"origin": ONLINE_ORIGIN,
        "license": "provider-terms-unverified", "publication_allowed": False,
        "review_status": "machine-screened-unreviewed", "rows": len(replay),
        "generator_spec_sha256": sorted({row["generator_spec_sha256"] for row in replay}),
        "generation_request_ids": sorted({row["generation_request_id"] for row in replay})})
    manifest["licenses"] = dict(Counter(row.get("license", "unspecified")
                                        for row in rows(anchor / "train.jsonl") + replay))
    manifest["training_origins"] = dict(Counter(row.get("origin", "unspecified")
                                                for row in rows(anchor / "train.jsonl") + replay))
    for filename in ["labels.json", "dev.jsonl", "test.jsonl", *manifest.get("license_files", {})]:
        if Path(filename).name != filename: raise ValueError("Unsafe dataset filename")
        shutil.copyfile(anchor / filename, output / filename)
    write_rows(output / "train.jsonl", rows(anchor / "train.jsonl") + replay)
    manifest["labels_sha256"] = hash_file(output / "labels.json")
    for split in ("train", "dev", "test"):
        manifest["splits"][split] = {"rows": len(rows(output / f"{split}.jsonl")),
                                    "sha256": hash_file(output / f"{split}.jsonl")}
    atomic_json(output / "manifest.json", manifest)
    write_rows(output / "judge-sources.jsonl", replay)
    return manifest
