"""Merge verified teacher snapshots deterministically; raw outputs stay private."""
import argparse
from collections import Counter
import json
from pathlib import Path

from pairs import hash_file, normalized, validate_pair


def merge(inputs, output):
    if output.exists(): raise ValueError("Merged output already exists; choose a fresh directory")
    records, conflicts, sources = {}, set(), []
    counts = Counter()
    for directory in inputs:
        path = directory / "candidates.jsonl"
        manifest = json.loads((directory / "manifest.json").read_text())
        digest = hash_file(path)
        if digest != manifest.get("candidates_sha256"):
            raise ValueError("Teacher snapshot hash mismatch; export a fresh snapshot before merging")
        sources.append({"run": directory.name, "fingerprint": manifest["fingerprint"],
                        "candidates_sha256": digest, "manifest_sha256": hash_file(directory / "manifest.json"),
                        "config": manifest["config"], "accepted_candidates": manifest["accepted_candidates"],
                        "completed_request_usage": manifest.get("completed_request_usage", {}),
                        "unfinished_requests": manifest["unfinished_requests"]})
        rows = [json.loads(line) for line in path.open()]
        if len(rows) != manifest["accepted_candidates"]: raise ValueError("Teacher snapshot population mismatch")
        for raw in rows:
            pair = validate_pair(raw)
            if (raw.get("origin") != "llm-teacher" or raw.get("license") != "provider-terms-unverified"
                    or raw.get("model") != manifest["config"]["model"]
                    or raw.get("prompt_sha256") != manifest["config"]["prompt_sha256"]):
                raise ValueError("Teacher row provenance mismatch")
            counts["scanned"] += 1
            key = normalized(pair["source"]).casefold()
            value = {**raw, **pair, "generation_run": manifest["fingerprint"]}
            if key in records:
                counts["duplicate_source"] += 1
                if normalized(records[key]["target"]) != normalized(pair["target"]): conflicts.add(key)
                # Stable provenance when the same row occurs in several snapshots.
                if (value["pair_id"], value["generation_run"]) < (records[key]["pair_id"], records[key]["generation_run"]): records[key] = value
            else: records[key] = value
    selected = sorted((row for key, row in records.items() if key not in conflicts), key=lambda row: row["pair_id"])
    if not selected: raise ValueError("No nonconflicting teacher pairs")
    counts["conflicting_sources_removed"] = len(conflicts)
    counts["accepted"] = len(selected)
    output.mkdir(parents=True)
    with (output / "candidates.jsonl").open("w") as stream:
        for row in selected: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"schema": 1, "kind": "merged-llm-teacher", "sources": sorted(sources, key=lambda source: source["fingerprint"]),
              "counts": dict(counts), "categories": dict(Counter(row["category"] for row in selected)),
              "candidates_sha256": hash_file(output / "candidates.jsonl"),
              "publication_allowed": False, "license": "provider-terms-unverified",
              "quality": "Unreviewed train-only weak supervision; structural validation is not grammar validation."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True, help="Teacher run directory with matching snapshot and manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.input, args.output)
    print(json.dumps({"counts": result["counts"], "categories": result["categories"], "sha256": result["candidates_sha256"]}))
