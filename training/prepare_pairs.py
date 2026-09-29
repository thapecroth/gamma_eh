"""Prepare tagged training data; weak labels are train-only, never dev/test."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from data import edit_tags, render, tokens
from pairs import group_id, hash_file, normalized, validate_pair

LICENSES = {"CC0-1.0", "CC-BY-4.0", "Apache-2.0", "MIT"}


def reconstruct(words, tags):
    result = []
    for word, tag in zip(words, tags):
        if tag == "KEEP": result.append(word)
        elif tag == "DELETE": continue
        elif tag.startswith("REPLACE:"):
            replacement = tag[8:]
            result.append(replacement.capitalize() if word[0].isupper() else replacement)
        elif tag.startswith("APPEND:"): result.extend([word, tag[7:]])
    return render(result)


def prepare(inputs, output, allow_weak_train=False, teacher_license=None, max_labels=4096):
    if teacher_license is not None and teacher_license not in LICENSES:
        raise ValueError("Unsupported teacher output license")
    if max_labels < 2: raise ValueError("max-labels must be at least2")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["train.jsonl", "dev.jsonl", "test.jsonl", "prepare.sqlite3"]):
        raise ValueError("Prepared output already exists. Choose a fresh directory")
    db = sqlite3.connect(output / "prepare.sqlite3")
    db.executescript("""
      CREATE TABLE records (source_key TEXT PRIMARY KEY, target_key TEXT NOT NULL, clean_group TEXT NOT NULL, reviewed INTEGER NOT NULL, value TEXT NOT NULL, conflict INTEGER NOT NULL DEFAULT 0);
      CREATE INDEX groups ON records(clean_group);
    """)
    counts = Counter()
    origins = Counter()
    license_counts = Counter()
    try:
        for path in inputs:
            with path.open() as stream:
                for line in stream:
                    counts["scanned"] += 1
                    try:
                        raw = json.loads(line)
                        row = validate_pair(raw)
                        license_id = raw.get("license")
                        if raw.get("origin") == "llm-teacher" and teacher_license: license_id = teacher_license
                        if license_id not in LICENSES: raise ValueError("license_unverified")
                        reviewed = raw.get("review_status") == "human-reviewed"
                        if not reviewed and not allow_weak_train: raise ValueError("unreviewed")
                        words, tags = edit_tags(row["source"], row["target"])
                        if all(tag == 'KEEP' for tag in tags) and normalized(row['source']) != normalized(row['target']):
                            raise ValueError('unrepresentable_spacing_change')
                        # Detect casing or spacing changes unsupported by the current
                        # lowercased edit vocabulary instead of silently losing them.
                        if normalized(reconstruct(words, tags)) != normalized(render(tokens(row["target"]))):
                            raise ValueError("unrepresentable_case_change")
                        row.update({"tokens": words, "tags": tags, "license": license_id,
                                    "origin": raw.get("origin", "unspecified"),
                                    "review_status": "human-reviewed" if reviewed else "weak-supervision",
                                    "source_revision": raw.get("source_revision"), "model": raw.get("model"),
                                    "source_url": raw.get("source_url"),
                                    "original_source_url": raw.get("original_source_url"),
                                    "prompt_sha256": raw.get("prompt_sha256")})
                        source_key = hashlib.sha256(normalized(row["source"]).casefold().encode()).hexdigest()
                        target_key = group_id(row["target"])
                        existing = db.execute("SELECT target_key,reviewed FROM records WHERE source_key=?", (source_key,)).fetchone()
                        if existing:
                            if existing[0] != target_key:
                                db.execute("UPDATE records SET conflict=1 WHERE source_key=?", (source_key,))
                                counts["conflicting_source"] += 1
                            elif reviewed and not existing[1]:
                                db.execute("UPDATE records SET reviewed=1,value=? WHERE source_key=?", (json.dumps(row), source_key))
                            else: counts["duplicate"] += 1
                            continue
                        db.execute("INSERT INTO records(source_key,target_key,clean_group,reviewed,value) VALUES (?,?,?,?,?)",
                                   (source_key, target_key, row["clean_group"], int(reviewed), json.dumps(row)))
                        origins[row["origin"]] += 1
                        license_counts[license_id] += 1
                    except (ValueError, KeyError, TypeError):
                        counts["rejected"] += 1
            db.commit()
        # If any variant is reviewed, its entire clean group follows the reviewed
        # split. Unreviewed variants assigned dev/test are dropped, never moved to
        # train: moving them would leak their held-out target into training.
        db.execute("CREATE TABLE reviewed_groups AS SELECT DISTINCT clean_group FROM records WHERE reviewed=1 AND conflict=0")
        db.execute("CREATE UNIQUE INDEX reviewed_group_index ON reviewed_groups(clean_group)")
        split_rows = Counter()
        tag_counts = Counter()
        streams = {s: (output / f"{s}.jsonl").open("w") for s in ["train", "dev", "test"]}
        try:
            query = "SELECT r.value,r.reviewed,g.clean_group FROM records r LEFT JOIN reviewed_groups g ON r.clean_group=g.clean_group WHERE r.conflict=0 ORDER BY r.clean_group,r.source_key"
            for value, reviewed, reviewed_group in db.execute(query):
                row = json.loads(value)
                bucket = int(row["clean_group"][:8], 16) % 100
                split = "train" if not reviewed_group or bucket < 80 else "dev" if bucket < 90 else "test"
                if split != "train" and not reviewed:
                    counts["weak_heldout_group_dropped"] += 1
                    continue
                if split == "train": tag_counts.update(row["tags"])
                streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                split_rows[split] += 1
        finally:
            for stream in streams.values(): stream.close()
        labels = ["KEEP"] + [tag for tag, _ in tag_counts.most_common() if tag != "KEEP"][:max_labels - 1]
        supported = set(labels)
        coverage = {}
        # Rewrite each split atomically with examples representable by the train
        # vocabulary. Rejection coverage must travel with any reported metric.
        for split in ["train", "dev", "test"]:
            filename = output / f"{split}.jsonl"
            replacement = output / f"{split}.filtered.jsonl"
            kept = dropped = 0
            with filename.open() as source, replacement.open("w") as destination:
                for line in source:
                    row = json.loads(line)
                    if all(tag in supported for tag in row["tags"]): destination.write(line); kept += 1
                    else: dropped += 1
            replacement.replace(filename)
            coverage[split] = {"accepted": kept, "unsupported_tag_examples": dropped,
                               "sha256": hash_file(filename)}
        (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
        manifest = {"schema": 1, "kind": "prepared-edit-tags", "inputs": [str(path) for path in inputs],
                    "counts": dict(counts), "origins_before_vocabulary_filter": dict(origins),
                    "licenses": dict(license_counts), "weak_labels_train_only": True,
                    "label_count": len(labels), "splits": coverage,
                    "evaluation_ready": all(coverage[s]["accepted"] > 0 for s in ["dev", "test"])}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        if not manifest["evaluation_ready"]:
            print("Training data prepared. Add human-reviewed dev/test pairs before using training/train.py; no synthetic evaluation split was fabricated.")
        return manifest
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/prepared/run-001"))
    parser.add_argument("--allow-weak-train", action="store_true")
    parser.add_argument("--teacher-license", choices=sorted(LICENSES), help="Only after checking your model provider's output terms")
    parser.add_argument("--max-labels", type=int, default=4096)
    args = parser.parse_args()
    prepare(args.input, args.output, args.allow_weak_train, args.teacher_license, args.max_labels)
