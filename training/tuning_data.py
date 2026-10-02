"""Frozen CC0 mixture and clean controls for controlled Tiny-model tuning."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from data import build as build_base, edit_tags, tokens
from pairs import group_id, hash_file, normalized, pair_id
from synthetic_challenge import build as build_extra, TRAIN_NOUNS, TRAIN_ADJECTIVES


def key(text):
    return normalized(text).casefold()


def clean_controls():
    items = [("lantern", "lanterns"), ("basket", "baskets"), ("camera", "cameras"),
             ("ticket", "tickets"), ("notebook", "notebooks"), ("map", "maps"),
             ("painting", "paintings"), ("instrument", "instruments")]
    places = ["studio", "museum", "station", "classroom", "workshop", "office", "garden", "library"]
    for item, plural in items:
        for place in places:
            yield f"They left their {plural} near the {place}."
            yield f"Their {plural} are inside the {place}."
            yield f"Your {item} is outside the {place}."
            yield f"There are several {plural} inside the {place}."
            yield f"We know that that {item} is inside the {place}."
            yield f"They said that their {plural} were outside the {place}."
    for singular, plural in TRAIN_NOUNS:
        for adjective in TRAIN_ADJECTIVES:
            yield f"You're {adjective} because the {singular} arrived."
            yield f"They're {adjective} because the {plural} arrived."
            yield f"The {singular} said that the {plural} have more time."
            yield f"The {plural} said that the {singular} has more time."
        for item, _ in items:
            yield f"Does the {singular} have a {item}?" if not item.startswith('i') else f"Does the {singular} have an {item}?"
            yield f"Do the {plural} have a {item}?" if not item.startswith('i') else f"Do the {plural} have an {item}?"
            yield f"The {singular} should have a {item}." if not item.startswith('i') else f"The {singular} should have an {item}."
            yield f"The {singular} does not have a {item}." if not item.startswith('i') else f"The {singular} does not have an {item}."
        yield f"I recommend that the {singular} have more time."
        yield f"If the {singular} were here, we could begin."
        yield f"The {singular} had had a busy day."
    for document in ["report", "book", "message", "letter", "invitation"]:
        for tail in ["yesterday", "last night", "last week", "before the meeting", "after the rehearsal"]:
            for subject in ["I", "He", "She", "They", "We"]:
                yield f"{subject} read the {document} {tail}."


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build(output, evaluation_dir=None):
    if output.exists():
        raise ValueError("Choose a fresh tuning-data directory")
    output.mkdir(parents=True)
    base, extra = output / "original", output / "extra"
    build_base(base)
    build_extra(extra)
    excluded = {key(row[field]) for row in json.loads((extra / "challenge.json").read_text())["cases"] for field in ["source", "target"]}
    evaluation_hashes = {}
    if evaluation_dir:
        for split in ["dev", "test"]:
            path = evaluation_dir / f"{split}.jsonl"
            evaluation_hashes[split] = hash_file(path)
            for row in read_rows(path):
                excluded.add(key(row["source"]))
                excluded.update(key(reference) for reference in row["references"])
    assignments, records, conflicts = {}, {}, set()
    counts = Counter()
    for directory in [base, extra]:
        for split in ["train", "dev", "test"]:
            for row in read_rows(directory / f"{split}.jsonl"):
                source, target = key(row["source"]), key(row["target"])
                # Preserve the original corpus's split assignment where shared.
                assigned = assignments.setdefault(target, split)
                if assigned != split:
                    row = {**row, "reassigned_split": assigned}
                # A singular 'read' is valid past tense; don't teach it as an error.
                if any(word.lower() == 'read' and tag == 'REPLACE:reads' for word, tag in zip(row["tokens"], row["tags"])):
                    counts["ambiguous_read_removed"] += 1
                    continue
                if source in excluded or target in excluded:
                    counts["evaluation_overlap_removed"] += 1
                    continue
                if source in records and key(records[source][1]["target"]) != target:
                    conflicts.add(source)
                else:
                    records.setdefault(source, (assigned, row))
    for clean in sorted(set(clean_controls())):
        source = key(clean)
        if source in excluded:
            counts["evaluation_overlap_removed"] += 1
            continue
        group = group_id(clean)
        bucket = int(group[:8], 16) % 100
        split = assignments.setdefault(source, "train" if bucket < 80 else "dev" if bucket < 90 else "test")
        words = tokens(clean)
        row = {"source": clean, "target": clean, "tokens": words, "tags": ["KEEP"] * len(words),
               "clean_group": group, "origin": "original-clean-controls-v1", "license": "CC0-1.0"}
        if source in records and key(records[source][1]["target"]) != source:
            conflicts.add(source)
        else:
            records.setdefault(source, (split, row))
    splits = {name: [] for name in ["train", "dev", "test"]}
    for source, (split, row) in sorted(records.items()):
        if source in conflicts:
            counts["conflicting_source_removed"] += 1
            continue
        words, tags = edit_tags(row["source"], row["target"])
        splits[split].append({**row, "tokens": words, "tags": tags, "clean_group": group_id(row["target"]),
                              "pair_id": pair_id(row["source"], row["target"]), "license": "CC0-1.0"})
    # Audit normalized source and target separation after merging different splitters.
    seen = set()
    for rows in splits.values():
        population = {key(row[field]) for row in rows for field in ["source", "target"]}
        if population & seen:
            raise ValueError("Merged synthetic splits overlap")
        seen |= population
    labels = ["KEEP"] + sorted({tag for row in splits["train"] for tag in row["tags"]} - {"KEEP"})
    if any(tag not in labels for rows in splits.values() for row in rows for tag in row["tags"]):
        raise ValueError("Training vocabulary does not cover synthetic heldouts")
    manifest = {"schema": 1, "origin": "original-cc0-tuning-v1", "license": "CC0-1.0", "publication_allowed": True,
                "evaluation_scope": "Synthetic token-label evaluation; independent natural/challenge evaluation required.",
                "generator_sha256": hash_file(Path(__file__)), "label_count": len(labels), "counts": dict(counts),
                "excluded_evaluation_hashes": evaluation_hashes, "challenge_sha256": hash_file(extra / "challenge.json"),
                "source_manifests": {"original": hash_file(base / "manifest.json"), "extra": hash_file(extra / "manifest.json")}, "splits": {}}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][split] = {"rows": len(rows), "clean_rows": sum(row["source"] == row["target"] for row in rows),
                                    "sha256": hash_file(path), "origins": dict(Counter(row["origin"] for row in rows))}
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.evaluation_dir), indent=2))
