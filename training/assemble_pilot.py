"""Bounded local teacher experiment with synthetic, non-teacher evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from pairs import group_id, hash_file, normalized


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def assemble(base, weak, output, max_template_train_rows=0):
    if max_template_train_rows < 0: raise ValueError("Template row limit must be nonnegative")
    base_manifest = json.loads((base / "manifest.json").read_text())
    weak_manifest = json.loads((weak / "manifest.json").read_text())
    if base_manifest.get("origin") != "original-template-v1" or base_manifest.get("license") != "CC0-1.0":
        raise ValueError("Pilot evaluation requires the original CC0 template corpus")
    if output.exists():
        raise ValueError("Pilot output already exists; choose a fresh directory")
    splits = {split: list(rows(base / f"{split}.jsonl")) for split in ["train", "dev", "test"]}
    if any(not split for split in splits.values()):
        raise ValueError("All template splits must be nonempty")
    heldout_groups = {group_id(row["target"]) for split in ["dev", "test"] for row in splits[split]}
    heldout_sources = {normalized(row["source"]).casefold() for split in ["dev", "test"] for row in splits[split]}
    if any(group_id(row["target"]) in heldout_groups or normalized(row["source"]).casefold() in heldout_sources
           for row in splits["train"]):
        raise ValueError("Template train data overlaps held-out data")
    if max_template_train_rows: splits["train"] = splits["train"][:max_template_train_rows]
    sources = {normalized(row["source"]).casefold(): group_id(row["target"]) for row in splits["train"]}
    counts = Counter({"template_train_selected": len(splits["train"])})
    for row in rows(weak / "train.jsonl"):
        key = normalized(row["source"]).casefold()
        target = group_id(row["target"])
        if target in heldout_groups or key in heldout_sources:
            counts["heldout_overlap_dropped"] += 1
        elif key in sources:
            counts["duplicate_or_conflicting_source_dropped"] += 1
        else:
            sources[key] = target
            splits["train"].append(row)
            counts["weak_train_added"] += 1
    train_tags = {tag for row in splits["train"] for tag in row["tags"]}
    for split in ["dev", "test"]:
        original = len(splits[split])
        splits[split] = [row for row in splits[split] if all(tag in train_tags for tag in row["tags"])]
        counts[f"{split}_unsupported_examples_dropped"] = original - len(splits[split])
        if not splits[split]: raise ValueError("No evaluation rows supported by training vocabulary")
    output.mkdir(parents=True)
    labels = ["KEEP"] + sorted(train_tags - {"KEEP"})
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    scope = "Original template in-distribution evaluation; teacher rows are unreviewed train-only. Not real-world GEC quality."
    manifest = {"schema": 1, "origin": "local-teacher-pilot", "evaluation_scope": scope,
                "publication_allowed": weak_manifest.get("publication_allowed", False),
                "licenses": {"CC0-1.0": "original templates", **weak_manifest["licenses"]},
                "base_manifest_sha256": hash_file(base / "manifest.json"),
                "weak_manifest_sha256": hash_file(weak / "manifest.json"),
                "label_count": len(labels), "counts": dict(counts), "splits": {}}
    for split, records in splits.items():
        path = output / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in records: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"rows": len(records), "sha256": hash_file(path),
                                      "clean_rows": sum(row["source"] == row["target"] for row in records)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/generated"))
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-template-train-rows", type=int, default=0,
                        help="Bound the synthetic smoke run; 0 retains the complete template training split")
    args = parser.parse_args()
    assemble(args.base, args.weak, args.output, args.max_template_train_rows)
