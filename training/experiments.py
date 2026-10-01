"""Bounded equal-raw-row comparisons; execution is explicit and sequential."""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import subprocess
import sys

from data import edit_tags
from pairs import evaluation_keys, group_id, hash_file, normalized, validate_pair
from prepare_pairs import prepare

TINY = ("google/bert_uncased_L-2_H-128_A-2", "30b0a37ccaaa32f332884b96992754e246e48c5f")
LARGER = ("google/bert_uncased_L-4_H-256_A-4", "387825ce42dbb39b87911cdf8e383ee3b25184f8")


def read_rows(path):
    with path.open() as stream:
        for line in stream: yield json.loads(line)


def excluded_evaluation(directory):
    keys = evaluation_keys([row for split in ["dev", "test"] for row in read_rows(directory / f"{split}.jsonl")])
    return keys, {group_id(value) for value in keys}


def select_rows(path, limit, max_scanned, seed, excluded_sources, excluded_targets, templates=False):
    """Stable bounded hash sample of scanned rows; never print private text."""
    heap, seen, conflicts = [], {}, set()
    counts = Counter()
    for raw in read_rows(path):
        if counts["scanned"] >= max_scanned: break
        counts["scanned"] += 1
        try:
            value = dict(raw)
            if templates:
                value.update({"license": "CC0-1.0", "review_status": "synthetic-template",
                              "category": "clean" if normalized(raw["source"]) == normalized(raw["target"]) else "mixed"})
            if value.get("evaluation_only") or value.get("license") == "CC-BY-NC-SA-4.0":
                raise ValueError("evaluation_only_source")
            pair = validate_pair(value)
            key = normalized(pair["source"]).casefold()
            target = normalized(pair["target"])
            if key in excluded_sources or group_id(target) in excluded_targets:
                counts["heldout_overlap"] += 1
                continue
            source_key = hashlib.sha256(key.encode()).hexdigest()
            if source_key in seen:
                if seen[source_key] != target: conflicts.add(source_key); counts["conflicting_source"] += 1
                else: counts["duplicate"] += 1
                continue
            seen[source_key] = target
            priority = int(hashlib.sha256(f"{seed}:{pair['pair_id']}".encode()).hexdigest(), 16)
            entry = (-priority, source_key, {**value, **pair})
            if len(heap) < limit: heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]: heapq.heapreplace(heap, entry)
        except (ValueError, TypeError, KeyError): counts["invalid"] += 1
    selected = [row for _, key, row in sorted(heap, reverse=True) if key not in conflicts]
    counts["selected"] = len(selected)
    return selected, dict(counts)


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def assemble_dataset(output, records, template, evaluation, max_labels):
    raw = output.parent / (output.name + "-raw.jsonl")
    write_rows(raw, records)
    manifest = prepare([raw], output, allow_weak_train=True, allow_unverified_teacher_terms=True,
                       max_labels=max_labels, schema=2)
    labels = set(json.loads((output / "labels.json").read_text()))
    for split in ["dev", "test"]:
        tagged, counts = [], Counter()
        for row in read_rows(template / f"{split}.jsonl"):
            counts["population"] += 1
            try:
                words, tags = edit_tags(row["source"], row["target"], 2)
                if not all(tag in labels for tag in tags): counts["unsupported_vocabulary"] += 1; continue
                tagged.append({**row, "tokens": words, "tags": tags})
            except ValueError: counts["unrepresentable"] += 1
        write_rows(output / f"{split}.jsonl", tagged)
        manifest["splits"][split] = {**dict(counts), "accepted": len(tagged), "sha256": hash_file(output / f"{split}.jsonl")}
        manifest["splits"][split]["scope"] = "Supported original-template token-label subset"
        shutil.copyfile(evaluation / f"{split}.jsonl", output / "evaluation" / f"{split}.jsonl")
        manifest["evaluation"][split] = {"rows": sum(1 for _ in read_rows(evaluation / f"{split}.jsonl")),
                                         "sha256": hash_file(evaluation / f"{split}.jsonl")}
    if (evaluation / "manifest.json").exists():
        shutil.copyfile(evaluation / "manifest.json", output / "evaluation" / "manifest.json")
    manifest.update({"evaluation_ready": True, "raw_train_rows": len(records), "raw_train_sha256": hash_file(raw),
                     "labels_sha256": hash_file(output / "labels.json"),
                     "evaluation_scope": "Independent complete natural-reference evaluation; tagged synthetic subset reported separately.",
                     "evaluation_license_separate_from_training": True})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def common_training_population(datasets, manifests, token_count, budget, seed):
    retained, counts = {}, {}
    for name in manifests:
        records = list(read_rows(datasets / name / "train.jsonl"))
        eligible = [row for row in records if token_count(row["tokens"]) <= budget]
        retained[name] = sorted(eligible, key=lambda row: hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).hexdigest())
        counts[name] = {"supported_before_budget": len(records), "within_common_budget": len(eligible),
                        "token_budget_rejected": len(records) - len(eligible)}
    selected = min(len(records) for records in retained.values())
    if not selected: raise ValueError("No common supported training population within token budget")
    for name, records in retained.items():
        path = datasets / name / "train.jsonl"
        write_rows(path, records[:selected])
        manifests[name]["splits"]["train"].update({"accepted": selected, "sha256": hash_file(path)})
        manifests[name]["controlled_training"] = {**counts[name], "selected": selected, "common_wordpiece_budget": budget}
        (datasets / name / "manifest.json").write_text(json.dumps(manifests[name], indent=2) + "\n")
    return {"rows_per_arm": selected, "wordpiece_budget": budget, "coverage": counts,
            "limitation": "All training rows fit the common budget; longer context comparison measures inference context, without long-context training examples."}


def build(args, token_count=None):
    args.output = args.output.resolve()
    if args.output.exists(): raise ValueError("Experiment directory already exists; choose a fresh directory")
    if args.rows < 4 or not args.weak or args.max_scanned < args.rows or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Positive bounded schedule, weak inputs, rows>=4 and max-scanned>=rows are required")
    excluded_sources, excluded_targets = excluded_evaluation(args.evaluation_dir)
    # Retain template heldout separation even when selection uses natural dev.
    extra_sources, extra_targets = excluded_evaluation(args.template)
    excluded_sources |= extra_sources
    excluded_targets |= extra_targets
    template_rows, template_counts = select_rows(args.template / "train.jsonl", args.rows, args.max_scanned,
                                                  args.seed, excluded_sources, excluded_targets, templates=True)
    weak_sets, weak_counts = [], []
    # Prevent conflicts and duplicate raw rows across all mixed sources.
    mixed_sources = set(excluded_sources) | {normalized(row["source"]).casefold() for row in template_rows}
    for path in args.weak:
        records, counts = select_rows(path, args.rows, args.max_scanned, args.seed,
                                      mixed_sources, excluded_targets)
        weak_sets.append(records); weak_counts.append(counts)
        mixed_sources.update(normalized(row["source"]).casefold() for row in records)
    unit = 2 * len(weak_sets)
    effective = min(args.rows, len(template_rows), *(len(records) * unit for records in weak_sets))
    effective -= effective % unit
    if effective < unit: raise ValueError("Insufficient independent raw rows for equal-count comparison")
    per_weak = effective // unit
    baseline = template_rows[:effective]
    mixed = template_rows[:effective // 2] + [row for records in weak_sets for row in records[:per_weak]]
    if len(baseline) != len(mixed): raise ValueError("Comparison row count mismatch")
    args.output.mkdir(parents=True)
    datasets = args.output / "datasets"
    datasets.mkdir()
    manifests = {"template": assemble_dataset(datasets / "template", baseline, args.template, args.evaluation_dir, args.max_labels),
                 "mixed": assemble_dataset(datasets / "mixed", mixed, args.template, args.evaluation_dir, args.max_labels)}
    controlled = None
    common_budget = getattr(args, "common_token_budget", 0)
    if common_budget:
        if not 4 <= common_budget <= 64: raise ValueError("Common budget must fit the smallest arm: 4..64")
        if token_count is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, revision=args.tokenizer_revision,
                                                      use_fast=True, local_files_only=True)
            if not tokenizer.is_fast or not getattr(tokenizer, "do_lower_case", False):
                raise ValueError("Common budget requires a fast uncased BERT tokenizer")
            token_count = lambda words: len(tokenizer(words, is_split_into_words=True)["input_ids"])
        controlled = common_training_population(datasets, manifests, token_count, common_budget, args.seed)
    arms = [("template-tiny64", "template", TINY, 64), ("mixed-tiny64", "mixed", TINY, 64),
            ("mixed-tiny128", "mixed", TINY, 128), ("mixed-larger128", "mixed", LARGER, 128)]
    commands = []
    for name, dataset, base, context in arms:
        command = [sys.executable, str(Path(__file__).resolve().with_name("train.py")),
                   "--data", str(datasets / dataset), "--evaluation-dir", str(args.evaluation_dir.resolve()),
                   "--output", str(args.output / name / "model"), "--checkpoint", str(args.output / name / "checkpoint"),
                   "--base-model", base[0], "--base-revision", base[1], "--max-length", str(context),
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--learning-rate", str(args.learning_rate), "--seed", str(args.seed),
                   "--scorer", getattr(args, "scorer", "approximate"),
                   "--min-dev-edits", str(getattr(args, "min_dev_edits", 25))]
        if args.local_files_only: command.append("--local-files-only")
        if args.device: command.extend(["--device", args.device])
        commands.append({"name": name, "command": command})
    plan = {"schema": 1, "requested_raw_rows": args.rows, "effective_raw_rows_per_arm": effective,
            "composition": {"mixed_templates": effective // 2, "per_weak_source": per_weak},
            "selection": "Smallest SHA256(seed:pair_id) from bounded scanned population; retained source order is not population-representative.",
            "selection_counts": {"template": template_counts, "weak": weak_counts},
            "evaluation_hashes": {split: hash_file(args.evaluation_dir / f"{split}.jsonl") for split in ["dev", "test"]},
            "dataset_manifests": {key: {"raw_train_rows": value["raw_train_rows"],
                                         "supported_train_rows": value["splits"]["train"]["accepted"],
                                         "publication_allowed": value["publication_allowed"]} for key, value in manifests.items()},
            "comparison_limit": "Raw counts and schedules are equal; supported counts may differ because edit vocabulary and context coverage are measured constraints.",
            "controlled_training": controlled,
            "dataset_hashes": {name: hash_file(datasets / name / "manifest.json") for name in manifests},
            "arms": commands}
    if controlled:
        plan["comparison_limit"] = "Supported training counts are equal after a shared WordPiece budget and deterministic downsampling; all mixed arms train identical rows. Classifier label inventories remain train-derived and may differ by dataset."
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def run(plan):
    # Deliberately foreground and sequential; never launch another arm on failure.
    for arm in plan["arms"]:
        command = arm["command"]
        evaluation = Path(command[command.index("--evaluation-dir") + 1])
        dataset = Path(command[command.index("--data") + 1])
        for split, expected in plan["evaluation_hashes"].items():
            if hash_file(evaluation / f"{split}.jsonl") != expected: raise ValueError("Evaluation population changed after planning")
        if hash_file(dataset / "manifest.json") != plan["dataset_hashes"][dataset.name]:
            raise ValueError("Training manifest changed after planning")
        manifest = json.loads((dataset / "manifest.json").read_text())
        if hash_file(dataset / "train.jsonl") != manifest["splits"]["train"]["sha256"]:
            raise ValueError("Training population changed after planning")
        if hash_file(dataset / "labels.json") != manifest["labels_sha256"]:
            raise ValueError("Training edit vocabulary changed after planning")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path)
    parser.add_argument("--weak", type=Path, action="append")
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-quality-v2"))
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--max-scanned", type=int, default=100000)
    parser.add_argument("--max-labels", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--common-token-budget", type=int, default=0,
                        help="Use equal supported training counts within a shared 4..64 WordPiece budget")
    parser.add_argument("--tokenizer-model", default=TINY[0])
    parser.add_argument("--tokenizer-revision", default=TINY[1])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--scorer", choices=["approximate", "errant"], default="approximate")
    parser.add_argument("--min-dev-edits", type=int, default=25)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-plan", type=Path, help="Execute an already prepared immutable plan")
    args = parser.parse_args()
    if args.run_plan:
        plan = json.loads(args.run_plan.read_text())
    elif args.prepare or args.execute:
        if args.template is None or args.evaluation_dir is None: parser.error("--template and --evaluation-dir are required for preparation")
        plan = build(args)
    else:
        plan = {"mode": "dry-run", "rows": args.rows, "epochs": args.epochs,
                "arms": ["template-tiny64", "mixed-tiny64", "mixed-tiny128", "mixed-larger128"],
                "note": "Use --prepare to build bounded corpora; --execute additionally trains four arms sequentially."}
    print(json.dumps(plan, indent=2))
    if args.execute: run(plan)
