"""Conservative inverse corruption and a provenance-preserving human/rule mixture."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from data import build as build_templates
from import_eracond import key
from pairs import evaluation_keys, hash_file, normalized, pair_id, validate_pair
from prepare_pairs import prepare

# Spelling inversions change only a known correct spelling into a nonword.
MISSPELLINGS = {"receive": "recieve", "received": "recieved", "believe": "beleive",
               "friends": "freinds", "separate": "seperate", "necessary": "neccessary",
               "definitely": "definately", "tomorrow": "tommorow", "because": "becuase",
               "writing": "writting", "grammar": "grammer", "address": "adress",
               "available": "avaliable", "really": "realy", "successful": "succesful",
               "different": "diffrent", "business": "buisness", "important": "importent"}
AGREEMENT = re.compile(r"^(I|He|She|It|We|They|You)([ \t]+)(am|is|are|has|have)\b")
EXPECTED = {"I": ("am", "have"), "He": ("is", "has"), "She": ("is", "has"),
            "It": ("is", "has"), "We": ("are", "have"), "They": ("are", "have"),
            "You": ("are", "have")}
WRONG = {"am": "is", "is": "are", "are": "is", "has": "have", "have": "has"}
WORD = re.compile(r"\b[A-Za-z]+\b")
PROTECTED = re.compile(r"https?://|\b[^\s@]+@[^\s@]+\.[^\s@]+|[`\"“”]|(?<!\w)['‘](?=\S)")


def preserve_case(word, replacement):
    if word.isupper():
        return replacement.upper()
    return replacement.capitalize() if word[0].isupper() else replacement


def corrupt(clean):
    """Only local, justified errors; no arbitrary tense/preposition/article flips."""
    if PROTECTED.search(clean):
        return []
    variants = []
    for match in WORD.finditer(clean):
        wrong = MISSPELLINGS.get(match.group().lower())
        if wrong:
            source = clean[:match.start()] + preserve_case(match.group(), wrong) + clean[match.end():]
            variants.append((source, "spelling", "known-nonword-spelling"))
    match = AGREEMENT.match(clean)
    if match and match[3] in EXPECTED[match[1]]:
        source = clean[:match.start(3)] + WRONG[match[3]] + clean[match.end(3):]
        variants.append((source, "agreement", "sentence-initial-pronoun-agreement"))
    # A duplicated determiner before a simple word is an accidental repetition,
    # unlike valid 'had had', 'that that', quoted speech, or arbitrary repeated words.
    for match in re.finditer(r"\b(the|a|an)([ \t]+)([A-Za-z]+)\b", clean, re.I):
        if match[3].lower() in {"the", "a", "an", "that", "had"}:
            continue
        source = clean[:match.start()] + match[1] + match[2] + clean[match.start():]
        variants.append((source, "mixed", "duplicate-determiner"))
    return sorted(set(variants))


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def corruption_pairs(clean, metadata):
    yield {**metadata, "source": clean, "target": clean, "category": "clean", "rule": "unchanged"}
    for source, category, rule in corrupt(clean):
        yield {**metadata, "source": source, "target": clean, "category": category, "rule": rule}


def seed_pairs(seeds):
    for seed in seeds:
        clean = seed["source"]
        if seed.get("review_status") == "human-reviewed" or seed.get("license") != "CC0-1.0":
            raise ValueError("Bundled agent seeds must remain unreviewed CC0 candidates")
        metadata = {"license": seed["license"], "review_status": "unreviewed",
                    "origin": "rule-corruption-v1", "seed_id": seed["id"], "family": seed["family"],
                    "seed_origin": seed["origin"]}
        yield from corruption_pairs(clean, metadata)


def human_evaluation(rows, split):
    """Keep complete human references, plus separately labeled clean controls."""
    result = []
    controls = set()
    for row in rows:
        result.append({**row, "evaluation_only": True})
        for index, reference in enumerate(row["references"]):
            if key(reference) in controls:
                continue
            controls.add(key(reference))
            result.append({**row, "id": f"{row['id']}:clean:{index}", "source": reference,
                           "target": reference, "references": [reference], "category": "clean",
                           "origin": "ErAConD-human-reference-clean-control",
                           "review_status": "human-reference-derived-control", "evaluation_only": True,
                           "pair_id": pair_id(reference, reference), "split": split})
    return result


def build(output, human_corpus, seeds, counterexamples, reserved_evaluation=(), max_rows=4096,
          max_labels=512, seed=42, template_data=None):
    if output.exists() or max_rows < 1 or max_labels < 2:
        raise ValueError("Fresh output and positive row/label budgets required")
    human_manifest = json.loads((human_corpus / "manifest.json").read_text())
    if human_manifest.get("dataset") != "ErAConD" or human_manifest.get("license") != "MIT" or not human_manifest.get("training_allowed"):
        raise ValueError("Expected separately licensed ErAConD training corpus")
    license_text = (human_corpus / "LICENSE").read_text()
    if "MIT License" not in license_text or "Copyright (c) 2022 YUAN Xun" not in license_text:
        raise ValueError("Human corpus must retain the original MIT notice")
    for split in ("train", "dev", "test"):
        if hash_file(human_corpus / f"{split}.jsonl") != human_manifest["splits"][split]["sha256"]:
            raise ValueError("Human corpus split hash mismatch")
    populations = {split: human_evaluation(list(read_rows(human_corpus / f"{split}.jsonl")), split)
                   for split in ("dev", "test")}
    excluded = evaluation_keys([row for rows in populations.values() for row in rows])
    diagnostic = json.loads(counterexamples.read_text())
    if diagnostic.get("review_status") != "unreviewed":
        raise ValueError("Agent counterexamples must not be labeled human-reviewed")
    excluded.update(evaluation_keys(diagnostic["cases"]))
    reserved_hashes = {}
    for directory in reserved_evaluation:
        for split in ("dev", "test"):
            path = directory / f"{split}.jsonl"
            if path.exists():
                excluded.update(evaluation_keys(list(read_rows(path))))
                reserved_hashes[str(path)] = hash_file(path)
    # Weak examples never create development/test labels. The natural dialog
    # populations and diagnostic set are reserved before generating variants.
    candidates = list(read_rows(human_corpus / "train.jsonl"))
    human_variants = []
    for row in candidates:
        # These are new weak synthetic labels, not additional human annotations.
        metadata = {"license": "MIT", "review_status": "unreviewed", "origin": "rule-corruption-v1",
                    "seed_id": row["id"], "seed_origin": "ErAConD-human-reference",
                    "dialog_id": row["dialog_id"], "source_url": row["source_url"],
                    "source_revision": row["source_revision"]}
        human_variants.extend(corruption_pairs(row["target"], metadata))
    candidates.extend(human_variants)
    candidates.extend(seed_pairs(list(read_rows(seeds))))
    # Reuse the established template generator as additional bounded coverage.
    output.mkdir(parents=True)
    shutil.copyfile(human_corpus / "LICENSE", output / "ERACOND-LICENSE")
    templates = template_data or output / "templates"
    if template_data is None:
        build_templates(templates)
    template_manifest = json.loads((templates / "manifest.json").read_text())
    if template_manifest.get("license") != "CC0-1.0" or template_manifest.get("origin") != "original-template-v1":
        raise ValueError("Expected original CC0 templates; external data needs separate provenance")
    if hash_file(templates / "train.jsonl") != template_manifest["splits"]["train"]["sha256"]:
        raise ValueError("Template training split hash mismatch")
    for row in read_rows(templates / "train.jsonl"):
        # 'He read ...' can be grammatical past tense without an explicit cue.
        if any(word.lower() == "read" and tag in {"REPLACE:reads", "SUFFIX:ADD_S"}
               for word, tag in zip(row["tokens"], row["tags"])):
            continue
        candidates.append({**row, "license": "CC0-1.0", "review_status": "unreviewed",
                           "category": "clean" if normalized(row["source"]) == normalized(row["target"]) else "mixed"})
    counts = Counter()
    unique, conflicts = {}, set()
    for raw in candidates:
        counts["raw_candidates"] += 1
        try:
            row = {**raw, **validate_pair(raw)}
        except (ValueError, KeyError, TypeError) as error:
            counts["invalid_candidate"] += 1
            if isinstance(error, ValueError):
                counts["invalid:" + str(error)] += 1
            continue
        if any(key(value) in excluded for value in [row["source"], row["target"], *row.get("references", [])]):
            counts["heldout_overlap_removed"] += 1
            continue
        source_key = key(row["source"])
        if source_key in unique and normalized(unique[source_key]["target"]) != normalized(row["target"]):
            conflicts.add(source_key)
        elif source_key not in unique or row["review_status"] == "human-reviewed":
            unique[source_key] = row
        else:
            counts["duplicate_removed"] += 1
    records = [row for source, row in unique.items() if source not in conflicts]
    counts["conflicting_sources_removed"] = len(conflicts)
    order = lambda row: hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).hexdigest()
    human = sorted([row for row in records if row["origin"] == "ErAConD"], key=order)
    rules = sorted([row for row in records if row["origin"] == "rule-corruption-v1"], key=order)
    # Retain both natural and broader-context rule examples before filling with
    # old templates. This is bounded selection, not a claim of representativeness.
    selected = human[:max_rows // 2] + rules[:max_rows // 4]
    used = {row["pair_id"] for row in selected}
    remaining = sorted([row for row in records if row["pair_id"] not in used], key=order)
    selected.extend(remaining[:max_rows - len(selected)])
    if not any(row["origin"] == "ErAConD" for row in selected) or not any(row["origin"] == "rule-corruption-v1" for row in selected):
        raise ValueError("Mixture needs both natural human corrections and rule examples")
    raw_path = output / "mixture.jsonl"
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(selected, key=order)))
    prepared = output / "prepared"
    manifest = prepare([raw_path], prepared, allow_weak_train=True, max_labels=max_labels, schema=2, train_only=True)
    shutil.copyfile(human_corpus / "LICENSE", prepared / "ERACOND-LICENSE")
    for split, rows in populations.items():
        path = prepared / "evaluation" / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["evaluation"][split] = {"rows": len(rows), "sha256": hash_file(path),
                                          "original_utterances": sum(row["origin"] == "ErAConD" for row in rows),
                                          "reference_clean_controls": sum(row["category"] == "clean" and row["origin"] != "ErAConD" for row in rows)}
    manifest.update({"evaluation_ready": True, "evaluation_scope": "Dialog-heldout ErAConD human corrections and human-reference clean controls; agent diagnostics separate.",
                     "human_corpus_manifest_sha256": hash_file(human_corpus / "manifest.json"),
                     "counterexamples_sha256": hash_file(counterexamples), "seeds_sha256": hash_file(seeds),
                     "template_manifest_sha256": hash_file(templates / "manifest.json"),
                     "reserved_evaluation_hashes": reserved_hashes, "mixture_seed": seed,
                     "mixture_counts": dict(counts), "selected_raw_origins": dict(Counter(row["origin"] for row in selected)),
                     "selected_raw_rows": len(selected), "labels_sha256": hash_file(prepared / "labels.json"),
                     "agent_examples_human_reviewed": False,
                     "human_attribution": human_manifest["attribution"],
                     "license_files": {"ERACOND-LICENSE": hash_file(prepared / "ERACOND-LICENSE")},
                     "generator_sha256": hash_file(Path(__file__))})
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--human-corpus", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, default=Path("data/rule-seeds.jsonl"))
    parser.add_argument("--counterexamples", type=Path, default=Path("data/counterexamples.json"))
    parser.add_argument("--reserved-evaluation", type=Path, action="append", default=[])
    parser.add_argument("--template-data", type=Path)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--max-labels", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.human_corpus, args.seeds, args.counterexamples,
                           args.reserved_evaluation, args.max_rows, args.max_labels, args.seed, args.template_data), indent=2))
