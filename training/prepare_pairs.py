"""Prepare tagged training data; weak labels are train-only, never dev/test."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from data import edit_tags, render, tokens
from edit_ops import reconstruct
from pairs import group_id, hash_file, normalized, validate_pair
from verified_teacher import MODEL, blind_id, blind_reason, critic_id, critic_reason

LICENSES = {"CC0-1.0", "CC-BY-4.0", "Apache-2.0", "MIT"}
RESEARCH_LICENSES = {"CC-BY-NC-SA-4.0", "CC-BY-SA-3.0",
                     "LicenseRef-Cambridge-WI", "LicenseRef-Cambridge-FCE",
                     "LicenseRef-NUCLE", "LicenseRef-Lang8"}
SOURCE_FIELDS = ("source_split", "supervision", "corpus_record_id", "source_path",
                 "license_scope", "release_eligible", "source_provenance", "normalization")
TEACHER_ORIGINS = {"llm-teacher", "glm-recipe-machine-screened"}
SCREENING_FIELDS = ("row_id", "recipe_id", "recipe_version", "family_id", "split", "register", "context",
                    "job_id", "run_fingerprint", "mutation", "guard_coverage", "blind_review", "pair_review", "critic_request_id",
                    "status", "screening_status", "admission_status", "human_reviewed", "weak_supervision")


def prepare(inputs, output, allow_weak_train=False, teacher_license=None, max_labels=4096,
            allow_unverified_teacher_terms=False, schema=1, allow_research=False):
    if teacher_license is not None and teacher_license not in LICENSES:
        raise ValueError("Unsupported teacher output license")
    if max_labels < 2: raise ValueError("max-labels must be at least2")
    if schema not in {1, 2}: raise ValueError("Unsupported edit schema")
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
    provenance = {}
    publication_allowed = True
    try:
        for path in inputs:
            with path.open() as stream:
                for line in stream:
                    counts["scanned"] += 1
                    try:
                        raw = json.loads(line)
                        if not isinstance(raw, dict): raise ValueError("schema")
                        if raw.get("evaluation_only"):
                            raise ValueError("evaluation_only")
                        if (raw.get("review_status") == "quarantined"
                                or raw.get("status") in {"quarantined", "unsupported", "duplicate"}
                                or "admission_status" in raw and raw["admission_status"] != "screened"):
                            raise ValueError("not_admitted")
                        recipe_teacher = raw.get("origin") == "glm-recipe-machine-screened"
                        if recipe_teacher:
                            if raw.get("split") != "train": raise ValueError("teacher_heldout")
                            if (raw.get("admission_status") != "screened" or raw.get("status") != "screened"
                                    or raw.get("review_status") != "machine-verified"
                                    or raw.get("human_reviewed") is not False or raw.get("weak_supervision") is not True
                                    or not isinstance(raw.get("family_id"), str)
                                    or len(raw["family_id"]) != 64
                                    or any(char not in "0123456789abcdef" for char in raw["family_id"])
                                    or raw.get("model") != MODEL or not isinstance(raw.get("row_id"), str)
                                    or not isinstance(raw.get("run_fingerprint"), str) or len(raw["run_fingerprint"]) != 64
                                    or any(char not in "0123456789abcdef" for char in raw["run_fingerprint"])
                                    or not isinstance(raw.get("blind_review"), dict) or not isinstance(raw.get("pair_review"), dict)):
                                raise ValueError("teacher_evidence_missing")
                            if (raw["blind_review"].get("id") != blind_id(raw, raw["run_fingerprint"])
                                    or raw["pair_review"].get("id") != raw["row_id"]
                                    or "critic_request_id" in raw and raw["critic_request_id"] != critic_id(raw, raw["run_fingerprint"])
                                    or blind_reason(raw, raw["blind_review"]) or critic_reason(raw, raw["pair_review"])):
                                raise ValueError("teacher_evidence_missing")
                        row = validate_pair(raw)
                        license_id = raw.get("license")
                        if raw.get("origin") in TEACHER_ORIGINS and teacher_license: license_id = teacher_license
                        local_teacher = (allow_unverified_teacher_terms and allow_weak_train
                                         and raw.get("origin") in TEACHER_ORIGINS
                                         and license_id == "provider-terms-unverified")
                        research = allow_research and license_id in RESEARCH_LICENSES
                        if license_id not in LICENSES and not local_teacher and not research:
                            raise ValueError("license_unverified")
                        publication_allowed &= (license_id in LICENSES and raw.get("release_eligible", True))
                        reviewed = raw.get("review_status") == "human-reviewed" and not local_teacher
                        if not reviewed and not allow_weak_train: raise ValueError("unreviewed")
                        row.update({"license": license_id,
                                    "origin": raw.get("origin", "unspecified"),
                                    "review_status": "human-reviewed" if reviewed else "weak-supervision",
                                    "source_revision": raw.get("source_revision"), "model": raw.get("model"),
                                    "source_url": raw.get("source_url"),
                                    "original_source_url": raw.get("original_source_url"),
                                    "prompt_sha256": raw.get("prompt_sha256")})
                        row.update({key: raw[key] for key in SOURCE_FIELDS if key in raw})
                        if recipe_teacher:
                            row.update({name: raw[name] for name in SCREENING_FIELDS if name in raw})
                            row["clean_group"] = raw["family_id"]
                        source_metadata = {name: row.get(name) for name in ["origin", "license", "source_revision", "source_url", "original_source_url", "model", "prompt_sha256"]}
                        source_metadata.update({key: row[key] for key in SOURCE_FIELDS
                                                if key in row and key != "corpus_record_id"})
                        if recipe_teacher: source_metadata["run_fingerprint"] = row["run_fingerprint"]
                        if row["origin"] == "martinsr/c4_200m":
                            source_metadata.update({"attribution": "Stahlberg and Kumar (2021), Google C4_200M; parquet conversion by martinsr",
                                                    "license_scope": "Corruption edits CC-BY-4.0; source-corpus notices also apply."})
                        provenance[json.dumps(source_metadata, sort_keys=True)] = source_metadata
                        # Retain the complete reviewed population before alignment;
                        # unsupported references still define missed corrections.
                        row["evaluation"] = {"source": raw["source"], "references": [raw["target"]],
                                             "category": row["category"], "origin": row["origin"],
                                             "license": license_id, "review_status": row["review_status"],
                                             "pair_id": row["pair_id"]}
                        try:
                            words, tags = edit_tags(row["source"], row["target"], schema)
                            if all(tag == 'KEEP' for tag in tags) and normalized(row['source']) != normalized(row['target']):
                                raise ValueError('unrepresentable_spacing_change')
                            expected = render(tokens(row["target"])) if schema == 1 else row["target"]
                            # Runtime KEEP makes no edits and preserves the original
                            # source, including punctuation/spacing the canonical
                            # token renderer cannot reproduce.
                            exact_identity = row["source"] == row["target"] and all(tag == "KEEP" for tag in tags)
                            if not exact_identity and normalized(reconstruct(words, tags, schema)) != normalized(expected):
                                raise ValueError("unrepresentable_case_or_spacing_change")
                            row.update({"tokens": words, "tags": tags})
                        except ValueError as error:
                            reason = str(error)
                            if reason.startswith("Pair cannot"): reason = "unrepresentable_alignment"
                            row["alignment_reason"] = reason
                            counts["rejected"] += 1
                            counts["rejected:" + reason] += 1
                        source_key = hashlib.sha256(normalized(row["source"]).casefold().encode()).hexdigest()
                        target_key = hashlib.sha256(normalized(row["target"]).encode()).hexdigest()
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
                    except (ValueError, KeyError, TypeError) as error:
                        counts["rejected"] += 1
                        reason = str(error) if isinstance(error, ValueError) else "schema"
                        if reason not in {"evaluation_only", "license_unverified", "unreviewed", "schema", "category", "category_mismatch", "length", "control_character", "no_english_letters", "contact_or_secret_pattern", "clean_mismatch", "word_limit", "not_admitted", "teacher_heldout", "teacher_evidence_missing"}:
                            reason = "schema"
                        counts["rejected:" + reason] += 1
            db.commit()
        # If any variant is reviewed, its entire clean group follows the reviewed
        # split. Unreviewed variants assigned dev/test are dropped, never moved to
        # train: moving them would leak their held-out target into training.
        db.execute("CREATE TABLE reviewed_groups AS SELECT DISTINCT clean_group FROM records WHERE reviewed=1 AND conflict=0")
        db.execute("CREATE UNIQUE INDEX reviewed_group_index ON reviewed_groups(clean_group)")
        split_rows = Counter()
        tag_counts = Counter()
        streams = {s: (output / f"{s}.jsonl").open("w") for s in ["train", "dev", "test"]}
        evaluation_dir = output / "evaluation"
        evaluation_dir.mkdir()
        eval_streams = {s: (evaluation_dir / f"{s}.jsonl").open("w") for s in ["dev", "test"]}
        categories = {s: {} for s in ["train", "dev", "test"]}
        population = Counter()
        try:
            query = "SELECT r.value,r.reviewed,g.clean_group FROM records r LEFT JOIN reviewed_groups g ON r.clean_group=g.clean_group WHERE r.conflict=0 ORDER BY r.clean_group,r.source_key"
            for value, reviewed, reviewed_group in db.execute(query):
                row = json.loads(value)
                bucket = int(row["clean_group"][:8], 16) % 100
                split = "train" if not reviewed_group or bucket < 80 else "dev" if bucket < 90 else "test"
                if split != "train" and not reviewed:
                    counts["weak_heldout_group_dropped"] += 1
                    continue
                population[split] += 1
                category = categories[split].setdefault(row["category"], Counter())
                category["population"] += 1
                if split in eval_streams:
                    eval_streams[split].write(json.dumps(row["evaluation"], ensure_ascii=False) + "\n")
                if "tags" not in row:
                    category["unrepresentable"] += 1
                    continue
                category["taggable"] += 1
                row.pop("evaluation")
                if split == "train": tag_counts.update(row["tags"])
                streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                split_rows[split] += 1
        finally:
            for stream in streams.values(): stream.close()
            for stream in eval_streams.values(): stream.close()
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
                    if all(tag in supported for tag in row["tags"]):
                        destination.write(line); kept += 1
                        categories[split][row["category"]]["supported"] += 1
                    else:
                        dropped += 1
                        categories[split][row["category"]]["unsupported_vocabulary"] += 1
            replacement.replace(filename)
            coverage[split] = {"accepted": kept, "population": population[split],
                               "unsupported_tag_examples": dropped,
                               "sha256": hash_file(filename)}
        (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
        manifest = {"schema": 1, "edit_schema": schema, "kind": "prepared-edit-tags", "inputs": [str(path) for path in inputs],
                    "input_files": [{"path": str(path), "sha256": hash_file(path)} for path in inputs],
                    "source_provenance": [provenance[key] for key in sorted(provenance)],
                    "counts": dict(counts), "origins_before_vocabulary_filter": dict(origins),
                    "licenses": dict(license_counts), "weak_labels_train_only": True,
                    "publication_allowed": publication_allowed,
                    "training_purpose": "local-research" if allow_research else "permissive-training",
                    "label_count": len(labels), "splits": coverage,
                    "coverage_by_category": categories,
                    "evaluation": {s: {"rows": population[s], "sha256": hash_file(evaluation_dir / f"{s}.jsonl")}
                                   for s in ["dev", "test"]},
                    "evaluation_ready": all(population[s] > 0 for s in ["dev", "test"])}
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
    parser.add_argument("--allow-unverified-teacher-terms", action="store_true",
                        help="Local experiments only: retain unverified terms and block publication; requires --allow-weak-train")
    parser.add_argument("--max-labels", type=int, default=4096)
    parser.add_argument("--schema", type=int, choices=[1, 2], default=1)
    parser.add_argument("--allow-research", action="store_true",
                        help="Retain restricted corpus terms for local experiments; block weight publication")
    args = parser.parse_args()
    prepare(args.input, args.output, args.allow_weak_train, args.teacher_license, args.max_labels,
            args.allow_unverified_teacher_terms, args.schema, args.allow_research)
