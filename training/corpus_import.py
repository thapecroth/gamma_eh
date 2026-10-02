"""Pinned, bounded natural-corpus adapters. No corpus text is committed or relabeled."""
import argparse
from collections import Counter
import codecs
import gzip
import hashlib
import heapq
import io
from itertools import islice
import json
from pathlib import Path
import re
import tarfile
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.parse import urlsplit

from edit_ops import render, tokenize
from pairs import hash_file, normalized, validate_pair

ROOT = Path(__file__).resolve().parent.parent
FORMATS = {"m2-tar", "m2", "tsv", "wiked-tar", "wiked-prefix", "github-gzip", "github-dolt", "c4-parquet"}
NORMALIZATION = "Join standard spaCy English contractions, then edit-schema-2 canonical token rendering; training only"


def canonical(text):
    text = re.sub(r"\b([A-Za-z]+)\s+(n['’]t|['’](?:s|m|d|ll|re|ve))\b", r"\1\2", text, flags=re.IGNORECASE)
    return render(tokenize(text), 2)


def repository_slug(value):
    if value.startswith("https://"):
        parsed = urlsplit(value)
        if parsed.netloc != "github.com" or parsed.query or parsed.fragment:
            raise ValueError("Invalid originating repository URL")
        value = parsed.path.strip("/").removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Invalid originating repository slug")
    return value


def keys(text):
    return {normalized(text).casefold(), normalized(canonical(text)).casefold()}


def exclusions(directories):
    result = set()
    for directory in directories:
        files = [directory] if directory.is_file() else sorted(directory.glob("*.jsonl"))
        if not files:
            raise ValueError("Heldout path has no evaluation JSONL")
        for path in files:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    for value in [row["source"], *row.get("references", [row.get("target")])]:
                        if not isinstance(value, str):
                            raise ValueError("Invalid heldout source/reference")
                        result.update(keys(value))
    return result


def load_spec(name):
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError("Invalid corpus identifier")
    spec = json.loads((Path(__file__).with_name("corpora") / (name + ".json")).read_text())
    if spec.get("id") != name or spec.get("format") not in FORMATS:
        raise ValueError("Invalid corpus specification")
    if spec.get("training_split") != "train" or spec.get("evaluation_only"):
        raise ValueError("Only explicit training corpora can be imported")
    if not isinstance(spec.get("release_eligible"), bool):
        raise ValueError("Explicit source release policy required")
    if spec["format"] == "m2-tar" and any("train" not in Path(member).name.split(".")
                                         for member in spec["members"]):
        raise ValueError("Only named original training members can be imported")
    return spec


def download(spec, destination):
    if destination.exists():
        raise ValueError("Archive destination exists")
    expected = spec.get("sha256")
    if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Automatic downloads require a verified archive SHA256")
    partial = destination.with_suffix(destination.suffix + ".partial")
    total = 0
    with urlopen(spec["download_url"], timeout=60) as source, partial.open("xb") as out:
        while chunk := source.read(min(1024 * 1024, spec.get("prefix_bytes", spec["max_download_bytes"] + 1) - total)):
            total += len(chunk)
            if total > spec["max_download_bytes"]:
                raise ValueError("Archive exceeds download bound")
            out.write(chunk)
            if total == spec.get("prefix_bytes"):
                break
    if hash_file(partial) != expected:
        raise ValueError("Archive hash mismatch")
    partial.replace(destination)
    return destination


def m2_blocks(stream):
    block = []
    for line in stream:
        if len(line) > 1_000_000:
            raise ValueError("Oversized M2 line")
        if line.strip():
            block.append(line.rstrip("\n"))
        elif block:
            yield block
            block = []
    if block:
        yield block


def m2_pairs(stream, path="local.m2"):
    for index, block in enumerate(m2_blocks(stream)):
        if not block[0].startswith("S "):
            raise ValueError("Missing M2 source")
        words = block[0][2:].split()
        annotators = {}
        unresolved = set()
        for line in block[1:]:
            fields = line.split("|||")
            if len(fields) != 6 or not fields[0].startswith("A "):
                raise ValueError("Invalid M2 edit")
            start, end = map(int, fields[0][2:].split())
            annotator = fields[-1]
            edits = annotators.setdefault(annotator, [])
            if fields[1] == "noop":
                if (start, end) != (-1, -1):
                    raise ValueError("Invalid M2 noop")
                continue
            if not 0 <= start <= end <= len(words):
                raise ValueError("Invalid M2 coordinates")
            if fields[1] == "UNK":
                unresolved.add(annotator)
            edits.append((start, end, [] if fields[2] == "-NONE-" else fields[2].split()))
        if not annotators:
            annotators = {"0": []}
        # Keep each alternative separate; never combine different annotators.
        for annotator in sorted(annotators):
            if annotator in unresolved:
                yield {"reject_reason": "unresolved_annotation"}
                continue
            edits = sorted(annotators[annotator], key=lambda e: (e[0], e[1]))
            if any(a[1] > b[0] for a, b in zip(edits, edits[1:])):
                yield {"reject_reason": "overlapping_annotation"}
                continue
            corrected = words[:]
            for start, end, replacement in reversed(edits):
                corrected[start:end] = replacement
            yield {"source": " ".join(words), "target": " ".join(corrected),
                   "corpus_record_id": f"{path}:{index}:{annotator}", "source_path": path}


def archive_m2(path, members):
    wanted = set(members)
    found = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            if not member.isfile() or member.size > 250_000_000:
                raise ValueError("Invalid M2 member")
            found.add(member.name)
            with archive.extractfile(member) as stream:
                yield from m2_pairs(codecs.getreader("utf-8")(stream), member.name)
    if found != wanted:
        raise ValueError("Required official training member missing")


def tsv_pairs(path):
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError("Expected aligned source/target TSV")
            yield {"source": fields[0], "target": fields[1], "corpus_record_id": str(index)}


def wiked_pairs(path, spec, max_scanned, cache):
    # Upstream tar stores source and target in different members. Retain only
    # the bounded prefix, then pair by line index without extracting the tar.
    files = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name not in spec["members"]:
                continue
            if not member.isfile():
                raise ValueError("Invalid WikEd member")
            p = cache / Path(member.name).name
            count = 0
            with archive.extractfile(member) as raw, p.open("w") as out:
                stream = codecs.getreader("utf-8")(raw)
                for line in stream:
                    if len(line) > 1_000_000:
                        raise ValueError("Oversized WikEd line")
                    out.write(line)
                    count += 1
                    if count == max_scanned:
                        break
            files[member.name] = (p, count)
    source_name, target_name = spec["members"]
    if set(files) != {source_name, target_name} or files[source_name][1] != files[target_name][1]:
        raise ValueError("WikEd paired-member population mismatch")
    with files[source_name][0].open() as source, files[target_name][0].open() as target:
        for index, (a, b) in enumerate(zip(source, target)):
            yield {"source": a.rstrip("\n"), "target": b.rstrip("\n"), "corpus_record_id": str(index)}


def github_pairs(path, license_cache):
    # Licenses must be audited at the originating commit; no blanket corpus grant.
    licenses = json.loads(license_cache.read_text()) if license_cache else {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 1_000_000:
                raise ValueError("Oversized GitHub record")
            commit = json.loads(line)
            repo, revision = repository_slug(commit["repo"]), commit["commit"]
            proof = licenses.get(f"{repo}@{revision}")
            for index, edit in enumerate(commit["edits"]):
                if edit["src"].get("lang") != "eng" or edit["tgt"].get("lang") != "eng":
                    yield {"reject_reason": "non_english"}
                    continue
                if not edit.get("is_typo"):
                    yield {"reject_reason": "not_classified_typo"}
                    continue
                if not proof or proof.get("license") not in {"MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0"}:
                    yield {"reject_reason": "source_license_unverified"}
                    continue
                if not proof.get("license_sha256") or proof.get("revision") != revision:
                    raise ValueError("Invalid originating-license proof")
                yield {"source": edit["src"]["text"], "target": edit["tgt"]["text"],
                       "license": proof["license"], "source_revision": revision,
                       "source_url": f"https://github.com/{repo}/commit/{revision}",
                       "source_path": edit["src"]["path"],
                       "source_provenance": proof,
                       "corpus_record_id": f"{repo}:{revision}:{index}"}


def wdiff_pair(text):
    # Exact reconstruction from the authors' wdiff_to_parallel.py.
    source = re.sub(r" *\{\+.*?\+\} *", " ", re.sub(r" *\[-(.*?)-\] *", r" \1 ", text)).strip()
    target = re.sub(r" *\[-.*?-\] *", " ", re.sub(r" *\{\+(.*?)\+\} *", r" \1 ", text)).strip()
    if any(marker in source + target for marker in ("[-", "-]", "{+", "+}")):
        raise ValueError("Incomplete WikEd markup")
    return source, target


def wiked_prefix_pairs(path, spec):
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name != spec["member"]:
                continue
            if not member.isfile():
                raise ValueError("Invalid WikEd nested member")
            metadata = {}
            with gzip.GzipFile(fileobj=archive.extractfile(member)) as nested, io.TextIOWrapper(nested, encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    if len(line) > 1_000_000:
                        raise ValueError("Oversized WikEd record")
                    if line.startswith("###"):
                        key, _, value = line[3:].strip().partition(":")
                        if key.strip() in {"id", "page", "timestamp"}:
                            metadata[key.strip()] = value.strip()
                        continue
                    if not line.strip():
                        continue
                    try:
                        source, target = wdiff_pair(line.replace("\t", " ").strip())
                    except ValueError:
                        yield {"reject_reason": "invalid_wdiff"}
                        continue
                    yield {"source": source, "target": target, "source_path": member.name,
                           "source_provenance": {"wiki_metadata": dict(metadata),
                                                 "dump": "enwiki-20140304", "member": member.name},
                           "corpus_record_id": f"{member.name}:{index}"}
            return
    raise ValueError("Pinned WikEd member missing")


def dolt_rows(spec, max_scanned):
    revision = spec["revision"]
    if not re.fullmatch(r"[a-z0-9]{32}", revision):
        raise ValueError("Invalid immutable Dolt revision")
    repos = spec.get("repositories", [])
    if not repos or any(not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) for repo in repos):
        raise ValueError("Explicit repository sampling scope required")
    remaining = max_scanned
    # Equality-scoped requests avoid sorting the entire multi-repository result
    # on the public API, while preserving a deterministic repository prefix.
    for repo in sorted(repos):
        offset = 0
        while remaining:
            limit = min(500, remaining)
            query = (f"SELECT * FROM edits AS OF '{revision}' WHERE repo='https://github.com/{repo}' "
                     "AND source_lang='eng' AND target_lang='eng' AND is_typo=1 "
                     "AND (LOWER(source_path) LIKE '%.md' OR LOWER(source_path) LIKE '%.rst' OR LOWER(source_path) LIKE '%.txt') "
                     f"ORDER BY commit_hash,edit_number LIMIT {limit} OFFSET {offset}")
            with urlopen(spec["api_url"] + "?" + urlencode({"q": query}), timeout=60) as response:
                value = json.loads(response.read(8_000_001))
            if value.get("query_execution_status") != "Success":
                raise ValueError("Pinned Dolt query failed; no partial corpus import")
            rows = value["rows"]
            if len(rows) > limit:
                raise ValueError("Dolt response exceeded requested bound")
            for row in rows:
                row["repo"] = repository_slug(row["repo"])
                if row["repo"] != repo or not re.fullmatch(r"[0-9a-f]{40}", row["commit_hash"]):
                    raise ValueError("Invalid originating repository/commit")
            yield from rows
            remaining -= len(rows)
            if len(rows) < limit:
                break
            offset += limit
        if not remaining:
            break


def dolt_pairs(spec, max_scanned, license_cache):
    licenses = json.loads(license_cache.read_text()) if license_cache else {}
    for row in dolt_rows(spec, max_scanned):
        repo, revision = row["repo"], row["commit_hash"]
        proof = licenses.get(f"{repo}@{revision}")
        if not proof or proof.get("license") not in {"MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0"}:
            yield {"reject_reason": "source_license_unverified"}
            continue
        if not proof.get("license_sha256") or proof.get("revision") != revision:
            raise ValueError("Invalid originating-license proof")
        yield {"source": row["source_text"], "target": row["target_text"],
               "license": proof["license"], "source_revision": revision,
               "source_url": f"https://github.com/{repo}/commit/{revision}",
               "source_path": row["source_path"], "source_provenance": proof,
               "corpus_record_id": f"{repo}:{revision}:{row['edit_number']}"}


def source_rows(spec, path, max_scanned, output, license_cache=None):
    kind = spec["format"]
    if kind == "m2-tar":
        return archive_m2(path, spec["members"])
    if kind == "m2":
        def rows():
            with path.open(encoding="utf-8") as stream:
                yield from m2_pairs(stream, path.name)
        return rows()
    if kind == "tsv":
        return tsv_pairs(path)
    if kind == "wiked-tar":
        return wiked_pairs(path, spec, max_scanned, output)
    if kind == "wiked-prefix":
        return wiked_prefix_pairs(path, spec)
    if kind == "github-gzip":
        return github_pairs(path, license_cache)
    if kind == "github-dolt":
        return dolt_pairs(spec, max_scanned, license_cache)
    if kind == "c4-parquet":
        from import_c4 import stream_c4
        def rows():
            upstream = stream_c4(spec["revision"])
            try:
                for index, row in enumerate(upstream):
                    yield {"source": row["input"], "target": row["output"], "corpus_record_id": str(index)}
            finally:
                upstream.close()
        return rows()
    raise ValueError("Unsupported source format")


def materialize(spec, rows, output, heldout, limit=10000, max_scanned=100000, seed=42, archive_hash=None):
    if limit < 1 or max_scanned < limit:
        raise ValueError("Positive bounded import schedule required")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "candidates.jsonl").exists() or (output / "manifest.json").exists():
        raise ValueError("Corpus output already exists")
    counts = Counter()
    heap, seen, conflicts = [], {}, set()
    iterator = iter(rows)
    try:
        for raw in islice(iterator, max_scanned):
            counts["scanned"] += 1
            if raw.get("reject_reason"):
                counts["rejected:" + raw["reject_reason"]] += 1
                continue
            try:
                if raw.get("evaluation_only") or spec["id"] in {"jfleg", "cweb"}:
                    raise ValueError("evaluation_only")
                category = "clean" if normalized(raw["source"]) == normalized(raw["target"]) else "mixed"
                original = validate_pair({**raw, "category": category})
                if keys(original["source"]) & heldout or keys(original["target"]) & heldout:
                    raise ValueError("heldout_overlap")
                source, target = canonical(original["source"]), canonical(original["target"])
                if source == target and category != "clean":
                    raise ValueError("normalization_erased_edit")
                pair = validate_pair({"source": source, "target": target, "category": category})
                if len(tokenize(target)) < 4 or len(tokenize(source)) < 4:
                    raise ValueError("fragment")
                source_key = normalized(source).casefold()
                if source_key in seen:
                    if seen[source_key] != target:
                        conflicts.add(source_key)
                        counts["conflicting_source"] += 1
                    else:
                        counts["duplicate"] += 1
                    continue
                seen[source_key] = target
                row = {**pair, "origin": spec["id"], "source_split": "train",
                       "supervision": spec["supervision"], "review_status": "corpus-supervision",
                       "license": raw.get("license", spec["license"]),
                       "license_scope": spec["license_scope"], "release_eligible": spec["release_eligible"],
                       "source_url": raw.get("source_url", spec["source_url"]),
                       "source_revision": raw.get("source_revision", archive_hash or spec.get("revision")),
                       "normalization": NORMALIZATION,
                       "corpus_record_id": raw.get("corpus_record_id", str(counts["scanned"]))}
                row.update({key: raw[key] for key in ("source_path", "source_provenance") if key in raw})
                priority = int(hashlib.sha256(f"{seed}:{pair['pair_id']}".encode()).hexdigest(), 16)
                entry = (-priority, source_key, row)
                if len(heap) < limit:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
                counts["valid"] += 1
            except (ValueError, KeyError, TypeError) as error:
                reason = str(error) if isinstance(error, ValueError) else "schema"
                counts["rejected:" + reason] += 1
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
    selected = [row for _, key, row in sorted(heap, reverse=True) if key not in conflicts]
    counts["selected_original_pairs"] = len(selected)
    # Corrected targets become train-only clean controls in the same clean group.
    controls = {}
    for row in selected:
        if row["category"] != "clean" and int(row["pair_id"][:8], 16) % 4 == 0:
            pair = validate_pair({"source": row["target"], "target": row["target"], "category": "clean"})
            if normalized(pair["source"]).casefold() not in seen:
                controls.setdefault(pair["pair_id"], {**row, **pair, "supervision": "corrected-target-clean-control"})
    selected.extend(controls[key] for key in sorted(controls))
    counts["derived_clean_controls"] = len(controls)
    destination = output / "candidates.jsonl"
    with destination.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema": 1, "corpus": spec["id"], "source_spec": spec,
                "source_archive_sha256": archive_hash, "counts": dict(counts), "seed": seed,
                "sha256": hash_file(destination), "heldout_keys": len(heldout),
                "normalization": NORMALIZATION, "publication_allowed": spec["release_eligible"],
                "sampling": "Lowest seeded pair hashes within bounded scanned prefix; not a representative full-corpus claim.",
                "training_split_only": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not selected:
        raise ValueError("No eligible training pairs; import is not a successful corpus run")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus")
    p.add_argument("--input", type=Path, help="Owner-provided archive/M2/aligned TSV")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heldout", type=Path, action="append", required=True)
    p.add_argument("--license-cache", type=Path)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--max-scanned", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    spec = load_spec(args.corpus)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "spec": spec, "rows": args.limit,
                          "max_scanned": args.max_scanned, "input": str(args.input), "output": str(args.output)}, indent=2))
        return
    args.output.resolve().relative_to(ROOT / "data/imported")
    if args.output.exists():
        raise ValueError("Output exists; choose a fresh directory")
    args.output.mkdir(parents=True)
    path = args.input
    if path is None and spec["format"] not in {"c4-parquet", "github-dolt"}:
        if not spec.get("download_url"):
            raise ValueError("Owner-provided training archive required; see source access instructions")
        path = download(spec, args.output / "upstream.archive")
    archive_hash = hash_file(path) if path else None
    if spec.get("sha256") and archive_hash != spec["sha256"]:
        raise ValueError("Pinned source archive hash mismatch")
    manifest = materialize(spec, source_rows(spec, path, args.max_scanned, args.output, args.license_cache),
                           args.output, exclusions(args.heldout), args.limit, args.max_scanned, args.seed, archive_hash)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
