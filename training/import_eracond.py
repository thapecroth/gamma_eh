"""Pinned human-annotated conversational pairs, split by dialog before augmentation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import gzip
from pathlib import Path
import tarfile
from urllib.request import urlopen

from pairs import hash_file, normalized, pair_id

REVISION = "8401d3601f58170b55f0b1ca3773329c56b116f9"
SOURCE_URL = "https://github.com/yuanxun-yx/eracond"
ARCHIVE_URL = f"https://codeload.github.com/yuanxun-yx/eracond/tar.gz/{REVISION}"
ARCHIVE_SHA256 = "0d7b1c386a1e6ce051662b6ddd5aa7c0ad2c3653eb8ac550bea5f20e9036410e"
MAX_BYTES = 2_000_000
MAX_TAR_BYTES = 4_000_000
MAX_MEMBERS = 2048


def fetch_archive():
    with urlopen(ARCHIVE_URL, timeout=30) as response:
        value = response.read(MAX_BYTES + 1)
    if len(value) > MAX_BYTES:
        raise ValueError("ErAConD archive exceeds download limit")
    return value


def dialog_split(dialog, seed):
    bucket = int(hashlib.sha256(f"{seed}:{dialog}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def key(text):
    return normalized(text).casefold()


def unpack(blob):
    """Read regular text members in memory; never extract untrusted paths."""
    if len(blob) > MAX_BYTES:
        raise ValueError("ErAConD archive exceeds download limit")
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as compressed:
        expanded = compressed.read(MAX_TAR_BYTES + 1)
    if len(expanded) > MAX_TAR_BYTES:
        raise ValueError("ErAConD decompressed archive exceeds limit")
    files, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
        for index, member in enumerate(archive):
            if index >= MAX_MEMBERS:
                raise ValueError("ErAConD archive exceeds member limit")
            if not member.isfile():
                continue
            total += member.size
            if total > MAX_BYTES:
                raise ValueError("ErAConD uncompressed archive exceeds limit")
            parts = Path(member.name).parts
            if len(parts) < 2 or ".." in parts or Path(member.name).is_absolute():
                raise ValueError("Invalid archive member")
            name = "/".join(parts[1:])
            if name in files:
                raise ValueError("Duplicate archive member")
            files[name] = archive.extractfile(member).read().decode("utf-8")
    if "MIT License" not in files.get("LICENSE", "") or "Copyright (c) 2022 YUAN Xun" not in files["LICENSE"]:
        raise ValueError("Missing expected ErAConD license")
    return files


def materialize(output, fetch=fetch_archive, seed=42, expected_sha256=ARCHIVE_SHA256):
    if output.exists():
        raise ValueError("Choose a fresh ErAConD output directory")
    blob = fetch()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256:
        raise ValueError("ErAConD archive hash mismatch")
    files = unpack(blob)
    counts = Counter()
    populations = {split: [] for split in ("train", "dev", "test")}
    dialogs = defaultdict(list)
    for name in sorted(files):
        if "/orig/" not in name or not name.endswith(".txt"):
            continue
        dialog = name.replace("/orig/", "/")[:-4]
        split = dialog_split(dialog, seed)
        sources = files[name].splitlines()
        references = [files[name.replace("/orig/", "/corr/")].splitlines()]
        second = files.get(name.replace("/orig/", "/corr2/"))
        if second is not None:
            references.append(second.splitlines())
        if any(len(column) != len(sources) for column in references):
            raise ValueError("Misaligned ErAConD dialog")
        dialogs[split].append(dialog)
        for index, source in enumerate(sources):
            refs = list(dict.fromkeys(column[index].strip() for column in references))
            if not source.strip() or any(not ref for ref in refs):
                raise ValueError("Empty ErAConD utterance/reference")
            counts["original_utterances"] += 1
            row = {"id": f"{dialog}:{index}", "source": source.strip(), "target": refs[0],
                   "references": refs, "dialog_id": dialog, "split": split,
                   "category": "clean" if normalized(source) == normalized(refs[0]) else "mixed",
                   "pair_id": pair_id(source.strip(), refs[0]), "origin": "ErAConD",
                   "review_status": "human-reviewed", "license": "MIT",
                   "source_url": SOURCE_URL, "source_revision": REVISION}
            populations[split].append(row)
    if not counts["original_utterances"]:
        raise ValueError("Archive contains no aligned dialog files")
    # Conversation separation alone does not prevent common utterances/references
    # from appearing in different splits. Drop every affected row, on both sides.
    memberships = defaultdict(set)
    for split, rows in populations.items():
        for row in rows:
            for text in [row["source"], *row["references"]]:
                memberships[key(text)].add(split)
    overlap = {text for text, splits in memberships.items() if len(splits) > 1}
    for split, rows in populations.items():
        retained = []
        seen = set()
        for row in rows:
            if any(key(text) in overlap for text in [row["source"], *row["references"]]):
                counts[f"cross_split_overlap_removed:{split}"] += 1
                continue
            signature = (key(row["source"]), tuple(row["references"]))
            if signature in seen:
                counts[f"duplicate_removed:{split}"] += 1
                continue
            seen.add(signature)
            retained.append(row)
        populations[split] = retained
    if any(not rows for rows in populations.values()):
        raise ValueError("Dialog split leaves an empty population")
    output.mkdir(parents=True)
    (output / "LICENSE").write_text(files["LICENSE"])
    manifest = {"schema": 1, "dataset": "ErAConD", "revision": REVISION, "source_url": SOURCE_URL,
                "archive_sha256": digest, "archive_bytes": len(blob), "license": "MIT",
                "attribution": "Yuan, Pham, Davidson, and Yu (2022), ErAConD; copyright 2022 YUAN Xun.",
                "training_allowed": True, "seed": seed, "split_unit": "dialog",
                "split_limitation": "Author identifiers unavailable; dialog disjointness does not establish author disjointness.",
                "reference_policy": "First supplied human correction for training; all supplied corrections for evaluation. All severity levels.",
                "scope": "New dialog-heldout experiment using aligned orig/corr text; not the paper's severity-3 protocol.",
                "counts": dict(counts), "dialogs": dict(dialogs), "splits": {}}
    for split, rows in populations.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][split] = {"rows": len(rows), "sha256": hash_file(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/imported/eracond"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(materialize(args.output, seed=args.seed), indent=2))
    else:
        print(json.dumps({"mode": "dry-run", "source_url": SOURCE_URL, "revision": REVISION,
                          "license": "MIT", "split_unit": "dialog", "output": str(args.output)}))
