"""Bounded streaming import from a pinned public C4_200M parquet mirror."""
import argparse
from collections import Counter
import json
from pathlib import Path

from pairs import hash_file, validate_pair

DATASET = "martinsr/c4_200m"
REVISION = "c46acd6e099d2bf712cd8d0496cb2ba135e786c8"
SOURCE_URL = "https://huggingface.co/datasets/" + DATASET
ORIGINAL_URL = "https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction"


def stream_c4(revision=REVISION):
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as parquet
    # The high-level datasets scanner uses background native reader threads.
    # A bounded early stop can leave them touching Python during finalization.
    # Read small batches synchronously, without native read-ahead, and close the
    # file/generator explicitly when the requested limit is reached.
    filesystem = HfFileSystem()
    files = sorted(name for name in HfApi().list_repo_files(DATASET, repo_type='dataset', revision=revision)
                   if name.startswith('train-') and name.endswith('.parquet'))
    if not files: raise ValueError('Pinned dataset revision has no training parquet files')
    for filename in files:
        path = f'datasets/{DATASET}@{revision}/{filename}'
        with filesystem.open(path, 'rb', block_size=1024 * 1024) as handle:
            reader = parquet.ParquetFile(handle, pre_buffer=False)
            try:
                for batch in reader.iter_batches(batch_size=1024, columns=['input', 'output'], use_threads=False):
                    yield from batch.to_pylist()
            finally:
                reader.close()


def materialize(rows, output, limit, max_scanned, revision=REVISION):
    if limit < 1 or max_scanned < limit:
        raise ValueError("Choose a positive limit and max-scanned at least as large as limit")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "candidates.jsonl"
    if destination.exists(): raise ValueError("Output already exists. Choose a new directory to avoid overwriting a corpus.")
    seen = set()
    counts = Counter()
    iterator = iter(rows)
    try:
        with destination.open("w") as stream:
            for row in iterator:
                if counts["scanned"] >= max_scanned or counts["accepted"] >= limit: break
                counts["scanned"] += 1
                try:
                    source, target = row["input"], row["output"]
                    category = "clean" if " ".join(source.split()) == " ".join(target.split()) else "mixed"
                    pair = validate_pair({"source": source, "target": target, "category": category})
                    if pair["pair_id"] in seen: counts["duplicate"] += 1; continue
                    seen.add(pair["pair_id"])
                    pair.update({"origin": DATASET, "source_revision": revision,
                                 "source_url": SOURCE_URL, "license": "CC-BY-4.0",
                                 "review_status": "unreviewed", "original_source_url": ORIGINAL_URL})
                    stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    counts["accepted"] += 1
                except (ValueError, KeyError, TypeError) as error:
                    counts["rejected:" + (str(error) if isinstance(error, ValueError) else "schema")] += 1
    finally:
        close = getattr(iterator, 'close', None)
        if close: close()
    manifest = {"schema": 1, "dataset": DATASET, "revision": revision,
                "declared_mirror_license": "CC-BY-4.0", "source_url": SOURCE_URL,
                "attribution": "Stahlberg and Kumar (2021), Google C4_200M; parquet conversion by martinsr",
                "original_source_url": ORIGINAL_URL, "quality": "synthetic weak supervision; not reviewed natural errors",
                "license_scope": "Original project licenses corruption edits CC-BY-4.0; preserve source-corpus notices too.",
                "selection": "First valid rows of pinned stream, bounded; not a random sample",
                "counts": dict(counts), "sha256": hash_file(destination)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(counts)), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--max-scanned", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path("data/imported/c4-sample"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "dataset": DATASET, "revision": args.revision,
                          "limit": args.limit, "max_scanned": args.max_scanned, "output": str(args.output)}))
    else:
        materialize(stream_c4(args.revision), args.output, args.limit, args.max_scanned, args.revision)
