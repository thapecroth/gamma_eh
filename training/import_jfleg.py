"""Pinned official human-reference JFLEG evaluation, never training data."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from pairs import hash_file

REVISION = "ee06ff806a208aba815ac45313f4e750a48330a5"
SOURCE_URL = "https://github.com/keisks/jfleg"
LICENSE = "CC-BY-NC-SA-4.0"
EXPECTED = {"dev": 754, "test": 747}


def fetch_file(name, revision=REVISION):
    url = f"https://raw.githubusercontent.com/keisks/jfleg/{revision}/{name}"
    with urlopen(url, timeout=60) as response:
        value = response.read(4_000_001)
    if len(value) > 4_000_000: raise ValueError("JFLEG file exceeds bounded download limit")
    return value


def materialize(output, fetch=fetch_file, revision=REVISION):
    if output.exists(): raise ValueError("Evaluation output already exists; choose a fresh directory")
    output.mkdir(parents=True)
    originals = output / "original"
    originals.mkdir()
    manifest = {"schema": 1, "dataset": "JFLEG", "revision": revision,
                "source_url": SOURCE_URL, "license": LICENSE, "evaluation_only": True,
                "training_allowed": False, "redistribution": "Separately licensed noncommercial evaluation; keep out of app bundles and training.",
                "attribution": "Napoles, Sakaguchi, and Tetreault (2017), JFLEG: A Fluency Corpus and Benchmark for Grammatical Error Correction.",
                "quality": "Human-written learner sources with four human fluency references; includes stylistic rewrites.",
                "files": {}, "splits": {}}
    for split, expected in EXPECTED.items():
        columns = []
        for suffix in ["src", "ref0", "ref1", "ref2", "ref3"]:
            name = f"{split}/{split}.{suffix}"
            raw = fetch(name, revision)
            path = originals / f"{split}.{suffix}"
            path.write_bytes(raw)
            lines = raw.decode("utf-8").splitlines()
            if len(lines) != expected or any(not line.strip() for line in lines):
                raise ValueError(f"Unexpected {split} {suffix} population; no rows were filtered")
            columns.append(lines)
            manifest["files"][name] = {"bytes": len(raw), "sha256": hash_file(path)}
        path = output / f"{split}.jsonl"
        with path.open("w") as stream:
            for index in range(expected):
                row = {"id": f"jfleg-{split}-{index}", "source": columns[0][index],
                       "references": [column[index] for column in columns[1:]],
                       "origin": "JFLEG", "source_revision": revision, "source_url": SOURCE_URL,
                       "license": LICENSE, "review_status": "human-reference-benchmark", "evaluation_only": True}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"rows": expected, "sha256": hash_file(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/imported/jfleg-evaluation"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(materialize(args.output), indent=2))
    else:
        print(json.dumps({"mode": "dry-run", "dataset": "JFLEG", "revision": REVISION,
                          "evaluation_only": True, "license": LICENSE, "splits": EXPECTED, "output": str(args.output)}))
