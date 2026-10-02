"""Pinned natural benchmarks: official JFLEG GLEU and standard CWEB ERRANT."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.request import urlopen

from import_jfleg import materialize, REVISION as JFLEG_REVISION
from pairs import normalized

ROOT = Path(__file__).resolve().parents[1]
CWEB_REVISION = "08d1da0ff2b78885b1f060b4fa11460a98eb9218"
GLEU_SHA256 = "b8bf3605b3c23a899734b406855dcfd8404ec7d8aa84cdfb25114439e5af9c29"
MODES = ("rules", "model", "combined")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def download(url, limit=4_000_000):
    with urlopen(url, timeout=60) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Source exceeds the bounded download limit")
    return raw


def m2_rows(raw):
    """Recover all references while preserving the original token coordinates."""
    result = []
    for block in raw.decode("utf-8").strip().split("\n\n"):
        lines = block.splitlines()
        if not lines or not lines[0].startswith("S "):
            raise ValueError("Invalid M2 source block")
        source = lines[0][2:]
        tokens = source.split()
        annotators = {}
        for line in lines[1:]:
            fields = line.split("|||")
            if len(fields) != 6 or not fields[0].startswith("A "):
                raise ValueError("Invalid M2 edit")
            start, end = map(int, fields[0][2:].split())
            edits = annotators.setdefault(fields[-1], [])
            if fields[1] == "noop":
                if (start, end) != (-1, -1):
                    raise ValueError("Invalid noop coordinates")
                continue
            if not 0 <= start <= end <= len(tokens):
                raise ValueError("Out-of-bounds M2 edit")
            edits.append((start, end, [] if fields[2] == "-NONE-" else fields[2].split()))
        if set(annotators) != {"0", "1"}:
            raise ValueError("Expected both CWEB annotators, including noops")
        references = []
        for key in ("0", "1"):
            corrected = tokens[:]
            edits = sorted(annotators[key], key=lambda edit: (edit[0], edit[1]))
            for previous, current in zip(edits, edits[1:]):
                if previous[1] > current[0]:
                    raise ValueError("Overlapping M2 edits")
            for start, end, replacement in reversed(edits):
                corrected[start:end] = replacement
            references.append(" ".join(corrected))
        if not source.strip() or any(not ref.strip() for ref in references):
            raise ValueError("Empty benchmark sentence")
        result.append({"source": source, "references": references})
    return result


def prepare(output):
    if output.exists():
        raise ValueError("Output exists; choose a fresh directory")
    output.resolve().relative_to(ROOT / "data/imported")
    output.mkdir(parents=True)
    materialize(output / "jfleg")
    scorer = download(f"https://raw.githubusercontent.com/keisks/jfleg/{JFLEG_REVISION}/eval/gleu.py")
    if digest(scorer) != GLEU_SHA256:
        raise ValueError("Official GLEU scorer hash mismatch")
    (output / "gleu.py").write_bytes(scorer)
    original = output / "cweb/original"
    original.mkdir(parents=True)
    files, rows, gold_blocks = {}, [], []
    for section, expected in (("G", 3981), ("S", 2864)):
        prefix = f"CWEB-{section}.test"
        gold_name = f"data/m2/{prefix}.m2"
        source_name = f"data/tokenized/{prefix}.tok.source"
        downloads = {}
        for name in (gold_name, source_name):
            raw = download(f"https://raw.githubusercontent.com/SimonHFL/CWEB/{CWEB_REVISION}/{name}")
            (original / Path(name).name).write_bytes(raw)
            files[name] = {"sha256": digest(raw), "bytes": len(raw)}
            downloads[name] = raw
        section_rows = m2_rows(downloads[gold_name])
        sources = downloads[source_name].decode("utf-8").splitlines()
        if len(section_rows) != expected or sources != [row["source"] for row in section_rows]:
            raise ValueError("CWEB gold/source population mismatch; no filtering allowed")
        for index, row in enumerate(section_rows):
            rows.append({"id": f"cweb-{section.lower()}-test-{index}", **row,
                         "category": f"CWEB-{section}", "origin": "CWEB", "evaluation_only": True,
                         "license": "CC-BY-NC-SA-4.0", "source_revision": CWEB_REVISION})
        gold_blocks.append(downloads[gold_name].decode("utf-8").strip())
    (output / "cweb/test.m2").write_text("\n\n".join(gold_blocks) + "\n\n", encoding="utf-8")
    (output / "cweb/test.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"dataset": "CWEB", "revision": CWEB_REVISION, "license": "CC-BY-NC-SA-4.0",
                "source_url": "https://github.com/SimonHFL/CWEB", "evaluation_only": True,
                "attribution": "Flachs et al. (2020), Grammatical Error Correction in Low Error Density Domains: A New Benchmark and Analyses.",
                "files": files, "sentences": len(rows), "reference_count": 2,
                "test_jsonl_sha256": digest((output / "cweb/test.jsonl").read_bytes()),
                "test_m2_sha256": digest((output / "cweb/test.m2").read_bytes())}
    (output / "cweb/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "JFLEG, its upstream GLEU scorer, and CWEB remain separately licensed CC-BY-NC-SA-4.0.\n"
        "Noncommercial research evaluation only; never bundle these files with the app or training.\n"
        "Sources: https://github.com/keisks/jfleg and https://github.com/SimonHFL/CWEB\n", encoding="utf-8")
    return {"prepared": True, "jfleg_test_sentences": 747, "cweb_test_sentences": len(rows)}


def validate_report(input_path, report_path):
    raw = input_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    report = json.loads(report_path.read_bytes())
    if (report.get("passed") is not True or report.get("inputSha256") != digest(raw)
            or report.get("network", {}).get("blockedRequests") != 0):
        raise ValueError("Execution failed or input/network evidence mismatched")
    if not report.get("code") or not report.get("engineBundleSha256"):
        raise ValueError("Missing executed-code provenance")
    if report["model"]["policy"].get("publicationAllowed") is not True:
        raise ValueError("Nonpublishable candidate model")
    if len(rows) != report["input"]["totalRows"] or len(rows) != report["input"]["selectedRows"]:
        raise ValueError("Partial populations cannot be published as full benchmarks")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate input IDs")
    outputs = {"unchanged": [row["source"] for row in rows]}
    for run in report["runs"]:
        mode = run["mode"]
        if mode not in MODES or mode in outputs or run["maxPasses"] != 1:
            raise ValueError("Expected exactly one shipped single-pass condition per mode")
        predictions = run["predictions"]
        if len(predictions) != len(rows):
            raise ValueError("Missing benchmark predictions")
        for index, (row, pred) in enumerate(zip(rows, predictions)):
            if (pred.get("id") != row["id"] or pred.get("index") != index or pred.get("source") != row["source"]
                    or pred.get("sourceSha256") != digest(row["source"].encode("utf-8"))
                    or pred.get("references") != row["references"]):
                raise ValueError("Prediction source/reference/order mismatch")
            if pred.get("failed") is not False or pred.get("failure") is not None or pred.get("executed") is not True:
                raise ValueError("Failed/skipped predictions cannot qualify a benchmark")
            if not isinstance(pred.get("corrected"), str) or not pred["corrected"].strip() or "\n" in pred["corrected"] or "\r" in pred["corrected"]:
                raise ValueError("Predictions must retain one nonempty sentence per line")
        outputs[mode] = [pred["corrected"] for pred in predictions]
    if set(outputs) != {"unchanged", *MODES}:
        raise ValueError("Missing benchmark modes")
    return rows, outputs, report


def clean_counts(rows, predictions):
    accepted = [i for i, row in enumerate(rows) if any(normalized(row["source"]) == normalized(ref) for ref in row["references"])]
    changed = sum(normalized(rows[i]["source"]) != normalized(predictions[i]) for i in accepted)
    return {"reference_accepted_sources": len(accepted), "reference_accepted_sources_changed": changed,
            "reference_accepted_change_rate": changed / len(accepted) if accepted else None,
            "all_sources_changed": sum(normalized(row["source"]) != normalized(pred) for row, pred in zip(rows, predictions))}


def parse_errant(stdout):
    match = re.search(r"^([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", stdout, re.MULTILINE)
    if not match:
        raise ValueError("Unexpected official ERRANT result format")
    tp, fp, fn = map(int, match.group(1, 2, 3))
    # Recompute full precision from official counts; CLI rounds to four decimals.
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": precision, "recall": recall,
            "f0_5": 1.25 * tp / (1.25 * tp + fp + .25 * fn) if tp + fp + fn else 1.0,
            "true_positive_edits": tp, "false_positive_edits": fp, "missed_edits": fn}


def run_tool(args):
    completed = subprocess.run(args, check=True, capture_output=True, text=True, timeout=600,
                               env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return completed.stdout


def score(evaluation_dir, jfleg_report, cweb_report):
    import errant
    from errant.commands.parallel_to_m2 import noop_edit
    from evaluate import score_browser_report

    scorer = evaluation_dir / "gleu.py"
    if digest(scorer.read_bytes()) != GLEU_SHA256:
        raise ValueError("Official GLEU scorer hash mismatch")
    jfleg_manifest = json.loads((evaluation_dir / "jfleg/manifest.json").read_bytes())
    cweb_manifest = json.loads((evaluation_dir / "cweb/manifest.json").read_bytes())
    if jfleg_manifest["revision"] != JFLEG_REVISION or cweb_manifest["revision"] != CWEB_REVISION:
        raise ValueError("Benchmark revision mismatch")
    cweb_gold = evaluation_dir / "cweb/test.m2"
    if digest(cweb_gold.read_bytes()) != cweb_manifest["test_m2_sha256"]:
        raise ValueError("CWEB original gold hash mismatch")
    result = {"schema": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Frozen shipped policy; full natural test populations; one browser correction pass; no training or threshold selection.",
              "data_use": "Noncommercial research evaluation. Separately licensed CC-BY-NC-SA-4.0 inputs, gold and predictions remain local; only aggregates are published.",
              "scorer_versions": {name: version(name) for name in ("errant", "spacy", "en-core-web-sm", "rapidfuzz", "numpy", "scipy")},
              "scoring_script_sha256": digest(Path(__file__).read_bytes()), "datasets": {}}
    annotator = errant.load("en")
    with tempfile.TemporaryDirectory(prefix="gamma-public-scoring-") as temporary:
        work = Path(temporary)
        for dataset, report_path, expected in (("jfleg", jfleg_report, 747), ("cweb", cweb_report, 6845)):
            input_path = evaluation_dir / dataset / "test.jsonl"
            rows, outputs, browser = validate_report(input_path, report_path)
            if len(rows) != expected:
                raise ValueError("Incorrect complete benchmark population")
            manifest = jfleg_manifest if dataset == "jfleg" else cweb_manifest
            expected_hash = manifest["splits"]["test"]["sha256"] if dataset == "jfleg" else manifest["test_jsonl_sha256"]
            if digest(input_path.read_bytes()) != expected_hash:
                raise ValueError("Evaluation file differs from pinned manifest")
            scores = {}
            for mode, predictions in outputs.items():
                hypothesis = work / f"{dataset}-{mode}.txt"
                hypothesis.write_text("\n".join(predictions) + "\n", encoding="utf-8")
                if dataset == "jfleg":
                    original = evaluation_dir / "jfleg/original"
                    for suffix in ("src", "ref0", "ref1", "ref2", "ref3"):
                        if digest((original / f"test.{suffix}").read_bytes()) != jfleg_manifest["files"][f"test/test.{suffix}"]["sha256"]:
                            raise ValueError("JFLEG original file hash mismatch")
                    stdout = run_tool([sys.executable, str(scorer), "-s", str(original / "test.src"),
                                       "-r", *[str(original / f"test.ref{i}") for i in range(4)], "--hyp", str(hypothesis)])
                    values = ast.literal_eval(stdout.splitlines()[-1])[0]
                    scores[mode] = {"gleu": float(values[0]), "random_reference_std": float(values[1]),
                                    "random_reference_interval_95": values[2], **clean_counts(rows, predictions)}
                else:
                    hyp_m2 = work / f"cweb-{mode}.m2"
                    with hyp_m2.open("w", encoding="utf-8") as stream:
                        for row, prediction in zip(rows, predictions):
                            source = row["source"]
                            stream.write("S " + source + "\n")
                            if source == prediction:
                                stream.write(noop_edit(0) + "\n")
                            else:
                                # Same defaults as errant_parallel, without retokenizing original offsets.
                                edits = annotator.annotate(annotator.parse(source, tokenise=False), annotator.parse(prediction, tokenise=False))
                                for edit in edits:
                                    stream.write(edit.to_m2(0) + "\n")
                            stream.write("\n")
                    stdout = run_tool([str(Path(sys.executable).parent / "errant_compare"), "-hyp", str(hyp_m2), "-ref", str(cweb_gold)])
                    scores[mode] = {**parse_errant(stdout), **clean_counts(rows, predictions)}
                print(json.dumps({"dataset": dataset, "mode": mode, "scored": len(rows)}), flush=True)
            executed = {run["mode"]: run["summary"] for run in browser["runs"]}
            result["datasets"][dataset] = {"sentences": len(rows), "references_per_sentence": 4 if dataset == "jfleg" else 2,
                "source_url": "https://github.com/keisks/jfleg" if dataset == "jfleg" else "https://github.com/SimonHFL/CWEB",
                "revision": manifest["revision"], "license": manifest["license"], "attribution": manifest["attribution"],
                "dataset_manifest_sha256": digest((evaluation_dir / dataset / "manifest.json").read_bytes()),
                "population_sha256": digest(input_path.read_bytes()), "browser_report_sha256": digest(report_path.read_bytes()),
                "engine_bundle_sha256": browser["engineBundleSha256"], "code": browser["code"], "model": browser["model"],
                "browser": browser["browser"], "network": browser["network"], "execution": executed, "scores": scores,
                "metric": "Official upstream corpus GLEU, 4-grams, 500 seeded reference selections; interval reflects reference sampling, not population uncertainty." if dataset == "jfleg" else "Standard ERRANT 3.0.2 span correction against original multiannotator M2; default corpus reference selection, tokenized inputs."}
        # Preserve the earlier metric for diagnosis; explicitly separate from standard GLEU/ERRANT.
        custom = score_browser_report(jfleg_report, evaluation_dir / "jfleg", "test", "errant")
        result["jfleg_custom_edit_diagnostics"] = {"scope": custom["scope"], "conditions": custom["conditions"]}
    for field in ("code", "model", "engine_bundle_sha256"):
        if result["datasets"]["jfleg"][field] != result["datasets"]["cweb"][field]:
            raise ValueError("Benchmark datasets executed different code or model policies")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=Path("data/imported/public-benchmark"))
    prepare_parser.add_argument("--execute", action="store_true")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--evaluation-dir", type=Path, required=True)
    score_parser.add_argument("--jfleg-report", type=Path, required=True)
    score_parser.add_argument("--cweb-report", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output) if args.execute else {"mode": "dry-run", "jfleg": 747, "cweb": 6845, "license": "CC-BY-NC-SA-4.0"}))
    else:
        if args.output.exists():
            raise ValueError("Report exists; choose a fresh filename")
        aggregate = score(args.evaluation_dir, args.jfleg_report, args.cweb_report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"published_aggregate": str(args.output), "datasets": {name: value["sentences"] for name, value in aggregate["datasets"].items()}}))
