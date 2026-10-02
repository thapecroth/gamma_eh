"""Score complete actual-browser outputs with pinned ERRANT, not official JFLEG GLEU."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import unicodedata


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.
    r = tp / (tp + fn) if tp + fn else 1.
    return {"edit_precision": p, "edit_recall": r, "edit_f0_5": 1.25*p*r/max(1e-12, .25*p+r),
            "true_positive_edits": tp, "false_positive_edits": fp, "missed_edits": fn,
            "predicted_edits": tp+fp}


def score(rows, predictions, edits):
    if len(rows) != len(predictions):
        raise ValueError("Prediction population mismatch")
    counts = Counter()
    for row, prediction in zip(rows, predictions):
        predicted = edits(row["source"], prediction)
        alternatives = []
        for reference in row["references"]:
            gold = edits(row["source"], reference)
            tp, fp, fn = len(predicted & gold), len(predicted-gold), len(gold-predicted)
            alternatives.append((metrics(tp, fp, fn)["edit_f0_5"], tp, -fp, -fn))
        _, tp, nfp, nfn = max(alternatives)
        counts.update({"tp": tp, "fp": -nfp, "fn": -nfn})
        clean = any(normalized(row["source"]) == normalized(ref) for ref in row["references"])
        counts["clean"] += clean
        counts["changed"] += clean and normalized(prediction) != normalized(row["source"])
    return {**metrics(counts["tp"], counts["fp"], counts["fn"]), "sentences": len(rows),
            "clean_sentences": counts["clean"], "clean_sentences_changed": counts["changed"],
            "clean_sentence_false_positive_rate": counts["changed"] / counts["clean"] if counts["clean"] else None}


def run(input_path, report_path):
    raw = input_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if not report["passed"] or report["input_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Execution failed or population hash mismatch")
    if len(report["rows"]) != len(rows):
        raise ValueError("Incomplete evaluation population")
    for row, output in zip(rows, report["rows"]):
        if row["id"] != output["id"] or row["source"] != output["source"]:
            raise ValueError("Evaluation source/order mismatch")
        if not row["references"] or not all(isinstance(ref, str) and ref.strip() for ref in row["references"]):
            raise ValueError("Invalid evaluation references")
    import errant
    annotator = errant.load("en")
    @lru_cache(maxsize=20000)
    def edits(source, target):
        if source == target:
            return frozenset()
        return frozenset((edit.o_start, edit.o_end, tuple(tok.text for tok in edit.c_toks))
                         for edit in annotator.annotate(annotator.parse(source, tokenise=True), annotator.parse(target, tokenise=True)))
    results = {"rules": score(rows, [row["rules"] for row in report["rows"]], edits)}
    for mode in ["model_only", "full_engine"]:
        results[mode] = []
        for threshold in report["thresholds"]:
            predictions = [next(p[mode] for p in row["predictions"] if p["threshold"] == threshold) for row in report["rows"]]
            results[mode].append({"threshold": threshold, **score(rows, predictions, edits)})
    return {"passed": True, "scope": "Complete-population JAX/WASM; actual ERRANT with best sentence F0.5 reference; not official JFLEG GLEU.",
            "input_sha256": report["input_sha256"], "browser_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "model": report["model"], "engine_source_sha256": report["engine_source_sha256"],
            "scorer_versions": {name: version(name) for name in ["errant", "spacy", "en-core-web-sm", "rapidfuzz"]}, "scores": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.input, args.report)
    except Exception as error:
        args.output.write_text(json.dumps({"passed": False, "failure": str(error)}) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["scores"], indent=2))
