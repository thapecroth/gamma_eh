"""Complete-population decoded evaluation; default scoring is approximate, not ERRANT."""
import argparse
from collections import Counter
import difflib
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import time

from edit_ops import TOKEN_RE, decode_word, preserve_case, render, tag_category
from pairs import hash_file, normalized
from legacy_spelling import known_word, spelling_distance, whole_words

# Mirror the two independently matched browser patterns. JS /u word boundaries
# are ASCII, while letters/numbers and ECMAScript whitespace remain Unicode.
JS_WHITESPACE = r"\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
PROTECTED_PATTERNS = [
    re.compile(rf"(`+)[\s\S]*?(?:(?<!`)\1(?!`)|\Z)|(?:https?://|www\.)[^{JS_WHITESPACE}]+|(?a:\b)[^{JS_WHITESPACE}@]+@[^{JS_WHITESPACE}@]+\.[^{JS_WHITESPACE}@]+"),
    re.compile(r"(?:\.\.?[/\\]|[/\\])[\w./\\-]+|(?a:\b)[\w-]+(?:[./\\][\w-]+)+|[#@]\w+"),
]


def protected_spans(text):
    return [(match.start(), match.end()) for pattern in PROTECTED_PATTERNS for match in pattern.finditer(text)]
ARTICLE_HEADS = {
    **dict.fromkeys("apple orange egg umbrella envelope idea hour honest honor heir".split(), "an"),
    **dict.fromkeys("book pencil bicycle camera notebook ticket friend message pen project library university user unicorn european one".split(), "a"),
}
SPELLING = {"recieve": "receive", "recieved": "received", "recieving": "receiving", "beleive": "believe",
            "beleived": "believed", "freinds": "friends", "seperate": "separate", "neccessary": "necessary",
            "definately": "definitely", "tommorow": "tomorrow", "becuase": "because", "freind": "friend",
            "writting": "writing", "grammer": "grammar", "thier": "their", "adress": "address",
            "avaliable": "available", "realy": "really", "succesful": "successful", "diffrent": "different",
            "peopel": "people", "buisness": "business", "importent": "important", "teh": "the"}
FUNCTION_WORDS = set("a an the to of in on at for from with is are am was were be been being has have do does did".split())
DELETION_THRESHOLD = .98
WORD = re.compile(r"^[A-Za-z]+(?:['’][A-Za-z]+)*$")
PUNCTUATION = re.compile(r"^[.,!?;:()[\]{}'\"-]$")


def load_evaluation(directory, split):
    path = directory / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows: raise ValueError(f"Empty {split} evaluation population")
    for row in rows:
        refs = row.get("references", [row.get("target")])
        if not isinstance(row.get("source"), str) or not isinstance(refs, list) or not refs or not all(isinstance(r, str) for r in refs):
            raise ValueError("Invalid evaluation row; populations must not be silently filtered")
        row["references"] = refs
    return rows


def approximate_edits(source, target):
    a, b = TOKEN_RE.findall(source), TOKEN_RE.findall(target)
    return {(i, j, tuple(b[k:l])) for kind, i, j, k, l in
            difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if kind != "equal"}


@lru_cache(maxsize=2)
def make_scorer(name):
    if name == "approximate":
        @lru_cache(maxsize=20000)
        def approximate(source, target):
            return {edit: "unclassified" for edit in approximate_edits(source, target)}
        return approximate
    if name != "errant": raise ValueError("Unknown scorer")
    try:
        import errant
        annotator = errant.load("en")
    except (ImportError, OSError):
        raise ValueError("Actual ERRANT requires the errant package and its English spaCy model; use --scorer approximate or install separately") from None
    @lru_cache(maxsize=20000)
    def edits(source, target):
        return {(edit.o_start, edit.o_end, tuple(token.text for token in edit.c_toks)): edit.type
                for edit in annotator.annotate(annotator.parse(source, tokenise=True), annotator.parse(target, tokenise=True))}
    edits.metadata = {package: version(package) for package in ["errant", "spacy", "en-core-web-sm", "rapidfuzz"]}
    return edits


def edit_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 1.
    recall = tp / (tp + fn) if tp + fn else 1.
    return {"edit_precision": precision, "edit_recall": recall,
            "edit_f0_5": 1.25 * precision * recall / max(1e-12, .25 * precision + recall),
            "true_positive_edits": tp, "false_positive_edits": fp, "missed_edits": fn,
            "predicted_edits": tp + fp, "reference_edits": tp + fn}


def score_predictions(rows, predictions, scorer="approximate", failures=0):
    if len(rows) != len(predictions): raise ValueError("Prediction population mismatch")
    edits = make_scorer(scorer)
    counts = Counter()
    categories = {}
    error_types = {}
    for row, prediction in zip(rows, predictions):
        source, references = row["source"], row["references"]
        if not references or not all(isinstance(reference, str) and reference.strip() for reference in references):
            raise ValueError("Evaluation references must be nonempty strings")
        if not isinstance(prediction, str): raise ValueError("Predictions must be strings")
        predicted_types = edits(source, prediction)
        predicted = set(predicted_types)
        alternatives = []
        for reference in references:
            gold_types = edits(source, reference)
            gold = set(gold_types)
            tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
            alternatives.append((edit_metrics(tp, fp, fn)["edit_f0_5"], tp, -fp, -fn, gold_types))
        _, tp, negative_fp, negative_fn, gold_types = max(alternatives, key=lambda value: value[:4])
        fp, fn = -negative_fp, -negative_fn
        for span, category in gold_types.items():
            error_types.setdefault(category, Counter())["tp" if span in predicted else "fn"] += 1
        for span in predicted - set(gold_types):
            error_types.setdefault(predicted_types[span], Counter())["fp"] += 1
        counts.update({"tp": tp, "fp": fp, "fn": fn})
        clean = any(normalized(source) == normalized(reference) for reference in references)
        counts["clean"] += clean
        counts["clean_changed"] += clean and normalized(source) != normalized(prediction)
        counts["exact"] += any(normalized(prediction) == normalized(reference) for reference in references)
        category = categories.setdefault(row.get("category", "mixed"), Counter())
        category.update({"sentences": 1, "tp": tp, "fp": fp, "fn": fn})
    result = {**edit_metrics(counts["tp"], counts["fp"], counts["fn"]), "sentences": len(rows),
              "sentence_exact_match": counts["exact"] / max(1, len(rows)),
              "clean_sentences": counts["clean"], "clean_sentences_changed": counts["clean_changed"],
              "clean_sentence_false_positive_rate": counts["clean_changed"] / counts["clean"] if counts["clean"] else None,
              "inference_failures": failures, "scorer": scorer,
              "scoring_note": "Approximate exact token-span edits using difflib; not ERRANT/M2/GLEU." if scorer == "approximate" else "Actual ERRANT token-span edits; best reference selected by sentence F0.5.",
              "reference_policy": "Highest sentence edit F0.5, then TP, then fewer FP/FN; all references retained.",
              "scorer_versions": getattr(edits, "metadata", {}),
              "by_error_type": {name: edit_metrics(c["tp"], c["fp"], c["fn"]) for name, c in sorted(error_types.items())},
              "by_category": {name: {**edit_metrics(c["tp"], c["fp"], c["fn"]), "sentences": c["sentences"]}
                              for name, c in categories.items()}}
    return result


def score_browser_report(path, directory, split, scorer="approximate"):
    """Join actual browser outputs to a frozen, complete evaluation population."""
    rows = load_evaluation(directory, split)
    report = json.loads(path.read_text())
    population_hash = hash_file(directory / f"{split}.jsonl")
    if report.get("inputSha256") != population_hash:
        raise ValueError("Browser input hash does not match the evaluation population")
    results = []
    for run in report.get("runs", []):
        predictions = run["predictions"]
        if len(predictions) != len(rows): raise ValueError("Browser output must retain the complete population")
        for index, (row, prediction) in enumerate(zip(rows, predictions)):
            source_hash = hashlib.sha256(row["source"].encode()).hexdigest()
            if (prediction.get("sourceSha256") != source_hash or prediction.get("index") != index
                    or prediction.get("source") != row["source"]):
                raise ValueError("Browser prediction order/source mismatch")
        score = score_predictions(rows, [p["corrected"] for p in predictions], scorer,
                                  sum(bool(p.get("failed")) for p in predictions))
        results.append({"mode": run["mode"], "max_passes": run["maxPasses"], **score,
                        "browser": run.get("summary", {})})
    if not results: raise ValueError("No browser conditions to score")
    return {"population_sha256": population_hash, "browser_report_sha256": hash_file(path),
            "scope": "Complete-population actual browser predictions; custom best-sentence-reference ERRANT scoring is not official JFLEG GLEU.",
            "model": report.get("model"), "browser": report.get("browser"),
            "network": report.get("network"), "passed_execution": report.get("passed"), "conditions": results}


def qualified(candidate, min_precision, max_clean_fp, min_support=25):
    return (candidate["predicted_edits"] > 0 and candidate["predicted_edits"] >= min_support
            and candidate.get("inference_failures", 0) == 0
            and candidate["edit_precision"] >= min_precision
            and candidate["clean_sentence_false_positive_rate"] is not None
            and candidate["clean_sentence_false_positive_rate"] <= max_clean_fp)


def choose_calibration(candidates, no_edit, min_precision=.95, max_clean_fp=.02, min_support=25):
    if not candidates: raise ValueError("Calibration requires development candidates")
    if min_support < 1: raise ValueError("Development edit support must be positive")
    eligible = [candidate for candidate in candidates if qualified(candidate, min_precision, max_clean_fp, min_support)]
    chosen = max(eligible, key=lambda r: (r["edit_f0_5"], r["threshold"])) if eligible else no_edit
    return {"split": "dev", "candidates": candidates, "selected": chosen,
            "best_unconstrained": max(candidates, key=lambda r: (r["edit_f0_5"], r["threshold"])),
            "constraints": {"min_edit_precision": min_precision, "max_clean_sentence_false_positive_rate": max_clean_fp,
                            "min_development_predicted_edits": min_support},
            "constraints_met": bool(eligible), "disable_model_edits": not bool(eligible),
            "fallback_reason": None if eligible else "No threshold met development precision, clean-text, and edit-support constraints; suppress model edits."}


def choose_joint_calibration(int8, fp32, no_edit, min_precision=.95, max_clean_fp=.02, min_support=25):
    """A shared browser policy must qualify both exported precision formats on dev."""
    report = choose_calibration(int8, no_edit, min_precision, max_clean_fp, min_support)
    other = {candidate["threshold"]: candidate for candidate in fp32}
    if len(other) != len(fp32) or set(other) != {candidate["threshold"] for candidate in int8}:
        raise ValueError("Export calibration grids differ")
    eligible = [candidate for candidate in int8 if qualified(candidate, min_precision, max_clean_fp, min_support)
                and qualified(other[candidate["threshold"]], min_precision, max_clean_fp, min_support)]
    chosen = max(eligible, key=lambda r: (min(r["edit_f0_5"], other[r["threshold"]]["edit_f0_5"]), r["threshold"])) if eligible else no_edit
    report.update({"selected": chosen, "constraints_met": bool(eligible), "disable_model_edits": not bool(eligible),
                   "fallback_reason": None if eligible else "No shared threshold met precision, clean-text, and edit-support constraints in both INT8 and FP32 development outputs.",
                   "fp32_candidates": fp32, "backend": "Shared INT8/FP32 ONNX CPU development policy"})
    return report


def valid_article(article, next_word):
    word = next_word.lower()
    return ARTICLE_HEADS.get(SPELLING.get(word, word)) == article


def valid_payload_articles(tokens, next_word):
    return all(token.lower() not in {"a", "an"} or valid_article(
        token.lower(), tokens[index + 1] if index + 1 < len(tokens) else next_word)
        for index, token in enumerate(tokens))


VERB_FAMILIES = [("is", "are", "am", False), ("was", "were", "was", False),
                 ("has", "have", None, False)] + [
    (singular, plural, None, True) for plural, singular in [
        ("work", "works"), ("walk", "walks"), ("read", "reads"), ("write", "writes"),
        ("play", "plays"), ("learn", "learns"), ("travel", "travels"), ("cook", "cooks"),
        ("wait", "waits"), ("talk", "talks"), ("sleep", "sleeps"), ("run", "runs"),
        ("study", "studies"), ("watch", "watches"), ("go", "goes")]]
AGREEMENT_VERBS = {verb: family for family in VERB_FAMILIES for verb in family[:3] if verb}
SINGULAR_HEADS = set("friend teacher neighbor student child colleague manager cat dog news".split())
PLURAL_HEADS = set("friends teachers neighbors students children colleagues managers cats dogs people".split())


def expected_verb(text, word, family):
    subject = re.search(rf"(?:^|[.!?][{JS_WHITESPACE}]+)(?:(i|you|we|they|he|she|it)|(?:my|your|our|his|her|their|the|a|an)[ \t]+([A-Za-z]+))[ \t]+\Z",
                        text[:word["start"]], re.IGNORECASE | re.ASCII)
    if not subject: return None
    tail = text[word["end"]:]
    if re.search(r"^[^.!?]*\?", tail): return None
    pronoun = (subject[1] or "").lower()
    head = (subject[2] or "").lower()
    head = SPELLING.get(head, head)
    if pronoun: singular = pronoun in {"he", "she", "it"}
    elif head in SINGULAR_HEADS: singular = True
    elif head in PLURAL_HEADS: singular = False
    else: return None
    singular_verb, plural_verb, first_person, habitual = family
    if habitual and (plural_verb == "read" or not re.search(
            r"^[^.!?\n]*\bevery[ \t]+(?:morning|day|evening|night|week)\b[^.!?\n]*(?:\.|$)",
            tail, re.IGNORECASE | re.ASCII)): return None
    return first_person if pronoun == "i" and first_person else singular_verb if singular else plural_verb


def valid_verb(text, word, proposed):
    """Mirror the browser's bounded full-source agreement guard, including abstention."""
    original_family = AGREEMENT_VERBS.get(word["text"].lower())
    proposed_family = AGREEMENT_VERBS.get(proposed.lower())
    if not original_family and not proposed_family: return True
    if not original_family or original_family != proposed_family: return False
    expected = expected_verb(text, word, original_family)
    return word["text"].lower() != expected and proposed.lower() == expected


COUNTABLE_OBJECTS = set("apple orange egg umbrella envelope idea hour heir book pencil bicycle camera notebook ticket friend message pen project library university user unicorn".split())


def valid_legacy_replacement(text, word, proposed, next_word):
    original, replacement = word["text"].lower(), proposed.lower()
    if original in AGREEMENT_VERBS or replacement in AGREEMENT_VERBS:
        return valid_verb(text, word, replacement)
    if original in {"a", "an"} or replacement in {"a", "an"}:
        return original in {"a", "an"} and original != replacement and valid_article(replacement, next_word)
    if not re.fullmatch(r"[A-Z]?[a-z]+", word["text"]): return False
    prefix = re.sub(rf"[{JS_WHITESPACE}]+\Z", "", text[:word["start"]])
    if word["text"][0].isupper() and not re.search(rf"(?:^|[.!?][{JS_WHITESPACE}]*)\Z", prefix):
        return False
    canonical = SPELLING.get(original, original)
    if canonical != original: return canonical == replacement
    if not re.fullmatch(r"[a-z]{3,32}", original) or not re.fullmatch(r"[a-z]{3,32}", replacement) or known_word(original) or not known_word(replacement):
        return False
    limit = 1 if len(original) < 5 else 2
    return spelling_distance(original, replacement, limit) <= limit


def valid_legacy_append(text, word, proposed, next_word):
    family = AGREEMENT_VERBS.get(word["text"].lower())
    return bool(family and next_word and family[0] == "has" and word["text"].lower() == expected_verb(text, word, family)
        and next_word["text"] == next_word["text"].lower() and re.fullmatch(r"[ \t]+", text[word["end"]:next_word["start"]])
        and SPELLING.get(next_word["text"], next_word["text"]) in COUNTABLE_OBJECTS and valid_article(proposed, next_word["text"])
        and re.match(rf"^[ \t]*(?:[.!?](?:[{JS_WHITESPACE}]|$)|$)", text[next_word["end"]:]))


def spaced_append(payload, source_after):
    # Insert before original whitespace; punctuation attaches to the anchor.
    prefix = "" if payload[:1] in ".,!?;:)]}" else " "
    suffix = " " if source_after and not source_after[0].isspace() and source_after[0] not in ".,!?;:)]}" else ""
    return prefix + payload + suffix


def decode_proposal(text, words, index, tag, confidence, schema):
    word = words[index]
    start, end, value = word["start"], word["end"], word["text"]
    if not WORD.fullmatch(value) and not (schema == 2 and PUNCTUATION.fullmatch(value)): return None
    if tag == "KEEP": return None
    if schema == 1:
        whole = whole_words(text)
        if (start, end) not in whole: return None
        next_word = words[index + 1] if index + 1 < len(words) else None
        if next_word and (next_word["start"], next_word["end"]) not in whole: next_word = None
        if tag == "DELETE":
            previous = words[index - 1] if index > 0 else None
            if value.lower() in {"had", "that"} or not previous or previous["text"].lower() != value.lower() or (previous["start"], previous["end"]) not in whole or not re.fullmatch(r"[ \t]+", text[previous["end"]:start]):
                return None
            while start > 0 and text[start - 1] in " \t": start -= 1
            return {"start": start, "end": end, "replacement": "", "confidence": confidence, "tag": tag, "category": tag_category(tag)}
        if tag.startswith("REPLACE:") and not valid_legacy_replacement(text, word, tag[8:], next_word["text"] if next_word else ""):
            return None
        if tag.startswith("APPEND:") and not valid_legacy_append(text, word, tag[7:], next_word):
            return None
    if tag == "DELETE":
        lower = value.lower()
        duplicate = bool(WORD.fullmatch(value)) and lower not in {"had", "that"} and any(
            0 <= adjacent < len(words) and words[adjacent]["text"].lower() == lower
            for adjacent in [index - 1, index + 1])
        punctuation = bool(PUNCTUATION.fullmatch(value))
        if not duplicate and not (schema == 2 and confidence >= DELETION_THRESHOLD
                                  and (punctuation or lower in FUNCTION_WORDS)):
            return None
        if punctuation:
            # Preserve existing spaces; comma deletion must not join words.
            replacement = " " if start > 0 and end < len(text) and text[start - 1].isalnum() and text[end].isalnum() else ""
        else:
            if end < len(text) and text[end] in " \t":
                while end < len(text) and text[end] in " \t": end += 1
            else:
                while start > 0 and text[start - 1] in " \t": start -= 1
            replacement = ""
    elif tag.startswith("REPLACE:"):
        proposed = tag[8:]
        next_word = words[index + 1]["text"] if index + 1 < len(words) else ""
        if proposed in {"a", "an"} and not valid_article(proposed, next_word): return None
        replacement = preserve_case(value, proposed)
        if value.lower() != replacement.lower() and not valid_verb(text, word, replacement): return None
    elif tag.startswith("APPEND:"):
        proposed = tag[7:]
        next_word = words[index + 1]["text"] if index + 1 < len(words) else ""
        if proposed in {"a", "an"} and not valid_article(proposed, next_word): return None
        start = end
        replacement = " " + proposed
    elif schema == 2:
        try:
            decoded = decode_word(value, tag, schema)
        except ValueError:
            return None
        if tag.startswith("APPEND_EXACT:"):
            payload = tag[len("APPEND_EXACT:"):]
            next_word = words[index + 1]["text"] if index + 1 < len(words) else ""
            if not valid_payload_articles(decoded[1:], next_word): return None
            start = end
            replacement = spaced_append(payload, text[end:])
        elif tag.startswith("PREPEND_EXACT:"):
            end = start
            payload = tag[len("PREPEND_EXACT:"):]
            if not valid_payload_articles(decoded[:-1], value): return None
            replacement = payload + ("" if payload.endswith(("(", "[", "{")) else " ")
        else:
            replacement = render(decoded, 2)
            if value.lower() != replacement.lower() and not valid_verb(text, word, replacement): return None
            next_word = words[index + 1]["text"] if index + 1 < len(words) else ""
            if not valid_payload_articles(decoded, next_word): return None
    else:
        return None
    if (schema == 2 and replacement and PUNCTUATION.fullmatch(value) and tag != "DELETE"
            and not tag.startswith(("APPEND", "PREPEND"))):
        if replacement[0].isalnum() and start > 0 and text[start - 1].isalnum(): replacement = " " + replacement
        if replacement[-1].isalnum() and end < len(text) and text[end].isalnum(): replacement += " "
    if text[start:end] == replacement: return None
    return {"start": start, "end": end, "replacement": replacement,
            "confidence": confidence, "tag": tag, "category": tag_category(tag)}


def apply_proposals(text, proposals, threshold, disabled=False, category_thresholds=None):
    if disabled: return text
    chosen = sorted((proposal for proposal in proposals if proposal["confidence"] >=
                     max(threshold, (category_thresholds or {}).get(proposal["category"], threshold))),
                    key=lambda p: (p["start"], p["end"]))
    accepted = []
    for proposal in chosen:
        if accepted and (proposal["start"] < accepted[-1]["end"] or proposal["start"] == accepted[-1]["start"]):
            continue
        accepted.append(proposal)
    for proposal in reversed(accepted):
        text = text[:proposal["start"]] + proposal["replacement"] + text[proposal["end"]:]
    return text


def encoded_chunks(text, tokenizer, max_length, suppressed=None):
    if not 4 <= max_length <= 512: raise ValueError("Invalid context length")
    chunks, current, size = [], [], 2
    for match in TOKEN_RE.finditer(text):
        value = match.group()
        pieces = tokenizer.encode(value, add_special_tokens=False)
        if len(pieces) > max_length - 2:
            if current: chunks.append(current)
            current, size = [], 2
            if suppressed is not None: suppressed["over_budget_words"] += 1
            continue
        if size + len(pieces) > max_length:
            if current: chunks.append(current)
            current, size = [], 2
        current.append({"text": value, "start": match.start(), "end": match.end()})
        size += len(pieces)
        if value in {".", "!", "?"}:
            chunks.append(current); current, size = [], 2
    if current: chunks.append(current)
    return chunks


def collect_proposals(rows, tokenizer, infer, labels, max_length, schema=1, batch_size=16):
    """Keep only decoded proposals; no population-sized logit tensor is retained."""
    import numpy as np
    if batch_size < 1 or not labels or labels[0] != "KEEP": raise ValueError("Invalid inference batch size or label vocabulary")
    records = [{"proposals": [], "failed": False} for _ in rows]
    pending = []
    suppressed = Counter()
    started = time.monotonic()
    def flush():
        if not pending: return
        try:
            encoded = tokenizer([[word["text"] for word in words] for _, words in pending],
                                is_split_into_words=True, padding=True, return_tensors="np")
            logits = infer({key: encoded[key].astype(np.int64) for key in
                            ["input_ids", "attention_mask", "token_type_ids"]})
            if logits.shape[:2] != encoded["input_ids"].shape or logits.shape[2] != len(labels):
                raise ValueError("Invalid model output shape")
            if not np.isfinite(logits).all(): raise ValueError("Nonfinite model output")
            for batch_index, (row_index, words) in enumerate(pending):
                previous = None
                for position, word_index in enumerate(encoded.word_ids(batch_index)):
                    if word_index is None or word_index == previous: continue
                    previous = word_index
                    scores = logits[batch_index, position]
                    label = int(scores.argmax())
                    if not label: continue
                    confidence = float(1 / np.exp(scores - scores[label]).sum())
                    if not math.isfinite(confidence): raise ValueError("Nonfinite confidence")
                    proposal = decode_proposal(rows[row_index]["source"], words, word_index,
                                               labels[label], confidence, schema)
                    if proposal: records[row_index]["proposals"].append(proposal)
                    else: suppressed[tag_category(labels[label])] += 1
        except Exception:
            # Fail the whole affected sentence, including earlier successful chunks.
            # Store only a count, never sensitive source or provider exception text.
            for row_index, _ in pending: records[row_index]["failed"] = True
        pending.clear()
    for index, row in enumerate(rows):
        text = row["source"]
        protected = protected_spans(text)
        try:
            chunks = encoded_chunks(text, tokenizer, max_length, suppressed)
            if text.strip() and not chunks: records[index]["failed"] = True
            for words in chunks:
                if any(words[0]["start"] < end and start < words[-1]["end"] for start, end in protected):
                    suppressed["protected_windows"] += 1
                    continue
                pending.append((index, words))
                if len(pending) >= batch_size: flush()
        except Exception:
            records[index]["failed"] = True
    flush()
    for record in records:
        if record["failed"]: record["proposals"] = []
    return records, {"inference_failures": sum(record["failed"] for record in records),
                     "suppressed_by_guard": dict(suppressed), "inference_seconds": round(time.monotonic() - started, 3)}


def evaluate_records(rows, records, threshold, disabled=False, scorer="approximate", category_thresholds=None):
    if len(rows) != len(records): raise ValueError("Decoded population mismatch")
    predictions = [apply_proposals(row["source"], record["proposals"], threshold, disabled, category_thresholds)
                   for row, record in zip(rows, records)]
    result = score_predictions(rows, predictions, scorer, sum(record["failed"] for record in records))
    return {**result, "threshold": threshold, "disable_model_edits": disabled}


def onnx_predictor(model_dir, filename="model.onnx"):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    manifest = json.loads((model_dir / "manifest.json").read_text())
    labels = json.loads((model_dir / "labels.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_dir / filename), sess_options=options, providers=["CPUExecutionProvider"])
    return tokenizer, lambda feed: session.run(["logits"], feed)[0], labels, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--predictions", type=Path, help="Score a full-population actual browser report")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--filename", choices=["model.onnx", "model_quantized.onnx"], default="model.onnx")
    parser.add_argument("--scorer", choices=["approximate", "errant"], default="approximate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("Report already exists; choose a fresh filename")
    if args.predictions:
        result = score_browser_report(args.predictions, args.evaluation_dir, args.split, args.scorer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"sentences": result["conditions"][0]["sentences"], "conditions": len(result["conditions"])}))
        raise SystemExit(0)
    tokenizer, infer, labels, manifest = onnx_predictor(args.model, args.filename)
    rows = load_evaluation(args.evaluation_dir, args.split)
    records, inference = collect_proposals(rows, tokenizer, infer, labels, manifest["maxSequenceLength"], manifest.get("editSchema", 1))
    result = evaluate_records(rows, records, manifest["confidenceThreshold"], manifest.get("disableModelEdits", False),
                              args.scorer, manifest.get("confidenceThresholds"))
    result.update({"population_sha256": hash_file(args.evaluation_dir / f"{args.split}.jsonl"),
                   "model_sha256": hash_file(args.model / args.filename), "inference": inference,
                   "scope": "Python neural model with deployment guards, one pass; rules/combined/browser behavior requires separate browser evaluation."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
