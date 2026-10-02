"""Validation and provenance shared by teacher generation and public imports."""
import hashlib
import re
import unicodedata

CATEGORIES = {"agreement", "articles", "prepositions", "verb-tense", "spelling",
              "punctuation", "word-order", "clean", "mixed"}


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def group_id(target):
    return hashlib.sha256(normalized(target).casefold().encode()).hexdigest()


def evaluation_keys(rows):
    """Exclude both sides of every heldout pair from either training field."""
    result = set()
    for row in rows:
        values = [row["source"], *row.get("references", [])]
        if row.get("target") is not None:
            values.append(row["target"])
        result.update(normalized(value).casefold() for value in values)
    return result


def pair_id(source, target):
    return hashlib.sha256((normalized(source) + "\0" + normalized(target)).encode()).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pair(value, expected_category=None):
    if not isinstance(value, dict) or not isinstance(value.get("source"), str) or not isinstance(value.get("target"), str):
        raise ValueError("schema")
    source, target = value["source"].strip(), value["target"].strip()
    category = value.get("category", expected_category or "mixed")
    if category not in CATEGORIES:
        raise ValueError("category")
    if expected_category and category != expected_category:
        raise ValueError("category_mismatch")
    if not (3 <= len(source) <= 600 and 3 <= len(target) <= 600):
        raise ValueError("length")
    if any(ord(c) < 32 and c not in "\t\n\r" for c in source + target):
        raise ValueError("control_character")
    if not re.search(r"[A-Za-z]", source) or not re.search(r"[A-Za-z]", target):
        raise ValueError("no_english_letters")
    if re.search(r"https?://|\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b|\b(?:sk-|AKIA)[A-Za-z0-9]{12,}", source + " " + target):
        raise ValueError("contact_or_secret_pattern")
    equal = normalized(source) == normalized(target)
    if (category == "clean") != equal:
        raise ValueError("clean_mismatch")
    if len(target.split()) > 80 or len(source.split()) > 80:
        raise ValueError("word_limit")
    return {"source": source, "target": target, "category": category,
            "pair_id": pair_id(source, target), "clean_group": group_id(target)}
