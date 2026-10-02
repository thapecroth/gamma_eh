"""Versioned clean-context recipes and bounded deterministic corruptions.

Recipe eligibility is a syntactic precondition, not proof of English validity.
The complete clean sibling and corrupted pair must pass machine screening.
"""
import hashlib
import json
from pathlib import Path
import re

from edit_ops import TOKEN_RE, align_v2, preserve_case, reconstruct
from evaluate import (ARTICLE_HEADS, COUNTABLE_OBJECTS, VERB_FAMILIES,
                      apply_proposals, decode_proposal, protected_spans)
from pairs import CATEGORIES, normalized

REGISTRY_PATH = Path(__file__).parent / "recipes/teacher-v1.json"
MUTATORS = {"agreement", "article_sound", "article_missing", "numeric_count",
            "question_auxiliary", "perfect_participle", "mass_quantifier",
            "governed_preposition", "duplicate_function", "identity"}
COUNT_NOUNS = {"book": "books", "pencil": "pencils", "bicycle": "bicycles",
               "camera": "cameras", "notebook": "notebooks", "ticket": "tickets",
               "message": "messages", "pen": "pens", "project": "projects",
               "apple": "apples", "orange": "oranges", "egg": "eggs",
               "umbrella": "umbrellas", "envelope": "envelopes", "idea": "ideas",
               "library": "libraries"}
PARTICIPLES = {"eaten": "ate", "taken": "took", "broken": "broke",
               "chosen": "chose", "written": "wrote", "gone": "went", "seen": "saw"}
MASS_NOUNS = {"information", "advice", "equipment", "furniture", "homework", "traffic"}
SINGULAR_HEADS = {"friend", "teacher", "neighbor", "student", "child", "colleague",
                  "manager", "cat", "dog", "box", "list", "picture", "price"}
PLURAL_HEADS = {"friends", "teachers", "neighbors", "students", "children", "colleagues",
                "managers", "cats", "dogs", "people", "boxes", "lists", "pictures", "prices"}
MODIFIER_OBJECTS = {"items", "cables", "books", "tickets", "projects", "buildings"}
HORIZONTAL = re.compile(r"^[ \t]+$")
BOUNDARY = re.compile(r"^(?:[.!?](?:\s|$)|\s+(?:for|from|at|in|on|near|with|during|about|of|to|after|before|between)\b|\s*$)", re.I)


def load_registry(path=REGISTRY_PATH):
    value = json.loads(Path(path).read_text())
    recipes = value.get("recipes") if isinstance(value, dict) else None
    if not isinstance(value, dict) or value.get("schema") != 1 or not isinstance(recipes, list) or not recipes:
        raise ValueError("recipe_registry_schema")
    seen = set()
    for recipe in recipes:
        if (not isinstance(recipe, dict) or not isinstance(recipe.get("id"), str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", recipe["id"])
                or recipe["id"] in seen or type(recipe.get("version")) is not int or recipe["version"] < 1
                or not isinstance(recipe.get("category"), str) or recipe["category"] not in CATEGORIES
                or not isinstance(recipe.get("mutator"), str) or recipe["mutator"] not in MUTATORS
                or not isinstance(recipe.get("instructions"), str) or not recipe["instructions"]
                or not isinstance(recipe.get("constraints"), list) or not all(isinstance(x, str) for x in recipe["constraints"])
                or not isinstance(recipe.get("examples"), list) or not all(isinstance(x, str) for x in recipe["examples"])):
            raise ValueError("recipe_registry_entry")
        seen.add(recipe["id"])
    return value


def lexical_family(text):
    """Stable surface cluster: case, punctuation, apostrophe and numeral variants.

    This is deliberately conservative; it does not detect semantic paraphrases.
    Raw texts and case are preserved. Clustering only affects family splitting.
    """
    values = ["#" if token.isdigit() else token.casefold().replace("’", "'")
              for token in TOKEN_RE.findall(normalized(text)) if any(char.isalnum() for char in token)]
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def synthetic_split(family):
    bucket = int(family[:8], 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def words_with_offsets(text):
    return [{"text": match.group(), "start": match.start(), "end": match.end()}
            for match in TOKEN_RE.finditer(text)]


def _replace(candidates, text, word, value):
    candidates.append({"start": word["start"], "end": word["end"],
                       "clean": word["text"], "corrupt": preserve_case(word["text"], value)})


def _subject_before(text, word):
    match = re.search(r"(?:^|[.!?][ \t]+)(?:(i|you|we|they|he|she|it)|(?:my|your|our|his|her|their|the|a|an)[ \t]+([A-Za-z]+)(?:[ \t]+of[ \t]+(?:the[ \t]+)?([A-Za-z]+))?)[ \t]+$",
                      text[:word["start"]], re.I)
    if not match: return None
    pronoun, head, modifier = (value.lower() if value else "" for value in match.groups())
    if modifier and modifier not in MODIFIER_OBJECTS: return None
    if pronoun: return pronoun, pronoun in {"he", "she", "it"}
    if head in SINGULAR_HEADS: return "", True
    if head in PLURAL_HEADS: return "", False
    return None


def mutation_candidates(text, mutator):
    words = words_with_offsets(text)
    candidates = []
    for index, word in enumerate(words):
        lower = word["text"].lower()
        next_word = words[index + 1] if index + 1 < len(words) else None
        previous = words[index - 1] if index else None
        contiguous = next_word and HORIZONTAL.fullmatch(text[word["end"]:next_word["start"]])
        if mutator == "agreement":
            subject = _subject_before(text, word)
            if not subject or re.search(r"^[^.!?]*\?", text[word["end"]:]): continue
            pronoun, singular = subject
            for singular_verb, plural_verb, first, habitual in VERB_FAMILIES:
                if lower not in {singular_verb, plural_verb, first}: continue
                if habitual and (plural_verb == "read" or not re.search(
                        r"^[^.!?\n]*\bevery[ \t]+(?:morning|day|evening|night|week)\b[^.!?\n]*(?:\.|$)",
                        text[word["end"]:], re.I)): continue
                expected = first if pronoun == "i" and first else singular_verb if singular else plural_verb
                if lower == expected:
                    wrong = plural_verb if expected != plural_verb else singular_verb
                    _replace(candidates, text, word, wrong)
        elif mutator == "article_sound" and contiguous and lower in {"a", "an"}:
            if ARTICLE_HEADS.get(next_word["text"].lower()) == lower:
                _replace(candidates, text, word, "an" if lower == "a" else "a")
        elif mutator == "article_missing" and contiguous and previous and lower in {"a", "an"}:
            if (previous["text"].lower() in {"has", "have"} and next_word["text"].lower() in COUNTABLE_OBJECTS
                    and ARTICLE_HEADS.get(next_word["text"].lower()) == lower
                    and HORIZONTAL.fullmatch(text[previous["end"]:word["start"]])
                    and BOUNDARY.match(text[next_word["end"]:])):
                candidates.append({"start": word["start"], "end": next_word["start"],
                                   "clean": text[word["start"]:next_word["start"]], "corrupt": ""})
        elif mutator == "numeric_count" and previous and lower in COUNT_NOUNS.values():
            if (previous["text"].lower() in {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
                    and HORIZONTAL.fullmatch(text[previous["end"]:word["start"]])
                    and BOUNDARY.match(text[word["end"]:])):
                singular = next(key for key, value in COUNT_NOUNS.items() if value == lower)
                _replace(candidates, text, word, singular)
        elif mutator == "question_auxiliary" and index == 0 and contiguous and lower in {"do", "does"}:
            subject = next_word["text"].lower()
            if subject in {"i", "you", "we", "they", "he", "she", "it"} and text.rstrip().endswith("?"):
                expected = "does" if subject in {"he", "she", "it"} else "do"
                if lower == expected: _replace(candidates, text, word, "do" if lower == "does" else "does")
        elif mutator == "perfect_participle" and previous and lower in PARTICIPLES:
            if previous["text"].lower() in {"has", "have", "had"} and HORIZONTAL.fullmatch(text[previous["end"]:word["start"]]):
                _replace(candidates, text, word, PARTICIPLES[lower])
        elif mutator == "mass_quantifier" and contiguous:
            head = next_word["text"].lower()
            if ((lower == "much" and head in MASS_NOUNS) or
                    (lower == "many" and head in COUNT_NOUNS.values())) and BOUNDARY.match(text[next_word["end"]:]):
                _replace(candidates, text, word, "many" if lower == "much" else "much")
        elif mutator == "governed_preposition" and previous and contiguous:
            governor = previous["text"].lower()
            expected = ("on" if governor in {"rely", "relies", "relied", "depend", "depends", "depended"}
                        else {"interested": "in", "responsible": "for", "similar": "to"}.get(governor))
            if expected == lower and next_word["text"].lower() in {"the", "this", "that", "these", "those", "my", "your", "our", "their", "his", "her"}:
                _replace(candidates, text, word, "of" if lower != "of" else "at")
        elif mutator == "duplicate_function" and lower in {"the", "a", "an", "can", "because"}:
            if (next_word and next_word["text"].lower() == lower) or (previous and previous["text"].lower() == lower): continue
            candidates.append({"start": word["end"], "end": word["end"],
                               "clean": "", "corrupt": " " + word["text"].lower()})
    return candidates


def derive_variant(clean, recipe, nonce):
    if recipe["mutator"] == "identity": return None
    candidates = mutation_candidates(clean, recipe["mutator"])
    if not candidates: return None
    mutation = candidates[int(hashlib.sha256(nonce.encode()).hexdigest()[:16], 16) % len(candidates)]
    start, end = mutation["start"], mutation["end"]
    source = clean[:start] + mutation["corrupt"] + clean[end:]
    corrupt_end = start + len(mutation["corrupt"])
    if source[:start] + mutation["clean"] + source[corrupt_end:] != clean:
        raise ValueError("mutation_roundtrip")
    return {"source": source, "target": clean, "category": recipe["category"],
            "mutation": {**mutation, "source_end": corrupt_end,
                         "offset_unit": "python_unicode_codepoints",
                         "clean_span_utf16": [len(clean[:offset].encode("utf-16-le")) // 2 for offset in (start, end)],
                         "source_span_utf16": [len(source[:offset].encode("utf-16-le")) // 2 for offset in (start, corrupt_end)]}}


def guard_coverage(source, target):
    """Alignment and current decoded-source roundtrip, not model accuracy."""
    try:
        tokens, tags = align_v2(source, target)
        if reconstruct(tokens, tags, 2) != target:
            return {"alignment": False, "guarded": False, "reason": "exact_spacing_roundtrip_unsupported"}
    except ValueError:
        return {"alignment": False, "guarded": False, "reason": "alignment_unsupported"}
    if source != target and protected_spans(source):
        return {"alignment": True, "guarded": False, "reason": "protected_source",
                "tags": tags, "protected_window_policy": "conservative_whole_source_abstention"}
    words = words_with_offsets(source)
    if [word["text"] for word in words] != tokens:
        return {"alignment": False, "guarded": False, "reason": "token_alignment_mismatch"}
    proposals = [proposal for index, tag in enumerate(tags) if tag != "KEEP"
                 for proposal in [decode_proposal(source, words, index, tag, 1., 2)] if proposal]
    guarded = apply_proposals(source, proposals, 0.) == target
    return {"alignment": True, "guarded": guarded,
            "reason": None if guarded else "guarded_roundtrip_unsupported", "tags": tags}
