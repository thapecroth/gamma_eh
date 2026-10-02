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
PRONOUNS = ("i", "you", "we", "they", "he", "she", "it")
SINGULAR_PRONOUNS = {"he", "she", "it"}
SUBJECT_DETERMINERS = ("my", "your", "our", "his", "her", "their", "the", "a", "an")
HABITUAL_TIMES = ("morning", "day", "evening", "night", "week")
NUMBER_WORDS = {word: index for index, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
COUNT_NUMERALS = {word for word, value in NUMBER_WORDS.items() if value >= 2} | {str(value) for value in NUMBER_WORDS.values() if value >= 2}
POSSESSION_AUXILIARIES = {"has", "have"}
PERFECT_AUXILIARIES = POSSESSION_AUXILIARIES | {"had"}
HEAD_PREPOSITIONS = ("for", "from", "at", "in", "on", "near", "with", "during", "about", "of", "to", "after", "before", "between")
OBJECT_STARTERS = {"the", "this", "that", "these", "those", "my", "your", "our", "their", "his", "her"}
GOVERNED_PREPOSITIONS = {**dict.fromkeys(("rely", "relies", "relied", "depend", "depends", "depended"), "on"),
                        "interested": "in", "responsible": "for", "similar": "to"}
DUPLICATE_WORDS = {"the", "a", "an", "can", "because"}
HORIZONTAL = re.compile(r"^[ \t]+$")
BOUNDARY = re.compile(r"^(?:[.!?](?:\s|$)|\s+(?:" + "|".join(HEAD_PREPOSITIONS) + r")\b|\s*$)", re.I)


def generation_contract(mutator):
    """Generator preconditions derive from the same constants the mutator uses."""
    if mutator not in MUTATORS: raise ValueError("unknown_mutator")
    head_boundary = {"allowed_following_prepositions": list(HEAD_PREPOSITIONS),
                     "otherwise_follow_with": "sentence-ending punctuation or end of text",
                     "no_noun_compound_or_intervening_adverb": True}
    value = {"version": 2, "mutator": mutator, "keep_situations_and_free_tails_varied": True}
    if mutator == "agreement":
        value.update({"sentence_start_pronouns": list(PRONOUNS), "singular_pronouns": sorted(SINGULAR_PRONOUNS),
                      "subject_determiners": list(SUBJECT_DETERMINERS), "singular_heads": sorted(SINGULAR_HEADS),
                      "plural_heads": sorted(PLURAL_HEADS), "optional_head_modifier": "of [the] OBJECT",
                      "modifier_objects": sorted(MODIFIER_OBJECTS),
                      "subject_forms": ["PRONOUN VERB", "DETERMINER HEAD VERB", "DETERMINER HEAD of [the] OBJECT VERB"],
                      "verb_immediately_follows_subject": True, "no_adjectives_inside_subject": True,
                      "declarative_only": True, "verb_families": [
                          {"singular": singular, "plural": plural, "first_person": first,
                           "requires_habitual_cue": habitual} for singular, plural, first, habitual in VERB_FAMILIES if plural != "read"],
                      "habitual_cue": "every TIME", "habitual_times": list(HABITUAL_TIMES)})
    elif mutator == "article_sound":
        value.update({"article_for_immediate_next_word": dict(sorted(ARTICLE_HEADS.items())), "no_intervening_word": True})
    elif mutator == "article_missing":
        value.update({"frame": "has/have a/an NOUN", "auxiliaries": sorted(POSSESSION_AUXILIARIES),
                      "article_for_noun": {word: ARTICLE_HEADS[word] for word in sorted(COUNTABLE_OBJECTS & ARTICLE_HEADS.keys())},
                      "head_boundary": head_boundary, "adjacent_auxiliary_article_noun": True})
    elif mutator == "numeric_count":
        value.update({"frame": "NUMERAL PLURAL_HEAD", "numerals": sorted(COUNT_NUMERALS),
                      "singular_plural": dict(sorted(COUNT_NOUNS.items())), "head_boundary": head_boundary})
    elif mutator == "question_auxiliary":
        value.update({"sentence_initial": True, "question_mark_required": True,
                      "auxiliary_for_pronoun": {word: "does" if word in SINGULAR_PRONOUNS else "do" for word in PRONOUNS},
                      "frame": "Do/Does PRONOUN BASE_VERB ...?", "no_intervening_adverb": True})
    elif mutator == "perfect_participle":
        value.update({"auxiliaries": sorted(PERFECT_AUXILIARIES), "participle_to_past": dict(sorted(PARTICIPLES.items())),
                      "frame": "has/have/had PARTICIPLE", "no_intervening_adverb": True})
    elif mutator == "mass_quantifier":
        value.update({"much_nouns": sorted(MASS_NOUNS), "many_nouns": sorted(COUNT_NOUNS.values()),
                      "quantifier_immediately_precedes_noun": True, "head_boundary": head_boundary})
    elif mutator == "governed_preposition":
        value.update({"governor_preposition": dict(sorted(GOVERNED_PREPOSITIONS.items())),
                      "object_starters": sorted(OBJECT_STARTERS), "frame": "GOVERNOR PREPOSITION OBJECT_STARTER ...",
                      "no_intervening_adverb": True})
    elif mutator == "duplicate_function":
        value.update({"words": sorted(DUPLICATE_WORDS), "no_existing_duplicate": True, "no_deliberate_emphasis_or_quotes": True})
    else:
        value.update({"identity_only": True, "preserve_informal_case_names_valid_tense_and_register": True})
    return value


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
    values = ["#" if token.isdigit() or token.casefold() in NUMBER_WORDS else token.casefold().replace("’", "'")
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
    match = re.search(r"(?:^|[.!?][ \t]+)(?:(" + "|".join(PRONOUNS) + r")|(?:" + "|".join(SUBJECT_DETERMINERS) + r")[ \t]+([A-Za-z]+)(?:[ \t]+of[ \t]+(?:the[ \t]+)?([A-Za-z]+))?)[ \t]+$",
                      text[:word["start"]], re.I)
    if not match: return None
    pronoun, head, modifier = (value.lower() if value else "" for value in match.groups())
    if modifier and modifier not in MODIFIER_OBJECTS: return None
    if pronoun: return pronoun, pronoun in SINGULAR_PRONOUNS
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
                        r"^[^.!?\n]*\bevery[ \t]+(?:" + "|".join(HABITUAL_TIMES) + r")\b[^.!?\n]*(?:\.|$)",
                        text[word["end"]:], re.I)): continue
                expected = first if pronoun == "i" and first else singular_verb if singular else plural_verb
                if lower == expected:
                    wrong = plural_verb if expected != plural_verb else singular_verb
                    _replace(candidates, text, word, wrong)
        elif mutator == "article_sound" and contiguous and lower in {"a", "an"}:
            if ARTICLE_HEADS.get(next_word["text"].lower()) == lower:
                _replace(candidates, text, word, "an" if lower == "a" else "a")
        elif mutator == "article_missing" and contiguous and previous and lower in {"a", "an"}:
            if (previous["text"].lower() in POSSESSION_AUXILIARIES and next_word["text"].lower() in COUNTABLE_OBJECTS
                    and ARTICLE_HEADS.get(next_word["text"].lower()) == lower
                    and HORIZONTAL.fullmatch(text[previous["end"]:word["start"]])
                    and BOUNDARY.match(text[next_word["end"]:])):
                candidates.append({"start": word["start"], "end": next_word["start"],
                                   "clean": text[word["start"]:next_word["start"]], "corrupt": ""})
        elif mutator == "numeric_count" and previous and lower in COUNT_NOUNS.values():
            if (previous["text"].lower() in COUNT_NUMERALS
                    and HORIZONTAL.fullmatch(text[previous["end"]:word["start"]])
                    and BOUNDARY.match(text[word["end"]:])):
                singular = next(key for key, value in COUNT_NOUNS.items() if value == lower)
                _replace(candidates, text, word, singular)
        elif mutator == "question_auxiliary" and index == 0 and contiguous and lower in {"do", "does"}:
            subject = next_word["text"].lower()
            if subject in PRONOUNS and text.rstrip().endswith("?"):
                expected = "does" if subject in SINGULAR_PRONOUNS else "do"
                if lower == expected: _replace(candidates, text, word, "do" if lower == "does" else "does")
        elif mutator == "perfect_participle" and previous and lower in PARTICIPLES:
            if previous["text"].lower() in PERFECT_AUXILIARIES and HORIZONTAL.fullmatch(text[previous["end"]:word["start"]]):
                _replace(candidates, text, word, PARTICIPLES[lower])
        elif mutator == "mass_quantifier" and contiguous:
            head = next_word["text"].lower()
            if ((lower == "much" and head in MASS_NOUNS) or
                    (lower == "many" and head in COUNT_NOUNS.values())) and BOUNDARY.match(text[next_word["end"]:]):
                _replace(candidates, text, word, "many" if lower == "much" else "much")
        elif mutator == "governed_preposition" and previous and contiguous:
            governor = previous["text"].lower()
            expected = GOVERNED_PREPOSITIONS.get(governor)
            if expected == lower and next_word["text"].lower() in OBJECT_STARTERS:
                _replace(candidates, text, word, "of" if lower != "of" else "at")
        elif mutator == "duplicate_function" and lower in DUPLICATE_WORDS:
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
            return {"alignment": False, "guarded": source == target, "reason": "exact_spacing_roundtrip_unsupported",
                    "guard_reason": "identity_no_edits" if source == target else "alignment_unsupported"}
    except ValueError:
        return {"alignment": False, "guarded": source == target, "reason": "alignment_unsupported",
                "guard_reason": "identity_no_edits" if source == target else "alignment_unsupported"}
    if source == target:
        return {"alignment": True, "guarded": True, "reason": None, "guard_reason": "identity_no_edits", "tags": tags}
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
