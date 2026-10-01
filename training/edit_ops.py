"""Versioned, bounded edit tags shared with the local browser decoder."""
import difflib
import re

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|[0-9]+|[^\w\s]", re.UNICODE)
SUFFIXES = ("ADD_S", "REMOVE_S", "ADD_ES", "REMOVE_ES", "Y_TO_IES", "IES_TO_Y")
CASES = ("LOWER", "UPPER", "TITLE")
MAX_PAYLOAD_TOKENS = 4


def tokenize(text):
    return TOKEN_RE.findall(text)


def render(words, schema=1):
    text = re.sub(r"\s+([.,!?;:])", r"\1", " ".join(words))
    if schema == 2:
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
    return text


def preserve_case(original, replacement):
    if len(original) > 1 and original == original.upper():
        return replacement.upper()
    if original and "A" <= original[0] <= "Z":
        return replacement[:1].upper() + replacement[1:]
    return replacement


def suffix_transform(word, operation):
    if not re.fullmatch(r"[A-Za-z]+", word):
        raise ValueError("suffix_requires_ascii_word")
    lower = word.lower()
    if operation == "ADD_S": value = lower + "s"
    elif operation == "ADD_ES": value = lower + "es"
    elif operation == "REMOVE_S" and lower.endswith("s") and len(lower) > 1: value = lower[:-1]
    elif operation == "REMOVE_ES" and lower.endswith("es") and len(lower) > 2: value = lower[:-2]
    elif operation == "Y_TO_IES" and lower.endswith("y") and len(lower) > 1: value = lower[:-1] + "ies"
    elif operation == "IES_TO_Y" and lower.endswith("ies") and len(lower) > 3: value = lower[:-3] + "y"
    else: raise ValueError("invalid_suffix_transform")
    return preserve_case(word, value)


def payload_words(value):
    words = tokenize(value)
    if not 1 <= len(words) <= MAX_PAYLOAD_TOKENS or render(words, 2) != value:
        raise ValueError("invalid_exact_payload")
    return words


def decode_word(word, tag, schema=1):
    """Return output tokens; inference spacing uses source offsets separately."""
    if tag == "KEEP": return [word]
    if tag == "DELETE": return []
    if tag.startswith("REPLACE:"):
        value = tag[8:]
        return [preserve_case(word, value)]
    if tag.startswith("APPEND:"): return [word, tag[7:]]
    if schema != 2: raise ValueError("unsupported_edit_tag")
    for prefix in ("REPLACE_EXACT:", "APPEND_EXACT:", "PREPEND_EXACT:"):
        if tag.startswith(prefix):
            value = payload_words(tag[len(prefix):])
            if prefix == "APPEND_EXACT:": return [word] + value
            if prefix == "PREPEND_EXACT:": return value + [word]
            return value
    if tag.startswith("CASE:"):
        operation = tag[5:]
        if operation == "LOWER": return [word.lower()]
        if operation == "UPPER": return [word.upper()]
        if operation == "TITLE": return [word[:1].upper() + word[1:].lower()]
    if tag.startswith("SUFFIX:"): return [suffix_transform(word, tag[7:])]
    raise ValueError("unsupported_edit_tag")


def reconstruct(words, tags, schema=1):
    if len(words) != len(tags): raise ValueError("tag_length_mismatch")
    return render([token for word, tag in zip(words, tags)
                   for token in decode_word(word, tag, schema)], schema)


def replacement_tag(source, target):
    if source == target: return "KEEP"
    for operation in CASES:
        tag = "CASE:" + operation
        if decode_word(source, tag, 2) == [target]: return tag
    for operation in SUFFIXES:
        try:
            if suffix_transform(source, operation) == target: return "SUFFIX:" + operation
        except ValueError:
            pass
    return "REPLACE_EXACT:" + target


def align_v2(source, target):
    words, wanted = tokenize(source), tokenize(target)
    if not words: raise ValueError("empty_source_tokens")
    tags = ["KEEP"] * len(words)
    for kind, i, j, k, l in difflib.SequenceMatcher(a=words, b=wanted, autojunk=False).get_opcodes():
        if kind == "equal": continue
        if kind == "delete":
            tags[i:j] = ["DELETE"] * (j - i)
        elif kind == "replace":
            if j - i == l - k:
                for position, value in zip(range(i, j), wanted[k:l]):
                    prior = tags[position]
                    if prior.startswith("PREPEND_EXACT:"):
                        prefix = payload_words(prior[len("PREPEND_EXACT:"):])
                        combined = prefix + [value]
                        if len(combined) > MAX_PAYLOAD_TOKENS: raise ValueError("combined_payload_limit")
                        tags[position] = "REPLACE_EXACT:" + render(combined, 2)
                    else:
                        tags[position] = replacement_tag(words[position], value)
            else:
                remaining = wanted[k:l]
                if tags[i].startswith("PREPEND_EXACT:"):
                    remaining = payload_words(tags[i][len("PREPEND_EXACT:"):]) + remaining
                for position in range(i, j):
                    block, remaining = remaining[:MAX_PAYLOAD_TOKENS], remaining[MAX_PAYLOAD_TOKENS:]
                    tags[position] = "REPLACE_EXACT:" + render(block, 2) if block else "DELETE"
                if remaining: raise ValueError("replacement_payload_limit")
        elif kind == "insert":
            if l - k > MAX_PAYLOAD_TOKENS: raise ValueError("insertion_payload_limit")
            block = wanted[k:l]
            if i > 0 and tags[i - 1] == "KEEP":
                tags[i - 1] = "APPEND_EXACT:" + render(block, 2)
            elif i == 0 and tags[0] == "KEEP":
                tags[0] = "PREPEND_EXACT:" + render(block, 2)
            else:
                position = i - 1 if i > 0 else 0
                existing = decode_word(words[position], tags[position], 2)
                merged = existing + block if i > 0 else block + existing
                if not 1 <= len(merged) <= MAX_PAYLOAD_TOKENS:
                    raise ValueError("combined_payload_limit")
                tags[position] = "REPLACE_EXACT:" + render(merged, 2)
    if tokenize(reconstruct(words, tags, 2)) != wanted:
        raise ValueError("alignment_roundtrip")
    return words, tags


def tag_category(tag):
    if tag == "KEEP": return "keep"
    if tag == "DELETE": return "delete"
    if tag.startswith("CASE:"): return "case"
    if tag.startswith("SUFFIX:"): return "suffix"
    value = tag.split(":", 1)[-1]
    if value and not re.search(r"\w", value): return "punctuation"
    if tag.startswith(("APPEND:", "APPEND_EXACT:")): return "append"
    if tag.startswith("PREPEND_EXACT:"): return "prepend"
    return "replace"
