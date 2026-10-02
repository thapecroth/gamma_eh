"""Pure-stdlib spelling evidence matching the bundled legacy JS decoder."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata


@lru_cache(maxsize=1)
def vocabulary():
    source = Path(__file__).resolve().parent.parent / "packages/engine/src"
    words = re.search(r"export const dictionaryWords = '([^']+)';", (source / "dictionary.generated.ts").read_text())
    technical = re.search(r"const technicalWords = new Set\('([^']+)'\.split", (source / "spelling.ts").read_text())
    if not words or not technical:
        raise ValueError("Bundled spelling vocabulary format changed")
    return set(words[1].split()) | set(technical[1].split())


def known_word(word):
    return word in vocabulary()


@lru_cache(maxsize=500)
def whole_words(text):
    def part(char):
        category = unicodedata.category(char)
        return category[0] in "LMN" or category == "Pc" or char in "\u200c\u200d"
    spans = set()
    index = 0
    while index < len(text):
        if not part(text[index]):
            index += 1
            continue
        start = index
        while index < len(text) and part(text[index]):
            index += 1
        while index + 1 < len(text) and text[index] in "'’\u2010\u2011-" and part(text[index + 1]):
            index += 1
            while index < len(text) and part(text[index]):
                index += 1
        if re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", text[start:index]):
            spans.add((start, index))
    return frozenset(spans)


def spelling_distance(a, b, limit):
    if abs(len(a)-len(b)) > limit:
        return limit+1
    previous_previous = []
    previous = list(range(len(b)+1))
    for i in range(1, len(a)+1):
        current = [limit+1]*(len(b)+1)
        current[0] = i
        for j in range(max(1, i-limit), min(len(b), i+limit)+1):
            current[j] = min(previous[j]+1, current[j-1]+1, previous[j-1]+(a[i-1] != b[j-1]))
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                current[j] = min(current[j], previous_previous[j-2]+1)
        if min(current) > limit:
            return limit+1
        previous_previous, previous = previous, current
    return min(previous[len(b)], limit+1)
