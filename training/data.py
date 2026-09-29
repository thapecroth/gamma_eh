"""Original, deterministic synthetic English edit-tagging corpus (CC0-1.0)."""
import argparse
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import random
import re

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|[0-9]+|[^\w\s]", re.UNICODE)
VERBS = [("work", "works"), ("walk", "walks"), ("read", "reads"),
         ("write", "writes"), ("play", "plays"), ("learn", "learns"),
         ("travel", "travels"), ("cook", "cooks"), ("wait", "waits"),
         ("talk", "talks"), ("sleep", "sleeps"), ("run", "runs"),
         ("study", "studies"), ("watch", "watches"), ("go", "goes")]
SUBJECTS = [("I", False), ("You", False), ("We", False), ("They", False),
            ("He", True), ("She", True), ("My friend", True),
            ("The teacher", True), ("Our neighbor", True),
            ("The students", False), ("My friends", False),
            ("Our teachers", False), ("The children", False)]
ADVERBIALS = ["every morning", "after lunch", "before dinner", "on Mondays",
              "during the week", "in the evening", "at home", "at school",
              "in the garden", "with a friend", "near the library", "on weekends"]
ADJECTIVES = ["ready", "happy", "busy", "careful", "kind", "helpful", "tired",
              "quiet", "excited", "patient", "creative", "friendly"]
OBJECTS = ["a book", "a pencil", "an apple", "an orange", "a bicycle", "an umbrella",
           "a notebook", "an idea", "a camera", "an egg", "a ticket", "an envelope"]
TYPO_MAP = {"receive": "recieve", "believe": "beleive", "separate": "seperate",
            "necessary": "neccessary", "definitely": "definately", "tomorrow": "tommorow",
            "because": "becuase", "friend": "freind", "writing": "writting",
            "grammar": "grammer", "their": "thier", "address": "adress",
            "available": "avaliable", "really": "realy", "successful": "succesful",
            "different": "diffrent", "people": "peopel", "business": "buisness",
            "important": "importent", "the": "teh"}


def tokens(text):
    return TOKEN_RE.findall(text)


def render(words):
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(words))


def edit_tags(source, target):
    a, b = tokens(source), tokens(target)
    tags = ["KEEP"] * len(a)
    for kind, i, j, k, l in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if kind == "equal":
            continue
        if kind == "replace" and j - i == l - k:
            for p, word in zip(range(i, j), b[k:l]):
                tags[p] = "REPLACE:" + word.lower()
        elif kind == "delete":
            tags[i:j] = ["DELETE"] * (j - i)
        elif kind == "insert" and i > 0 and l - k == 1 and tags[i - 1] == "KEEP":
            tags[i - 1] = "APPEND:" + b[k].lower()
        else:
            raise ValueError("Pair cannot be represented with this single-pass edit vocabulary")
    return a, tags


def clean_sentences():
    # These templates and vocabulary are original; no customer text or external corpus.
    for subject, singular in SUBJECTS:
        for tail in ADVERBIALS:
            for base, third in VERBS:
                yield f"{subject} {third if singular else base} {tail}."
            copula = "am" if subject == "I" else "is" if singular else "are"
            past = "was" if singular or subject == "I" else "were"
            for adjective in ADJECTIVES:
                yield f"{subject} {copula} {adjective} {tail}."
                yield f"{subject} {past} {adjective} {tail}."
            for item in OBJECTS:
                yield f"{subject} {'has' if singular else 'have'} {item} {tail}."
                question_subject = subject if subject == 'I' else subject[0].lower() + subject[1:]
                yield f"{'Does' if singular else 'Do'} {question_subject} have {item} {tail}?"
                yield f"Would {question_subject} have {item} {tail}?"
            for base, _ in VERBS:
                question_subject = subject if subject == 'I' else subject[0].lower() + subject[1:]
                yield f"Can {question_subject} {base} {tail}?"
    for item in OBJECTS:
        for tail in ADVERBIALS:
            for verb in ["found", "bought", "borrowed", "saw", "needed", "carried"]:
                yield f"I {verb} {item} {tail}."
    for adjective in ADJECTIVES:
        for item in OBJECTS:
            yield f"You are {adjective} because you have {item}."
            yield f"Your friend is {adjective} because she has {item}."
            yield f"Their teacher is {adjective} today."
            yield f"They are {adjective} today."
            yield f"You're {adjective} because you have {item}."
            yield f"They're {adjective} because they have {item}."
            yield f"There is {item} near the door."
    yield "If she were here, she would help."
    yield "I recommend that he have more time."
    yield "The teacher had had a busy week."
    yield "We agreed that that answer was correct."
    yield "You're welcome at the library."
    yield "We watched her duck."
    typo_templates = [
        "Please {word} the message {tail}.", "I can {word} your letter {tail}."]
    for word in ["receive", "believe"]:
        for tail in ADVERBIALS:
            # 'believe your letter' is grammatical, but keep less contrived prompts.
            for template in typo_templates:
                yield template.format(word=word, tail=tail)
    for tail in ADVERBIALS:
        for item in OBJECTS:
            yield f"My friend is writing about grammar {tail}."
            yield f"The address is important because the business is successful."
            yield f"Different people really need {item} tomorrow."
            yield f"It is definitely necessary to keep {item} separate."
            yield f"Their teacher is available {tail}."


def corruptions(clean, rng):
    words = tokens(clean)
    choices = []
    # Context-aware corruptions: do not treat arbitrary valid tense changes as errors.
    subject = next((s for s, _ in sorted(SUBJECTS, key=lambda x: -len(x[0]))
                    if clean.startswith(s + " ")), None)
    verb_at = len(tokens(subject)) if subject else -1
    flip = {"is": "are", "are": "is", "am": "is", "was": "were", "were": "was",
            "has": "have", "have": "has"}
    for base, third in VERBS:
        flip[base] = third
        flip[third] = base
    if 0 <= verb_at < len(words) and words[verb_at] in flip:
        choices.append((verb_at, flip[words[verb_at]]))
    for i, word in enumerate(words):
        lower = word.lower()
        if lower in TYPO_MAP:
            choices.append((i, TYPO_MAP[lower]))
        if lower in {"a", "an"}:
            choices.append((i, "an" if lower == "a" else "a"))
        if lower == "your":
            choices.append((i, "You're" if word[0].isupper() else "you're"))
        if lower == "their":
            choices.append((i, "There" if word[0].isupper() else "there"))
        if lower == "there":
            choices.append((i, "Their" if word[0].isupper() else "their"))
        if lower == "you're":
            choices.append((i, "Your" if word[0].isupper() else "your"))
        if lower == "they're":
            choices.append((i, "Their" if word[0].isupper() else "their"))
    rng.shuffle(choices)
    variants = set()
    for i, wrong in choices[:3]:
        changed = words.copy()
        changed[i] = wrong.capitalize() if words[i][0].isupper() else wrong
        variants.add(render(changed))
    if len(choices) >= 2:
        changed = words.copy()
        for i, wrong in choices[:2]:
            changed[i] = wrong.capitalize() if words[i][0].isupper() else wrong
        variants.add(render(changed))
    # Duplicate-token removal and missing determiner insertion.
    if rng.random() < .25:
        i = rng.randrange(len(words) - 1)
        changed = words.copy()
        changed.insert(i, words[i])
        variants.add(render(changed))
    article_indices = [i for i, w in enumerate(words) if w in {"a", "an"} and i > 0]
    if article_indices and rng.random() < .4:
        changed = words.copy()
        changed.pop(rng.choice(article_indices))
        variants.add(render(changed))
    return sorted(variants - {clean})


def build(output, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    labels = {"KEEP", "DELETE"}
    for clean in sorted(set(clean_sentences())):
        group = hashlib.sha256(clean.encode()).hexdigest()
        bucket = int(group[:8], 16) % 100
        split = "train" if bucket < 80 else "dev" if bucket < 90 else "test"
        for source in [clean] + corruptions(clean, rng):
            source_tokens, tags = edit_tags(source, clean)
            labels.update(tags)
            splits[split].append({"source": source, "target": clean, "tokens": source_tokens,
                                  "tags": tags, "clean_group": group, "origin": "original-template-v1"})
    label_list = ["KEEP"] + sorted(labels - {"KEEP"})
    (output / "labels.json").write_text(json.dumps(label_list, indent=2) + "\n")
    manifest = {"schema": 1, "seed": seed, "license": "CC0-1.0", "origin": "original-template-v1",
                "split_policy": "SHA256(clean sentence), 80/10/10 before corruption",
                "limitation": "Templates shared across splits; metrics are synthetic in-distribution only.",
                "label_count": len(label_list), "splits": {}}
    for split, rows in splits.items():
        rng.shuffle(rows)
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][split] = {"rows": len(rows), "clean_rows": sum(r["source"] == r["target"] for r in rows),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "tags": dict(Counter(t for row in rows for t in row["tags"]))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["rows"] for k, v in manifest["splits"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.output, args.seed)
