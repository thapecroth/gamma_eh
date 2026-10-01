"""Original CC0 training augmentation and a separate adversarial synthetic holdout.

Gold edits come from explicit corruptions of grammatical targets, never model
predictions. This remains synthetic evaluation, not a natural-language benchmark.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from data import edit_tags
from pairs import group_id, hash_file, normalized, pair_id

VERSION = "original-diverse-synthetic-v1"
TRAIN_NOUNS = [
    ("engineer", "engineers"), ("gardener", "gardeners"),
    ("librarian", "librarians"), ("nurse", "nurses"), ("pilot", "pilots"),
    ("baker", "bakers"), ("cat", "cats"), ("dog", "dogs"),
    ("rabbit", "rabbits"), ("painter", "painters"),
    ("musician", "musicians"), ("scientist", "scientists"),
    ("doctor", "doctors"), ("chef", "chefs"),
    ("visitor", "visitors"), ("volunteer", "volunteers"),
]
CHALLENGE_NOUNS = [
    ("archivist", "archivists"), ("mechanic", "mechanics"),
    ("otter", "otters"), ("owl", "owls"), ("child", "children"),
    ("woman", "women"), ("man", "men"), ("mouse", "mice"),
]
TRAIN_ADJECTIVES = ["hungry", "calm", "alert", "diligent", "curious", "eager", "polite", "restless"]
CHALLENGE_ADJECTIVES = ["relieved", "exhausted", "astonished", "optimistic"]
TRAIN_TAILS = ["after the workshop", "before the concert", "during the rehearsal",
               "at the exhibition", "outside the studio", "inside the museum"]
CHALLENGE_TAILS = ["following the inspection", "before the ceremony"]
TRAIN_TYPOS = [("calendar", "calender"), ("tomorrow", "tommorow"),
               ("separate", "seperate"), ("address", "adress"),
               ("business", "buisness"), ("grammar", "grammer"),
               ("writing", "writting"), ("necessary", "neccessary")]
CHALLENGE_TYPOS = [("environment", "enviroment"), ("occurrence", "occurence"),
                   ("maintenance", "maintainance"), ("accommodation", "accomodation"),
                   ("recommendation", "recomendation"), ("embarrassment", "embarassment"),
                   ("privilege", "privelege"), ("independent", "independant")]


def utf16_length(text):
    return len(text.encode("utf-16-le")) // 2


def mutation(clean, original, wrong, occurrence=0):
    """Identify the target span before corrupting it; callers choose its context."""
    start = -1
    for _ in range(occurrence + 1):
        start = clean.index(original, start + 1)
    return start, start + len(original), wrong


def example(clean, family, changes=(), holdout="augmentation"):
    pieces = []
    gold = []
    cursor = 0
    for start, end, wrong in sorted(changes):
        if start < cursor or end < start or end > len(clean):
            raise ValueError("Overlapping or invalid corruption")
        pieces.append(clean[cursor:start])
        offset = utf16_length("".join(pieces))
        pieces.append(wrong)
        gold.append({"start": offset, "end": offset + utf16_length(wrong),
                     "original": wrong, "replacement": clean[start:end]})
        cursor = end
    pieces.append(clean[cursor:])
    source = "".join(pieces)
    return {"id": pair_id(source, clean), "source": source, "target": clean,
            "family": family, "holdout": holdout, "clean": source == clean,
            "goldEdits": gold, "clean_group": group_id(clean),
            "origin": VERSION, "license": "CC0-1.0"}


def paired(clean, family, changes, holdout="augmentation"):
    return [example(clean, family, holdout=holdout), example(clean, family, changes, holdout)]


def subjects(noun_pairs):
    return [(word, singular) for singular, index in [(True, 0), (False, 1)]
            for pair in noun_pairs for word in [pair[index]]]


def training_examples():
    heads = subjects(TRAIN_NOUNS)
    for index, (head, singular) in enumerate(heads):
        distractor = TRAIN_NOUNS[(index + 3) % len(TRAIN_NOUNS)][1 if singular else 0]
        copula, wrong = ("is", "are") if singular else ("are", "is")
        past, wrong_past = ("was", "were") if singular else ("were", "was")
        possession, wrong_possession = ("has", "have") if singular else ("have", "has")
        for adjective, tail in itertools.product(TRAIN_ADJECTIVES, TRAIN_TAILS):
            for subject, family in [(f"The {head}", "simple-copula"),
                                    (f"The {head} beside the {distractor}", "prepositional-copula")]:
                clean = f"{subject} {copula} {adjective} {tail}."
                yield from paired(clean, family, [mutation(clean, f" {copula} ", f" {wrong} ")])
            clean = f"The {head} {past} {adjective} {tail}."
            yield from paired(clean, "simple-past", [mutation(clean, f" {past} ", f" {wrong_past} ")])
        for item, tail in itertools.product(["a lantern", "a basket", "an instrument", "an invitation"], TRAIN_TAILS):
            clean = f"The {head} {possession} {item} {tail}."
            yield from paired(clean, "simple-possession", [mutation(clean, f" {possession} ", f" {wrong_possession} ")])
        auxiliary = "does" if singular else "do"
        for tail in TRAIN_TAILS:
            clean = f"The {head} {auxiliary} not have a lantern {tail}."
            yield from paired(clean, "negative-possession", [mutation(clean, " have ", " has ")])
            clean = f"The {head} can have a basket {tail}."
            yield from paired(clean, "can-possession", [mutation(clean, " have ", " has ")])
        for base, third in [("walk", "walks"), ("cook", "cooks"), ("work", "works"), ("sleep", "sleeps")]:
            for cue in ["every morning", "every evening", "every week"]:
                verb, wrong_verb = (third, base) if singular else (base, third)
                clean = f"The {head} {verb} {cue}."
                yield from paired(clean, "habitual-verb", [mutation(clean, f" {verb} ", f" {wrong_verb} ")])
    for correct, typo in TRAIN_TYPOS:
        for tail in TRAIN_TAILS:
            # Mentioning a word avoids grammatically nonsensical adjective/noun substitutions.
            clean = f"We checked the word {correct} {tail}."
            yield from paired(clean, "spelling-in-context", [mutation(clean, correct, typo)])
    for item in ["a lantern", "a basket", "an instrument", "an invitation"]:
        for tail in TRAIN_TAILS:
            article, noun = item.split(" ", 1)
            clean = f"We carried {item} {tail}."
            yield from paired(clean, "ordinary-article", [mutation(clean, f" {article} {noun}", f" {'an' if article == 'a' else 'a'} {noun}")])


def challenge_examples():
    """16 error/clean pairs per family, plus 32 clean construction controls."""
    heads = subjects(CHALLENGE_NOUNS)
    for index, (head, singular) in enumerate(heads):
        distractor = CHALLENGE_NOUNS[(index + 1) % len(CHALLENGE_NOUNS)][1 if singular else 0]
        adjective = CHALLENGE_ADJECTIVES[index % len(CHALLENGE_ADJECTIVES)]
        tail = CHALLENGE_TAILS[(index // 2) % len(CHALLENGE_TAILS)]
        copula, wrong = ("is", "are") if singular else ("are", "is")
        past, wrong_past = ("was", "were") if singular else ("were", "was")
        possession, wrong_possession = ("has", "have") if singular else ("have", "has")
        forms = [
            ("reserved-subjects", f"The {head} {copula} {adjective} {tail}.", "lexical"),
            ("agreement-attraction", f"The {head} near the {distractor} {copula} {adjective} {tail}.", "lexical"),
            ("relative-clauses", f"The {head} who helped the {distractor} {copula} {adjective} {tail}.", "structural"),
            ("embedded-agreement", f"We learned that the {head} {copula} {adjective} {tail}.", "structural"),
        ]
        for family, clean, holdout in forms:
            yield from paired(clean, family, [mutation(clean, f" {copula} ", f" {wrong} ")], holdout)
        clean = f"The {head} and the {distractor} are {adjective} {tail}."
        yield from paired(clean, "compound-subjects", [mutation(clean, " are ", " is ")], "structural")
        clean = f"The {head} near the {distractor} {past} {adjective} yesterday."
        yield from paired(clean, "past-attraction", [mutation(clean, f" {past} ", f" {wrong_past} ")], "structural")
        clean = f"The {head} {'does' if singular else 'do'} not have a telescope {tail}."
        yield from paired(clean, "negative-auxiliaries", [mutation(clean, " have ", " has ")], "lexical")
        auxiliary = "Does" if singular else "Do"
        clean = f"{auxiliary} the {head} have a telescope {tail}?"
        yield from paired(clean, "inverted-questions", [mutation(clean, auxiliary, "Do" if singular else "Does")], "structural")
        clean = f"The {head} should have a telescope {tail}."
        yield from paired(clean, "modal-auxiliaries", [mutation(clean, " have ", " has ")], "structural")
        clean = f"The {head} {possession} a telescope and received an invitation."
        yield from paired(clean, "multiple-errors", [mutation(clean, f" {possession} ", f" {wrong_possession} "),
                         mutation(clean, "received", "recieved")], "structural")
        clean = f"😀. The {head} {possession} a telescope.\r\nWe received an invitation."
        yield from paired(clean, "unicode-offsets", [mutation(clean, f" {possession} ", f" {wrong_possession} "),
                         mutation(clean, "received", "recieved")], "structural")
        clean = f"The label `teh = 1` remains unchanged while the {head} {possession} a telescope."
        yield from paired(clean, "protected-prose", [mutation(clean, f" {possession} ", f" {wrong_possession} ")], "structural")
    for item, article in [("hour", "an"), ("honest answer", "an"), ("heir", "an"), ("honor", "an"),
                          ("university", "a"), ("useful tool", "a"), ("unicorn", "a"), ("European visitor", "a")]:
        for verb in ["mentioned", "described"]:
            clean = f"We {verb} {article} {item} before the ceremony."
            yield from paired(clean, "article-exceptions", [mutation(clean, f" {article} {item}", f" {'a' if article == 'an' else 'an'} {item}")], "lexical")
    for correct, typo in CHALLENGE_TYPOS:
        for tail in CHALLENGE_TAILS:
            clean = f"We checked the word {correct} {tail}."
            yield from paired(clean, "reserved-spelling", [mutation(clean, correct, typo)], "lexical")
    # One sentence, no full stops in the prefix: actually crosses tokenizer windows.
    for count in range(20, 36):
        prefix = ", ".join(f"item {index}" for index in range(1, count + 1))
        clean = f"After reviewing {prefix}, we received the invitation."
        yield from paired(clean, "window-boundaries", [mutation(clean, "received", "recieved")], "structural")
    for head, _ in heads:
        clean = f"I recommend that the {head} have more time."
        yield example(clean, "valid-constructions", holdout="structural")
    for clean in [
        "If I were you, I would wait.", "She had had enough.",
        "I knew that that was correct.", "I saw her duck.",
        "Does my cat have food?", "Do the owls have food?",
        "The news is accurate.", "These scissors are sharp.",
        "Your advice was helpful.", "You're welcome to stay.",
        "They left their coats there.", "We should have left earlier.",
        "The code is `recieved = teh`.", "The address is https://example.test/recieved.",
        "I read the report yesterday.", "The otters read every evening.",
    ]:
        yield example(clean, "valid-constructions", holdout="structural")


def canonical(text):
    return normalized(text).casefold()


def unique(rows):
    records = {}
    source_targets = {}
    for row in rows:
        source, target = canonical(row["source"]), canonical(row["target"])
        if source in source_targets and source_targets[source] != target:
            raise ValueError("One source has conflicting gold targets")
        source_targets[source] = target
        records.setdefault(row["id"], row)
    return sorted(records.values(), key=lambda row: (row["family"], row["id"]))


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    train = unique(training_examples())
    challenge = unique(challenge_examples())
    train_text = {canonical(row[key]) for row in train for key in ["source", "target"]}
    challenge_text = {canonical(row[key]) for row in challenge for key in ["source", "target"]}
    if train_text & challenge_text:
        raise ValueError("Training/challenge source or target overlap")
    splits = {"train": [], "dev": [], "test": []}
    for row in train:
        bucket = int(row["clean_group"][:8], 16) % 100
        split = "train" if bucket < 80 else "dev" if bucket < 90 else "test"
        words, tags = edit_tags(row["source"], row["target"])
        splits[split].append({**row, "tokens": words, "tags": tags})
    labels = ["KEEP"] + sorted({tag for row in splits["train"] for tag in row["tags"]} - {"KEEP"})
    # Do not hide examples whose labels were unseen in training.
    unsupported = {name: sum(any(tag not in labels for tag in row["tags"]) for row in rows)
                   for name, rows in splits.items()}
    if any(unsupported.values()):
        raise ValueError(f"Train vocabulary does not cover synthetic splits: {unsupported}")
    manifest = {"schema": 1, "origin": VERSION, "license": "CC0-1.0", "publication_allowed": True,
                "generator_sha256": hash_file(Path(__file__)), "label_count": len(labels),
                "evaluation_scope": "Original synthetic templates only; not general grammar accuracy.",
                "split_policy": "SHA256(normalized casefolded clean target), 80/10/10 before corruption",
                "challenge_policy": "Reserved inventories/templates and zero normalized source/target overlap against this new augmentation only; lexical cases share grammar structures; fixed clean controls may use familiar vocabulary. Baseline/external training is not audited.",
                "source_target_overlap": 0, "splits": {}}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][name] = {"rows": len(rows), "clean_rows": sum(row["clean"] for row in rows),
                                    "sha256": hash_file(path), "families": dict(Counter(row["family"] for row in rows))}
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    # Interoperable with prepare_pairs; explicitly generated/unreviewed, never human-reviewed.
    candidates = output / "augmentation-candidates.jsonl"
    candidates.write_text("".join(json.dumps({**row, "category": "clean" if row["clean"] else "mixed",
                                            "review_status": "synthetic-generated"}, ensure_ascii=False) + "\n"
                                  for row in splits["train"]))
    manifest["augmentation"] = {"rows": len(splits["train"]), "sha256": hash_file(candidates),
                                 "review_status": "synthetic-generated", "weak_train_only": True}
    challenge_path = output / "challenge.json"
    for row in challenge:
        _, tags = edit_tags(row["source"], row["target"])
        row["requiredTags"] = sorted(set(tags) - {"KEEP"})
    challenge_path.write_text(json.dumps({"schema": 1, "name": VERSION, "license": "CC0-1.0",
                                         "scope": manifest["evaluation_scope"], "cases": challenge},
                                        indent=2, ensure_ascii=False) + "\n")
    manifest["challenge"] = {"rows": len(challenge), "clean_rows": sum(row["clean"] for row in challenge),
                             "sha256": hash_file(challenge_path), "families": dict(Counter(row["family"] for row in challenge))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/generated/harder-v1"))
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps({"splits": {name: value["rows"] for name, value in manifest["splits"].items()},
                      "challenge": manifest["challenge"], "augmentation": manifest["augmentation"]}, indent=2))
