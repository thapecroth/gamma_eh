from data import build, edit_tags, render, tokens
import json
import pytest


def apply(words, tags):
    result = []
    for word, tag in zip(words, tags):
        if tag == "KEEP":
            result.append(word)
        elif tag == "DELETE":
            continue
        elif tag.startswith("REPLACE:"):
            value = tag.split(":", 1)[1]
            result.append(value.capitalize() if word[0].isupper() else value)
        elif tag.startswith("APPEND:"):
            result.extend([word, tag.split(":", 1)[1]])
    return render(result)


def test_replacement_deletion_insertion():
    for source, target in [("She have a book.", "She has a book."),
                           ("I have have a book.", "I have a book."),
                           ("I have book.", "I have a book.")]:
        words, tags = edit_tags(source, target)
        assert apply(words, tags) == target


def test_unsupported_pair_rejected():
    with pytest.raises(ValueError):
        edit_tags("She reads.", "The careful teacher reads.")


def test_generated_corpus_is_reversible_and_disjoint(tmp_path):
    build(tmp_path)
    seen_groups = set()
    seen_sources = set()
    for split in ["train", "dev", "test"]:
        rows = [json.loads(line) for line in (tmp_path / f"{split}.jsonl").read_text().splitlines()]
        groups = {r["clean_group"] for r in rows}
        sources = {r["source"] for r in rows}
        assert not seen_groups & groups
        assert not seen_sources & sources
        seen_groups |= groups
        seen_sources |= sources
        for row in rows:
            assert tokens(row["source"]) == row["tokens"]
            assert apply(row["tokens"], row["tags"]) == row["target"]
