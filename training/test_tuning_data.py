import json

from pairs import hash_file
from tuning_data import build, key, read_rows


def test_mixture_is_disjoint_excludes_evaluation_and_removes_ambiguous_labels(tmp_path):
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    for split in ["dev", "test"]:
        (evaluation / f"{split}.jsonl").write_text(json.dumps({"source": "They left their baskets near the studio.",
            "references": ["Your camera is outside the museum."]}) + '\n')
    output = tmp_path / "mixture"
    manifest = build(output, evaluation)
    excluded = {key(value) for value in ["They left their baskets near the studio.", "Your camera is outside the museum."]}
    challenge = json.loads((output / "extra/challenge.json").read_text())["cases"]
    excluded |= {key(row[field]) for row in challenge for field in ["source", "target"]}
    seen = set()
    labels = json.loads((output / "labels.json").read_text())
    for split in ["train", "dev", "test"]:
        path = output / f"{split}.jsonl"
        rows = read_rows(path)
        population = {key(row[field]) for row in rows for field in ["source", "target"]}
        assert not population & (excluded | seen)
        seen |= population
        assert hash_file(path) == manifest["splits"][split]["sha256"]
        assert all(tag in labels for row in rows for tag in row["tags"])
        assert not any(word.lower() == "read" and tag == "REPLACE:reads" for row in rows for word, tag in zip(row["tokens"], row["tags"]))
    assert manifest["counts"]["ambiguous_read_removed"] > 0
