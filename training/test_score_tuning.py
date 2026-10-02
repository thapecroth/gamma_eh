import pytest

from score_tuning import score, normalized


def test_clean_identity_preserves_case_and_unicode_normalization():
    assert normalized("Cafe\u0301\ttext") == normalized("Café text")
    assert normalized("I am here.") != normalized("i am here.")


def test_multireference_counts_errors_and_clean_changes():
    rows = [{"source": "bad", "references": ["good", "better"]}, {"source": "clean", "references": ["clean"]}]
    def edits(source, target):
        return frozenset() if source == target else frozenset({(source, target)})
    result = score(rows, ["better", "damaged"], edits)
    assert result["true_positive_edits"] == 1
    assert result["false_positive_edits"] == 1
    assert result["clean_sentences"] == result["clean_sentences_changed"] == 1
    assert result["edit_precision"] == .5
    with pytest.raises(ValueError, match="population mismatch"):
        score(rows, ["better"], edits)
