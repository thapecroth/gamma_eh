import math

import pytest

from calibration import choose_threshold


def metric(threshold=.8, precision=.99, fp=0., edits=100, f=.8):
    return {"threshold": threshold, "edit_precision": precision, "clean_sentence_false_positive_rate": fp,
            "true_positive_edits": edits, "false_positive_edits": 0, "edit_f0_5": f}


def test_failed_constraints_explicitly_disable_instead_of_exporting_best_unsafe():
    unsafe = [metric(precision=.5, f=.99), metric(fp=.1, f=.98), metric(edits=1, f=1.)]
    no_edit = metric(threshold=1., edits=0, f=0.)
    result = choose_threshold(unsafe, no_edit)
    assert result["constraints_met"] is False
    assert result["disable_model_edits"] is True
    assert result["selected"] == no_edit
    assert result["fallback_reason"]


def test_qualification_selects_best_safe_development_threshold():
    result = choose_threshold([metric(f=.5), metric(threshold=.9, f=.9), metric(precision=.9, f=1.)], metric(edits=0))
    assert result["constraints_met"] is True
    assert result["selected"]["threshold"] == .9


def test_zero_support_and_nonfinite_precision_never_qualify():
    result = choose_threshold([metric(edits=0), metric(precision=math.nan)], metric(edits=0))
    assert result["disable_model_edits"] is True
    with pytest.raises(ValueError):
        choose_threshold([], metric())
    with pytest.raises(ValueError):
        choose_threshold([metric()], metric(), min_edits=0)
