"""Pure development-only qualification; an empty safe set never enables edits."""
import math


def choose_threshold(candidates, no_edit, min_precision=.95, max_clean_fp=.02, min_edits=25):
    if not candidates or not 0 <= min_precision <= 1 or not 0 <= max_clean_fp <= 1 or min_edits < 1:
        raise ValueError("Invalid calibration population or constraints")
    eligible = [row for row in candidates
                if all(math.isfinite(row[key]) for key in ["edit_precision", "edit_f0_5", "clean_sentence_false_positive_rate"])
                and row["true_positive_edits"] + row["false_positive_edits"] >= min_edits
                and row["edit_precision"] >= min_precision
                and row["clean_sentence_false_positive_rate"] <= max_clean_fp]
    return {"candidates": candidates, "selected": max(eligible, key=lambda row: (row["edit_f0_5"], row["threshold"])) if eligible else no_edit,
            "split": "dev", "constraints_met": bool(eligible), "disable_model_edits": not bool(eligible),
            "constraints": {"min_precision": min_precision, "max_clean_false_positive_rate": max_clean_fp, "min_predicted_edits": min_edits},
            "fallback_reason": None if eligible else "No development threshold meets precision, clean-text and edit-support requirements."}
