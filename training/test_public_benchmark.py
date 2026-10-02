import copy
import json

import pytest

from public_benchmark import clean_counts, digest, m2_rows, parse_errant, validate_report


def test_multireference_m2_preserves_noop_insertion_and_deletion():
    raw = ("S She have book .\n"
           "A 1 2|||R:VERB:SVA|||has|||REQUIRED|||-NONE-|||0\n"
           "A 2 2|||M:DET|||a|||REQUIRED|||-NONE-|||0\n"
           "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1\n\n"
           "S The the cat .\n"
           "A 1 2|||R:OTHER|||-NONE-|||REQUIRED|||-NONE-|||0\n"
           "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1\n\n").encode()
    rows = m2_rows(raw)
    assert rows[0]["references"] == ["She has a book .", "She have book ."]
    assert rows[1]["references"] == ["The cat .", "The the cat ."]
    assert clean_counts(rows, ["She has a book .", "The cat ."])["reference_accepted_sources_changed"] == 2


def test_gold_insertion_and_replacement_can_share_a_start_offset():
    raw = ("S She have book .\n"
           "A 1 1|||M:ADV|||always|||REQUIRED|||-NONE-|||0\n"
           "A 1 2|||R:VERB:SVA|||has|||REQUIRED|||-NONE-|||0\n"
           "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1\n\n").encode()
    assert m2_rows(raw)[0]["references"][0] == "She always has book ."


def fixture(tmp_path):
    rows = [{"id": "case-0", "source": "The cat is ready .", "references": ["The cat is ready ."]}]
    input_path = tmp_path / "test.jsonl"
    raw = (json.dumps(rows[0]) + "\n").encode()
    input_path.write_bytes(raw)
    prediction = {**rows[0], "index": 0, "sourceSha256": digest(rows[0]["source"].encode()),
                  "corrected": rows[0]["source"], "executed": True, "failed": False, "failure": None}
    report = {"passed": True, "inputSha256": digest(raw), "network": {"blockedRequests": 0},
              "code": {"engineSourcesSha256": "hash"}, "engineBundleSha256": "bundle",
              "model": {"policy": {"publicationAllowed": True}}, "input": {"totalRows": 1, "selectedRows": 1},
              "runs": [{"mode": mode, "maxPasses": 1, "predictions": [copy.deepcopy(prediction)]} for mode in ("rules", "model", "combined")]}
    return input_path, tmp_path / "report.json", report


@pytest.mark.parametrize("damage", ["failed", "skipped", "partial", "reference", "order", "mode", "private", "network", "input", "multiline"])
def test_benchmark_fails_closed_on_invalid_evidence(tmp_path, damage):
    input_path, report_path, report = fixture(tmp_path)
    prediction = report["runs"][0]["predictions"][0]
    if damage == "failed": prediction["failed"] = True
    elif damage == "skipped": prediction["executed"] = False
    elif damage == "partial": report["input"]["totalRows"] = 2
    elif damage == "reference": prediction["references"] = ["An altered gold answer ."]
    elif damage == "order": prediction["index"] = 1
    elif damage == "mode": report["runs"][1]["mode"] = "rules"
    elif damage == "private": report["model"]["policy"]["publicationAllowed"] = False
    elif damage == "network": report["network"]["blockedRequests"] = 1
    elif damage == "input": report["inputSha256"] = "incorrect"
    elif damage == "multiline": prediction["corrected"] = "One .\nTwo ."
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_report(input_path, report_path)


def test_complete_population_and_noop_are_accepted(tmp_path):
    input_path, report_path, report = fixture(tmp_path)
    report_path.write_text(json.dumps(report))
    rows, outputs, _ = validate_report(input_path, report_path)
    assert set(outputs) == {"unchanged", "rules", "model", "combined"}
    assert clean_counts(rows, outputs["unchanged"])["reference_accepted_change_rate"] == 0


def test_standard_errant_counts_keep_precision_beyond_cli_rounding():
    score = parse_errant("TP\tFP\tFN\tPrec\tRec\tF0.5\n7\t1\t1499\t0.875\t0.0046\t0.023\n")
    assert score["precision"] == .875
    assert score["recall"] == 7 / 1506
    assert score["f0_5"] == 8.75 / (8.75 + 1 + .25 * 1499)
    with pytest.raises(ValueError):
        parse_errant("No successful evaluation result")
