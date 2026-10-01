import json

import pytest

from pairs import hash_file
from report_tuning import score_receipt, validate_browser


def test_browser_validation_rejects_truncation_and_reordering():
    rows = [{"id": "one", "source": "One."}, {"id": "two", "source": "Two."}]
    report = {"passed": True, "input_sha256": "hash", "rows": rows.copy()}
    validate_browser(report, rows, "hash")
    report["rows"] = rows[:1]
    with pytest.raises(ValueError, match="incomplete"):
        validate_browser(report, rows, "hash")
    report["rows"] = rows[::-1]
    with pytest.raises(ValueError, match="order"):
        validate_browser(report, rows, "hash")


def test_score_receipt_binds_browser_population_and_model(tmp_path):
    population = tmp_path / "dev.jsonl"
    row = {"id": "one", "source": "One.", "references": ["One."]}
    population.write_text(json.dumps(row) + '\n')
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.onnx").write_text('weights')
    identity = {"model_sha256": hash_file(model / "model.onnx")}
    browser = {"passed": True, "rows": [row], "input_sha256": hash_file(population),
               "model": identity, "engine_source_sha256": "engine"}
    browser_path = tmp_path / "dev-browser.json"
    browser_path.write_text(json.dumps(browser))
    receipt = {"input_sha256": hash_file(population), "browser_report_sha256": hash_file(browser_path),
               "model": identity, "engine_source_sha256": "engine", "scores": {
                   "rules": {"sentences": 1}, "model_only": [{"sentences": 1}], "full_engine": [{"sentences": 1}]}}
    path = tmp_path / "dev-score.json"
    path.write_text(json.dumps(receipt))
    assert score_receipt(path, population, model, "engine")["model"] == identity
    (model / "model.onnx").write_text('other weights')
    with pytest.raises(ValueError, match="model/engine"):
        score_receipt(path, population, model, "engine")
    (model / "model.onnx").write_text('weights')
    population.write_text(json.dumps({**row, "source": "Other."}) + '\n')
    with pytest.raises(ValueError, match="population"):
        score_receipt(path, population, model, "engine")
