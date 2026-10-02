"""Exercise the trainer control flow without downloads, CUDA or ONNX compilation.

The optional training dependencies are present in the training environment;
stdlib-only CI continues to cover dataset admission separately.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("development_only", [True, False])
@pytest.mark.parametrize("objective", ["supervised", "anchored-reinforce"])
def test_test_population_is_never_inferred_in_development_only_mode(tmp_path, monkeypatch, development_only, objective):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("transformers")
    import train

    data = tmp_path / "data"
    evaluation = tmp_path / "evaluation"
    data.mkdir()
    evaluation.mkdir()
    (data / "labels.json").write_text(json.dumps(["KEEP", "REPLACE:is"]))
    (data / "manifest.json").write_text(json.dumps({"edit_schema": 1}))
    train_row = {"source": "She are ready.", "target": "She is ready."}
    (data / "train.jsonl").write_text(json.dumps(train_row) + "\n")
    for split, text in [("dev", "We are ready."), ("test", "Heldout sentence stays untouched.")]:
        (data / f"{split}.jsonl").write_text("\n")
        (evaluation / f"{split}.jsonl").write_text(json.dumps({"source": text, "references": [text]}) + "\n")

    class Tokenizer:
        do_lower_case = True
        backend_tokenizer = SimpleNamespace(normalizer=SimpleNamespace(__getstate__=lambda: json.dumps({
            "type": "BertNormalizer", "lowercase": True, "handle_chinese_chars": True,
            "clean_text": True, "strip_accents": None}).encode()))

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]]),
                    "token_type_ids": torch.tensor([[0, 0]])}

        def save_pretrained(self, path):
            (Path(path) / "vocab.txt").write_text("[PAD]\n[UNK]\n")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([2., -2.]))
            self.config = SimpleNamespace(model_type="bert", max_position_embeddings=512,
                save_pretrained=lambda path: (Path(path) / "config.json").write_text("{}"))

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=self.bias.expand(*input_ids.shape, 2))

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            self.config.save_pretrained(path)
            torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")

    loads, inferred = [], []

    def load(path, *_args, **_kwargs):
        split = path.stem
        loads.append(split)
        if development_only: assert split != "test"
        values = Tokenizer()()
        tensor = torch.utils.data.TensorDataset(*values.values(), torch.tensor([[0, 0]]))
        return tensor, {"population": 1, "accepted": 1}

    def collect(rows, *_args, **_kwargs):
        split = "test" if rows[0]["source"].startswith("Heldout") else "dev"
        inferred.append(split)
        if development_only: assert split != "test"
        return [{"proposals": [], "failed": False} for _ in rows], {"inference_failures": 0}

    class Session:
        def __init__(self, *_args, **_kwargs): pass
        def run(self, _outputs, feed):
            ids = torch.from_numpy(feed["input_ids"])
            return [Model()(ids).logits.detach().numpy()]

    monkeypatch.setattr(train, "BertTokenizerFast", Tokenizer)
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    monkeypatch.setattr(train.AutoModelForTokenClassification, "from_pretrained", lambda *_args, **_kwargs: Model())
    monkeypatch.setattr(train, "load_split", load)
    monkeypatch.setattr(train, "collect_proposals", collect)
    monkeypatch.setattr(train.ort, "InferenceSession", Session)
    monkeypatch.setattr(train.torch.onnx, "export", lambda _model, _dummy, path, **_kwargs: Path(path).write_bytes(b"fake graph"))
    monkeypatch.setattr(train.onnx.checker, "check_model", lambda _path: None)
    monkeypatch.setattr(train, "quantize_dynamic", lambda _source, target, **_kwargs: Path(target).write_bytes(b"fake quantized graph"))
    args = train.parser().parse_args(["--data", str(data), "--evaluation-dir", str(evaluation),
        "--output", str(tmp_path / "output"), "--checkpoint", str(tmp_path / "checkpoint"),
        "--epochs", "1", "--device", "cpu", "--keep-weight", "1.0", "--objective", objective,
        *(["--development-only"] if development_only else [])])
    train.main(args)
    report = json.loads((args.output / "evaluation.json").read_text())
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert report["schedule"]["keep_weight"] == 1.
    assert report["objective"]["keep_weight"] == 1.
    assert report["objective"]["name"] == objective
    assert report["test_status"] == ("deferred" if development_only else "evaluated")
    assert {value["split"] for value in report["exports"].values()} == {"dev" if development_only else "test"}
    if development_only:
        assert "test" not in inferred and "test" not in loads
        assert report["test"] is report["diagnostic_unconstrained_test"] is None
        assert report["pytorch_checkpoint_test"] is report["inference"]["test"] is None
        assert "test_by_origin" not in report
        assert {value["by_origin"]["unspecified"]["sentences"] for value in report["exports"].values()} == {1}
        assert manifest["disableModelEdits"] is True
    else:
        assert "test" in inferred and "test" in loads and report["test"] is not None
        assert report["test_by_origin"]["unspecified"]["sentences"] == 1
