"""Run a small immutable KEEP-weight/learning-rate comparison, sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from pairs import hash_file

PROFILES = [
    {"name": "keep03-lr5e4", "keep_weight": .3, "learning_rate": .0005},
    {"name": "keep10-lr5e4", "keep_weight": 1., "learning_rate": .0005},
    {"name": "keep03-lr2e4", "keep_weight": .3, "learning_rate": .0002},
    {"name": "keep10-lr2e4", "keep_weight": 1., "learning_rate": .0002},
]


def plan(data, epochs=8, device="cuda"):
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("license") != "CC0-1.0" or manifest.get("publication_allowed") is not True:
        raise ValueError("This comparison requires the original publishable CC0 mixture")
    for split in ["train", "dev", "test"]:
        if hash_file(data / f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError("Frozen dataset hash mismatch")
    return {"schema": 2, "dataset_manifest_sha256": hash_file(data / "manifest.json"),
            "labels_sha256": hash_file(data / "labels.json"),
            "trainer_sha256": hash_file(Path(__file__).with_name("train.py")),
            "calibration_sha256": hash_file(Path(__file__).with_name("calibration.py")),
            "runner_sha256": hash_file(Path(__file__)), "device": device,
            "profiles": PROFILES, "epochs": epochs, "batch_size": 128,
            "max_length": 64, "seed": 42,
            "selection": "Development only; frozen challenge and natural test are confirmation audits after selection."}


def completed(model, profile, spec):
    if not (model / "evaluation.json").exists() or not (model / "manifest.json").exists():
        return False
    manifest = json.loads((model / "manifest.json").read_text())
    receipt = json.loads((model / "evaluation.json").read_text())
    expected = {"learning_rate": profile["learning_rate"], "keep_weight": profile["keep_weight"],
                "batch_size": spec["batch_size"], "max_length": spec["max_length"], "epochs": spec["epochs"]}
    if receipt.get("training_config") != expected or receipt.get("device") != spec["device"] or receipt.get("seed") != spec["seed"]:
        raise ValueError("Completed receipt does not match the frozen profile")
    required = {"model.onnx", "labels.json", "vocab.txt", "dataset-manifest.json"}
    if not required <= manifest.get("files", {}).keys():
        raise ValueError("Incomplete exported model receipt")
    for name, asset in manifest["files"].items():
        if Path(name).name != name or hash_file(model / name) != asset["sha256"]:
            raise ValueError("Exported model asset mismatch")
    if hash_file(model / "dataset-manifest.json") != spec["dataset_manifest_sha256"] or hash_file(model / "labels.json") != spec["labels_sha256"]:
        raise ValueError("Exported training population mismatch")
    return True


def run(data, output, epochs=8, device="cuda", execute=False):
    spec = plan(data, epochs, device)
    if not execute:
        return spec
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != spec:
        raise ValueError("Tuning settings or frozen inputs changed; choose a new output directory")
    plan_path.write_text(json.dumps(spec, indent=2) + "\n")
    trainer = Path(__file__).with_name("train.py")
    for profile in PROFILES:
        model, checkpoint = output / profile["name"] / "model", output / profile["name"] / "checkpoint"
        if completed(model, profile, spec):
            continue
        print(json.dumps({"profile": profile["name"], "status": "training"}), flush=True)
        subprocess.run([sys.executable, str(trainer), "--data", str(data), "--output", str(model),
                        "--checkpoint", str(checkpoint), "--epochs", str(epochs), "--batch-size", "128",
                        "--max-length", "64", "--seed", "42", "--local-files-only", "--device", device,
                        "--learning-rate", str(profile["learning_rate"]), "--keep-weight", str(profile["keep_weight"]),
                        "--name", "gamma-eh-tiny-" + profile["name"]], check=True)
        print(json.dumps({"profile": profile["name"], "status": "exported"}), flush=True)
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tuning-v1"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.data.resolve(), args.output.resolve(), args.epochs, args.device, args.execute), indent=2))
