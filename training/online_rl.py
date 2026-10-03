"""Foreground, resumable generate/screen/matched-train/evaluate loop; never ships weights."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from calibrate_judge import load_fixture
from compare_rl import BASE, REVISION, ROOT, plan as anchor_plan, verify_inputs
from evaluate import qualified
from judge import (JudgeTransportError, canonical, digest, judge_spec, ledger_append,
                   ledger_read, validate_calibration, validate_calibration_training)
from judge_objective import select_rows
from online_data import (atomic_json, filter_pairs, generate, generation_spec, prepare_cycle,
                         rows, screen_pairs, write_rows)
from pairs import evaluation_keys, hash_file, normalized
from rl_objective import checkpoint_hashes, validate_checkpoint_labels


class StopRequested(Exception):
    pass


def plan(args):
    if not args.continuous and args.cycles < 1: raise ValueError("A bounded invocation requires positive cycles")
    if not 1 <= args.epochs <= 8 or not 1 <= args.warmup_epochs <= 32:
        raise ValueError("Bounded continuation/bootstrap epochs required")
    if not 4 <= args.new_pairs <= args.rl_rows <= 256 or args.replay_rows < args.rl_rows or args.replay_rows > 16384:
        raise ValueError("Fresh pairs must fit 4..256 reward rows and a bounded replay pool")
    if args.judge_batch_size not in {8, 16}: raise ValueError("Judge batches must contain 8 or 16 pairs")
    if (args.max_total_requests < 0 or args.max_storage_mb < 64 or args.max_failures < 1
            or not math.isfinite(args.min_gain) or not 0 <= args.min_gain <= 1):
        raise ValueError("Invalid cumulative request, storage, failure or gain limits")
    spec = anchor_plan(args.data, args.evaluation_dir, args.output, args.warmup_epochs, args.epochs,
                       args.batch_size, args.max_length, args.learning_rate, args.seed, args.scorer,
                       args.device, args.rl_coefficient, args.counterexamples, args.max_anchor_rows)
    route = judge_spec()
    fixture, calibration = args.fixture.resolve(), args.calibration.resolve()
    validate_calibration(calibration, route, fixture)
    validate_calibration_training(rows(args.data / "train.jsonl"), fixture)
    for path in (fixture, calibration): spec["input_hashes"][str(path)] = hash_file(path)
    for name in ("online_rl.py", "online_data.py", "judge.py", "judge_objective.py", "calibrate_judge.py", "generate_llm.py"):
        path = ROOT / "training" / name
        spec["code_hashes"][str(path)] = hash_file(path)
    labels = json.loads((args.data / "labels.json").read_text())
    initial = args.initial_checkpoint.resolve() if args.initial_checkpoint else None
    if initial:
        for name, value in validate_checkpoint_labels(initial, labels).items():
            spec["input_hashes"][str(initial / name)] = value
    generator = generation_spec(route, args.new_pairs, args.generation_batch_size, args.seed)
    requests = args.epochs * math.ceil(2 * args.rl_rows / args.judge_batch_size)
    if requests > 256: raise ValueError("Cycle RL request budget exceeds bounded judge limits")
    spec.update({"schema": 1, "kind": "continual-judge-rl-v1", "route": route,
                 "arms": ["supervised", "llm-judge-reinforce"],
                 "continuation_update_upper_bound": args.epochs * math.ceil(
                     (spec["raw_train_rows"] + args.replay_rows) / args.batch_size),
                 "generator": generator, "fixture": str(fixture), "calibration": str(calibration),
                 "judge_batch_size": args.judge_batch_size,
                 "initial_checkpoint": str(initial) if initial else None,
                 "new_pairs": args.new_pairs, "rl_rows": args.rl_rows, "replay_rows": args.replay_rows,
                 "rl_max_requests": requests, "rl_max_pairs": 2 * args.epochs * args.rl_rows,
                 "worst_cycle_requests": math.ceil(args.new_pairs / args.generation_batch_size)
                     + math.ceil(args.new_pairs / args.judge_batch_size) + requests,
                 "max_total_requests": args.max_total_requests, "max_storage_mb": args.max_storage_mb,
                 "max_failures": args.max_failures, "min_gain": args.min_gain,
                 "selection": "Repeated development selection, not independent generalization; test inference deferred.",
                 "publication_allowed": False})
    return spec


def checkpoint_entry(directory, report):
    checkpoint = Path(directory) / "checkpoint"
    return {"checkpoint": str(checkpoint), "files": checkpoint_hashes(checkpoint),
            "quality": development_quality(report)}


def development_quality(report):
    if report.get("evaluation_mode") != "development-only" or report.get("test_status") != "deferred":
        raise ValueError("Online selection requires development-only evaluation")
    scores, research_scores, passed = [], [], True
    for name in ("model.onnx", "model_quantized.onnx"):
        exported = report["exports"][name]
        if exported["split"] != "dev" or exported["inference"]["inference_failures"]:
            raise ValueError("Incomplete development export inference")
        metrics = exported["metrics"]
        passed &= qualified(metrics, .95, .02, 25)
        human = exported["by_origin"]["ErAConD"]
        score = human["edit_f0_5"]
        if type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid human development score")
        scores.append(score)
        research = exported["diagnostic_by_origin"]["ErAConD"]["edit_f0_5"]
        if type(research) not in {int, float} or not math.isfinite(research) or not 0 <= research <= 1:
            raise ValueError("Invalid human research development score")
        research_scores.append(research)
    return {"qualified": bool(passed), "score": min(scores) if passed else 0.,
            "human_fp32_f0_5": scores[0], "human_int8_f0_5": scores[1],
            "research_score": min(research_scores),
            "research_human_fp32_f0_5": research_scores[0], "research_human_int8_f0_5": research_scores[1]}


def choose_candidate(incumbent, candidates, min_gain):
    accepted = [(name, entry) for name, entry in candidates.items()
                if entry["quality"]["qualified"]
                and entry["quality"]["score"] > incumbent["quality"]["score"] + min_gain]
    return max(accepted, key=lambda item: (item[1]["quality"]["score"], item[0] == "supervised")) if accepted else (None, incumbent)


def choose_research(incumbent, candidates, min_gain):
    improved = [(name, entry) for name, entry in candidates.items()
                if entry["quality"]["research_score"] > incumbent["quality"]["research_score"] + min_gain]
    return max(improved, key=lambda item: (item[1]["quality"]["research_score"], item[0] == "supervised")) if improved else (None, incumbent)


def command(spec, data, stage, initial, objective, seed, epochs, pool=None):
    result = [sys.executable, str(ROOT / "training/train.py"), "--data", str(data),
              "--evaluation-dir", spec["evaluation_dir"], "--output", str(stage / "model"),
              "--checkpoint", str(stage / "checkpoint"), "--epochs", str(epochs),
              "--batch-size", str(spec["batch_size"]), "--max-length", str(spec["max_length"]),
              "--learning-rate", str(spec["learning_rate"]), "--seed", str(seed), "--keep-weight", ".3",
              "--base-model", BASE, "--base-revision", REVISION, "--device", spec["device"],
              "--scorer", spec["scorer"], "--local-files-only", "--development-only", "--objective", objective]
    if initial: result.extend(["--initial-checkpoint", str(initial)])
    if objective == "llm-judge-reinforce":
        values = {"--rl-coefficient": spec["rl_coefficient"], "--judge-calibration": spec["calibration"],
                  "--judge-calibration-fixture": spec["fixture"], "--judge-source-pool": pool,
                  "--judge-cache": stage / "judge-cache", "--judge-rl-rows": spec["rl_rows"],
                  "--judge-rl-batch-size": 16, "--judge-batch-size": spec["judge_batch_size"],
                  "--judge-max-requests": spec["rl_max_requests"], "--judge-max-pairs": spec["rl_max_pairs"],
                  "--judge-timeout": 60}
        for key, value in values.items(): result.extend([key, str(value)])
    return result


def train_stage(stage, arguments, stopped, trainer=None):
    if stopped(): raise StopRequested()
    if stage.exists():
        completion = stage / "completion.json"
        if not completion.exists(): raise ValueError("Partial training cannot resume; preserve it and use a new cycle")
        saved = json.loads(completion.read_text())
        if saved["command_sha256"] != digest(canonical(arguments)) or any(
                hash_file(stage / name) != expected for name, expected in saved["files"].items()):
            raise ValueError("Completed training stage changed")
        return json.loads((stage / "model/evaluation.json").read_text())
    stage.mkdir()
    if trainer:
        trainer(arguments)
    else:
        with (stage / "training.log").open("w") as log:
            process = subprocess.Popen(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if stopped():
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    raise StopRequested()
                time.sleep(.5)
            if process.returncode: raise ValueError("Training subprocess failed; private stage log retained")
    report = json.loads((stage / "model/evaluation.json").read_text())
    files = {str(path.relative_to(stage)): hash_file(path) for directory in ("model", "checkpoint")
             for path in sorted((stage / directory).iterdir()) if path.is_file()}
    atomic_json(stage / "completion.json", {"files": files, "command_sha256": digest(canonical(arguments))})
    return report


def bootstrap_stage(output, spec, args, stopped, trainer=None):
    """An interrupted optimizer cannot resume; preserve it and use a numbered attempt."""
    attempts = sorted(output.glob("bootstrap-*"))
    for stage in attempts:
        if (stage / "completion.json").exists():
            report = train_stage(stage, command(spec, args.data, stage, spec["initial_checkpoint"],
                "supervised", args.seed, args.warmup_epochs), stopped, trainer)
            return checkpoint_entry(stage, report)
    if len(attempts) >= spec["max_failures"]:
        raise ValueError("Bootstrap attempt limit reached; partial attempts retained")
    stage = output / f"bootstrap-{len(attempts):04d}"
    report = train_stage(stage, command(spec, args.data, stage, spec["initial_checkpoint"],
        "supervised", args.seed, args.warmup_epochs), stopped, trainer)
    return checkpoint_entry(stage, report)


def requests_used(output):
    return sum(len(ledger_read(path)[0]) for path in output.glob("cycles/cycle-*/**/requests.jsonl"))


def verify_checkpoint(entry):
    if checkpoint_hashes(Path(entry["checkpoint"])) != entry["files"]:
        raise ValueError("Accepted checkpoint changed")


def retain_replay(records, cycle, count, limit):
    result = records + [{"path": str(cycle / "admitted.jsonl"),
                         "sha256": hash_file(cycle / "admitted.jsonl"), "rows": count}]
    while len(result) > 1 and sum(row["rows"] for row in result[1:]) >= limit:
        result.pop(0)
    return result


def training_lock():
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip()
    common = (ROOT / common).resolve()
    lock = (common / "gamma-training.lock").open("a")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise ValueError("Another repository training job holds the shared lock") from None
    return lock


def run(args, transport=None, trainer=None):
    for name in ("data", "evaluation_dir", "calibration", "output", "fixture", "counterexamples", "initial_checkpoint"):
        value = getattr(args, name)
        if value is not None: setattr(args, name, value.resolve())
    spec = plan(args)
    if not args.execute: return {"mode": "dry-run", "plan": spec}
    output = args.output.resolve()
    output.relative_to(ROOT / "artifacts")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock, training_lock() as repository_lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: raise ValueError("This loop is already running") from None
        saved_plan = output / "plan.json"
        if saved_plan.exists() and json.loads(saved_plan.read_text()) != spec:
            raise ValueError("Loop plan changed; choose a new output directory")
        if not saved_plan.exists(): atomic_json(saved_plan, spec)
        events, chain = ledger_read(output / "events.jsonl")
        for event in events:
            if event["kind"] == "cycle" and hash_file(Path(event["result_path"])) != event["result_sha256"]:
                raise ValueError("Completed loop result changed")
            for name, expected in event.get("request_ledgers", {}).items():
                if not Path(name).exists() or hash_file(Path(name)) != expected:
                    raise ValueError("Recorded request ledger changed or disappeared")
        reservations, budget_chain = ledger_read(output / "budget.jsonl")
        reserved_indices = {row["cycle"] for row in reservations}
        reserved = sum(row["requests"] for row in reservations)
        if (len(reserved_indices) != len(reservations)
                or any(row["requests"] != spec["worst_cycle_requests"] for row in reservations)
                or reserved < max((event["state"].get("requests_reserved", 0) for event in events), default=0)
                or requests_used(output) < max((event["state"].get("requests_used", 0) for event in events), default=0)):
            raise ValueError("Lifetime request accounting changed")
        stopping = [False]
        previous_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda *_args: stopping.__setitem__(0, True))
        stopped = lambda: stopping[0] or (output / "STOP").exists()
        def before_request():
            if stopped(): raise StopRequested()
        try:
            if events:
                state = events[-1]["state"]
            else:
                state = {"next_cycle": 0, "failures": 0,
                         "accepted": bootstrap_stage(output, spec, args, stopped, trainer), "replay": []}
                state["training"] = state["accepted"]
                chain = ledger_append(output / "events.jsonl", {"kind": "bootstrap", "state": state}, chain)
            atomic_json(output / "state.json", state)
            verify_checkpoint(state["accepted"])
            verify_checkpoint(state["training"])
            tokenizer = None
            if trainer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(state["training"]["checkpoint"], use_fast=True, local_files_only=True)
            excluded = evaluation_keys(rows(args.evaluation_dir / "dev.jsonl") + rows(args.evaluation_dir / "test.jsonl"))
            _, cases = load_fixture(args.fixture)
            excluded.update(normalized(row[field]).casefold() for row in cases for field in ("source", "candidate"))
            if args.counterexamples: excluded |= evaluation_keys(json.loads(args.counterexamples.read_text())["cases"])
            seen = {normalized(row["source"]).casefold() for row in rows(args.data / "train.jsonl")}
            for path in output.glob("cycles/cycle-*/generation/candidates.jsonl"):
                seen.update(normalized(row["source"]).casefold() for row in rows(path))
            completed = 0
            stop_reason = "cycle-limit"
            (output / "cycles").mkdir(exist_ok=True)
            while not stopped() and (args.continuous or completed < args.cycles):
                verify_inputs(spec)
                verify_checkpoint(state["accepted"])
                verify_checkpoint(state["training"])
                replay = []
                for record in state["replay"]:
                    if hash_file(Path(record["path"])) != record["sha256"]: raise ValueError("Replay data changed")
                    replay.extend(rows(record["path"]))
                used = requests_used(output)
                storage = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                if storage > spec["max_storage_mb"] * 1024 ** 2 or shutil.disk_usage(output).free < 1024 ** 3:
                    stop_reason = "storage-budget"
                    break
                index = state["next_cycle"]
                cycle = output / "cycles" / f"cycle-{index:06d}"
                next_state = {**state, "next_cycle": index + 1}
                if cycle.exists() or index in reserved_indices:
                    if index not in reserved_indices: raise ValueError("Cycle request reservation disappeared")
                    cycle.mkdir(exist_ok=True)
                    next_state["failures"] = state["failures"] + 1
                    result = {"status": "abandoned-partial", "cycle": index, "test_inference": False}
                else:
                    if spec["max_total_requests"] and reserved + spec["worst_cycle_requests"] > spec["max_total_requests"]:
                        stop_reason = "request-budget"
                        break
                    budget_chain = ledger_append(output / "budget.jsonl",
                        {"cycle": index, "requests": spec["worst_cycle_requests"]}, budget_chain)
                    reserved += spec["worst_cycle_requests"]
                    reserved_indices.add(index)
                    cycle.mkdir()
                    try:
                        fresh = []
                        phase = "generation"
                        generator = {**spec["generator"], "seed": args.seed + index}
                        candidates = generate(cycle / "generation", generator, transport, before_request)
                        filtered, rejected = filter_pairs(candidates, set(json.loads((args.data / "labels.json").read_text())),
                            json.loads((args.data / "manifest.json").read_text()).get("edit_schema", 1), excluded, seen,
                            tokenizer, spec["max_length"])
                        seen.update(normalized(row["source"]).casefold() for row in candidates)
                        phase = "screening"
                        fresh, screening = screen_pairs(cycle / "screening", filtered, spec["route"], transport,
                                                        before_request, spec["judge_batch_size"])
                        write_rows(cycle / "admitted.jsonl", fresh)
                        if not fresh: raise ValueError("No representable screened fresh pairs")
                        next_state["replay"] = retain_replay(state["replay"], cycle, len(fresh), spec["replay_rows"])
                        phase = "preparation"
                        replay = (replay + fresh)[-spec["replay_rows"]:]
                        data = cycle / "prepared"
                        prepare_cycle(args.data, data, replay)
                        fresh_keys = {row["source"] for row in fresh}
                        old = [row for row in replay if row["source"] not in fresh_keys]
                        pool = fresh + (select_rows(old, min(len(old), spec["rl_rows"] - len(fresh)), args.seed + index)
                                        if old and len(fresh) < spec["rl_rows"] else [])
                        write_rows(data / "judge-sources.jsonl", pool)
                        initial = Path(state["training"]["checkpoint"])
                        reports, entries = {}, {}
                        for name, objective in (("supervised", "supervised"), ("judge-rl", "llm-judge-reinforce")):
                            phase = name
                            if stopped(): raise StopRequested()
                            stage = cycle / name
                            reports[name] = train_stage(stage, command(spec, data, stage, initial, objective,
                                args.seed + index, args.epochs, data / "judge-sources.jsonl"), stopped, trainer)
                            entries[name] = checkpoint_entry(stage, reports[name])
                        left, right = reports["supervised"], reports["judge-rl"]
                        phase = "selection"
                        if (any(left["training_budget"][key] != right["training_budget"][key]
                                for key in ("optimizer_updates", "examples_seen", "rows_per_epoch", "optimizer_state"))
                                or left["initialization"]["files"] != right["initialization"]["files"]
                                or left["training_data_sha256"] != right["training_data_sha256"]):
                            raise ValueError("Matched continuations differ in initialization, data or update budget")
                        for report in reports.values():
                            if (report["training_data_sha256"] != hash_file(data / "train.jsonl")
                                    or report["labels_sha256"] != hash_file(args.data / "labels.json")
                                    or report["evaluation_hashes"] != {split: hash_file(args.evaluation_dir / f"{split}.jsonl")
                                                                        for split in ("dev", "test")}):
                                raise ValueError("Continuation data, label or evaluation identity differs")
                        verify_inputs(spec)
                        chosen, accepted = choose_candidate(state["accepted"], entries, args.min_gain)
                        research_arm, research = choose_research(state["training"], entries, args.min_gain)
                        next_state.update({"accepted": accepted, "training": research,
                                           "failures": 0})
                        result = {"status": "completed", "cycle": index, "generated": len(candidates),
                                  "fresh_admitted": len(fresh), "replay_rows": len(replay), "rejections": rejected,
                                  "screening": screening, "arms": {name: entry["quality"] for name, entry in entries.items()},
                                  "accepted_arm": chosen, "test_inference": False,
                                  "training_arm": research_arm,
                                  "rl_minus_supervised_human_dev_f0_5": entries["judge-rl"]["quality"]["human_fp32_f0_5"]
                                      - entries["supervised"]["quality"]["human_fp32_f0_5"],
                                  "rl_minus_supervised_research_human_dev_f0_5": entries["judge-rl"]["quality"]["research_human_fp32_f0_5"]
                                      - entries["supervised"]["quality"]["research_human_fp32_f0_5"]}
                    except StopRequested:
                        raise
                    except Exception as error:
                        next_state["failures"] = state["failures"] + 1
                        result = {"status": "failed", "cycle": index, "error_type": type(error).__name__,
                                  "phase": phase, "reason": str(error) if type(error) in {ValueError, JudgeTransportError}
                                      else type(error).__name__,
                                  "fresh_admitted": len(fresh),
                                  "last_accepted_retained": True, "test_inference": False}
                result_path = cycle / ("abandonment.json" if result["status"] == "abandoned-partial" else "result.json")
                atomic_json(result_path, result)
                next_state["requests_used"] = requests_used(output)
                next_state["requests_reserved"] = reserved
                chain = ledger_append(output / "events.jsonl", {"kind": "cycle", "result_path": str(result_path),
                                      "result_sha256": hash_file(result_path),
                                      "request_ledgers": {str(path): hash_file(path)
                                                          for path in sorted(cycle.glob("**/requests.jsonl"))},
                                      "state": next_state}, chain)
                state = next_state
                atomic_json(output / "state.json", state)
                print(canonical(result), flush=True)
                completed += 1
                if state["failures"] >= spec["max_failures"]:
                    stop_reason = "failure-limit"
                    break
            return {"status": "stopped" if stopped() else "bounded-chunk-complete", "state": state,
                    "stop_reason": "requested" if stopped() else stop_reason,
                    "publication_allowed": False, "test_inference": False}
        except StopRequested:
            return {"status": "interrupted-partial-retained", "publication_allowed": False, "test_inference": False}
        finally:
            for signum, handler in previous_handlers.items(): signal.signal(signum, handler)


def parser():
    result = argparse.ArgumentParser(__doc__)
    for name in ("data", "evaluation-dir", "calibration", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--initial-checkpoint", type=Path)
    result.add_argument("--fixture", type=Path, default=ROOT / "data/judge-calibration.json")
    result.add_argument("--counterexamples", type=Path, default=ROOT / "data/counterexamples.json")
    for name, default in (("new-pairs", 64), ("generation-batch-size", 4), ("judge-batch-size", 8), ("rl-rows", 128), ("replay-rows", 2048),
                          ("epochs", 2), ("warmup-epochs", 2), ("batch-size", 64), ("max-length", 96),
                          ("seed", 42), ("max-anchor-rows", 50000), ("cycles", 1), ("max-total-requests", 0),
                          ("max-storage-mb", 2048), ("max-failures", 3)):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--learning-rate", type=float, default=.0002)
    result.add_argument("--rl-coefficient", type=float, default=.1)
    result.add_argument("--min-gain", type=float, default=.001)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--scorer", choices=("errant", "approximate"), default="errant")
    result.add_argument("--continuous", action="store_true", help="Keep bounded cycles running in the foreground until stopped")
    result.add_argument("--execute", action="store_true")
    return result


if __name__ == "__main__":
    print(canonical(run(parser().parse_args())), flush=True)
