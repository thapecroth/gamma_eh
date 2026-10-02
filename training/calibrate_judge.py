"""Two independent judge calibration rounds on agent-authored diagnostics, never human gold."""
import argparse
import json
from pathlib import Path

from judge import FLAGS, Judge, flags_reward, judge_spec, spec_hash
from pairs import hash_file


def load_fixture(path):
    value = json.loads(Path(path).read_text())
    if (value.get("schema") != 1 or value.get("license") != "CC0-1.0"
            or value.get("origin") != "agent-authored-judge-calibration-v1"
            or value.get("review_status") != "unreviewed" or value.get("selection_use") is not False):
        raise ValueError("Judge calibration must remain unreviewed diagnostic data")
    rows = value.get("cases")
    if not isinstance(rows, list) or not rows or len(rows) > 32 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Invalid bounded calibration cases/IDs")
    for row in rows:
        if not all(isinstance(row.get(name), str) and 1 <= len(row[name]) <= 2000 for name in ("source", "candidate")):
            raise ValueError("Invalid calibration text")
        flags_reward(row["expected"])
    return value, rows


def measure(rows, rounds):
    if len(rounds) != 2 or any(len(values) != len(rows) for values in rounds):
        raise ValueError("Calibration requires two complete independent rounds")
    agreement, by_flag, unsafe, critical, failing = [], [], 0, 0, set()
    for values in rounds:
        correct = 0
        for row, verdict in zip(rows, values):
            flags = verdict["flags"]
            reward = flags_reward(flags)
            matches = sum(flags[name] == row["expected"][name] for name in FLAGS)
            correct += matches
            if matches != 5: failing.add(row["id"])
            expected_unsafe = not row["expected"]["meaning_preserved"] or not row["expected"]["protected_text_preserved"]
            unsafe += expected_unsafe and reward > 0
            critical += sum(not row["expected"][name] and flags[name] for name in
                            ("meaning_preserved", "protected_text_preserved"))
        agreement.append(correct / (5 * len(rows)))
        by_flag.append({name: sum(verdict["flags"][name] == row["expected"][name]
                                for row, verdict in zip(rows, values)) / len(rows) for name in FLAGS})
    repeat = sum(a["flags"][name] == b["flags"][name] for a, b in zip(*rounds) for name in FLAGS) / (5 * len(rows))
    return {"passed": min(agreement) >= .95 and repeat >= .98 and unsafe == 0 and critical == 0,
            "criteria_agreement": agreement, "repeat_agreement": repeat, "unsafe_high_reward_count": unsafe,
            "criteria_agreement_by_flag": by_flag,
            "repeat_agreement_by_flag": {name: sum(a["flags"][name] == b["flags"][name] for a, b in zip(*rounds)) / len(rows)
                                         for name in FLAGS},
            "critical_false_positive_flags": critical, "failing_case_ids": sorted(failing),
            "criteria_checks_per_round": 5 * len(rows)}


def run(fixture, output, spec=None, execute=False, batch_size=32, timeout=60):
    fixture, output = Path(fixture).resolve(), Path(output).resolve()
    value, rows = load_fixture(fixture)
    spec = spec or judge_spec()
    planned = {"schema": 1, "judge_spec_sha256": spec_hash(spec), "judge_spec": spec,
               "code_sha256": {name: hash_file(Path(__file__).with_name(name)) for name in ("judge.py", "calibrate_judge.py")},
               "fixture_sha256": hash_file(fixture), "cases": len(rows), "batch_size": batch_size, "timeout": timeout,
               "rounds": 2, "scope": value["scope"], "review_status": value["review_status"],
               "selection_use": False, "criteria": {"min_round_agreement": .95, "min_repeat_agreement": .98,
                   "max_unsafe_high_rewards": 0, "max_critical_false_positive_flags": 0}}
    if not execute: return {"mode": "dry-run", **planned}
    if output.exists(): raise ValueError("Calibration requires a fresh directory and independent round caches")
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(planned, indent=2) + "\n")
    rounds, receipts = [], []
    for index in range(2):
        with Judge(output / f"round-{index + 1}", spec, batch_size=batch_size, timeout=timeout,
                   max_pairs=32, max_requests=4) as judge:
            rounds.append(judge.judge([(row["source"], row["candidate"]) for row in rows]))
            receipts.append({"stats": judge.stats(), "ledgers": {
                name: hash_file(judge.directory / name) for name in ("spec.json", "requests.jsonl", "verdicts.jsonl")}})
    if hash_file(fixture) != planned["fixture_sha256"]:
        raise ValueError("Calibration fixture changed during evaluation")
    result = {**planned, **measure(rows, rounds), "round_receipts": receipts}
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("data/judge-calibration.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/judge-calibration"))
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--auth", default="subscription")
    parser.add_argument("--provider-header", default="X-CLIProxy-Provider")
    parser.add_argument("--auth-header", default="X-CLIProxy-Auth-Mode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = judge_spec(args.model, args.provider, args.auth, args.provider_header, args.auth_header)
    print(json.dumps(run(args.fixture, args.output, spec, args.execute, args.batch_size, args.timeout), indent=2))
