"""Bounded root-license checks at original commits using the authenticated gh CLI."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from corpus_import import dolt_rows, load_spec


def audit(spec, output, max_scanned=500, max_commits=100):
    if not 1 <= max_commits <= 1000 or not 1 <= max_scanned <= 100000:
        raise ValueError("Bounded license audit required")
    if output.exists():
        raise ValueError("License cache exists; retain it and choose a fresh path")
    proofs, counts = {}, {"checked": 0, "verified": 0, "unverified": 0}
    for row in dolt_rows(spec, max_scanned):
        repo, revision = row["repo"], row["commit_hash"]
        key = repo + "@" + revision
        if key in proofs:
            continue
        if counts["checked"] == max_commits:
            break
        counts["checked"] += 1
        result = subprocess.run(["gh", "api", f"repos/{repo}/license?ref={revision}"],
                                capture_output=True, text=True)
        proof = {"repository": repo, "revision": revision, "license": "unverified"}
        if result.returncode == 0:
            value = json.loads(result.stdout)
            license_id = (value.get("license") or {}).get("spdx_id")
            if license_id in {"MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0"}:
                content = base64.b64decode(value["content"])
                proof.update({"license": license_id, "license_sha256": hashlib.sha256(content).hexdigest(),
                              "license_path": value["path"], "license_url": value["html_url"],
                              "scope": "Root repository license at originating commit; file-specific exceptions still require review."})
        counts["verified" if proof["license"] != "unverified" else "unverified"] += 1
        proofs[key] = proof
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(proofs, indent=2) + "\n")
        print(json.dumps(counts), flush=True)
    if not counts["checked"]:
        raise ValueError("No originating commits found; license audit did not succeed")
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-scanned", type=int, default=500)
    p.add_argument("--max-commits", type=int, default=100)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.execute:
        print(json.dumps(audit(load_spec("github-typo"), args.output, args.max_scanned, args.max_commits)))
    else:
        print(json.dumps({"mode": "dry-run", "max_scanned": args.max_scanned, "max_commits": args.max_commits}))
