# Continual generated-data and judge RL

`training/online_rl.py` runs a foreground, resumable research loop with the
existing tiny BERT edit model. It generates fresh fictional correction pairs
through the attributed CLIProxyAPI Codex subscription `gpt-6-luna` route, checks
the fixed edit vocabulary locally, and admits pairs only when all five judge
rubric flags are true. It then trains a supervised control and a judge REINFORCE
candidate from the same research checkpoint, data, seed and optimizer schedule.
The supervised anchor remains present in both arms.

Each cycle includes fresh admitted sources in its reward pool and fills the
remaining slots with recent screened replay. A bounded replay window grows the
supervised dataset without indefinitely increasing each training epoch. The
classifier label mapping stays fixed: unsupported edits and contexts that exceed
the actual tokenizer budget are rejected and counted. Broadening that mapping
requires a separate model experiment.

## Start, stop and resume

Install the existing [training and scoring dependencies](dataset-and-training.md).
Use a prepared anchor dataset with disjoint human ErAConD development and test
populations. The loop cooperates with the repository-wide `gamma-training.lock`;
another training job cannot start alongside it.

Provide `GAMMA_JUDGE_API_KEY` and, if required for private attribution,
`GAMMA_JUDGE_ATTRIBUTION_TOKEN` through the environment. The base URL defaults to
`http://127.0.0.1:8317/v1`. Do not put credentials in command arguments, a tracked
file, experiment metadata or a PR. Missing/mismatched provider, authentication,
model or response metadata fails closed; there is no API-key provider fallback.

First create a fresh two-round calibration receipt with the current judge code:

```sh
python training/calibrate_judge.py --fixture data/judge-calibration.json \
  --output artifacts/online-calibration --execute
```

Inspect the plan without `--execute`, then run a bounded chunk:

```sh
python training/online_rl.py \
  --data data/prepared/rules-human-v1/prepared \
  --evaluation-dir data/prepared/rules-human-v1/prepared/evaluation \
  --calibration artifacts/online-calibration/receipt.json \
  --output artifacts/continual-judge-rl \
  --new-pairs 64 --rl-rows 128 --replay-rows 2048 \
  --warmup-epochs 2 --epochs 2 --cycles 2 --execute
```

`--initial-checkpoint` optionally starts bootstrap training from a local BERT
checkpoint with exactly the ordered anchor labels. Omit it to use the pinned
cached base model. Bootstrap evaluation and all continuation evaluation are
development-only.

Repeat the identical command with `--continuous` replacing `--cycles 2` to keep
executing bounded cycles. The cumulative request limit, storage limit and other
settings are frozen when the output directory is created; choose them on the
first invocation. The defaults allow no lifetime call cap
(`--max-total-requests 0`), at most 2 GiB of retained output and three consecutive
failed cycles. A larger storage allocation can be selected explicitly. Default
cycles generate four pairs per request and judge eight pairs per request; this
reduces timeout risk on the shared route. `--judge-batch-size 16` is also supported.
Default cycles reserve at most 88 remote requests each; actual calls can be lower because
of filtering and cache hits. Bootstrap makes no remote judge calls.

Stop with Ctrl-C, SIGTERM or:

```sh
touch artifacts/continual-judge-rl/STOP
```

Remove only that STOP file and rerun the same command to resume. The hash-chained
event journal is authoritative; `state.json` is a readable snapshot. Each cycle
durably reserves its worst-case call budget before it starts. Reservations are
never refunded, including failed/interrupted cycles, so a restart cannot reopen
the lifetime cap. Actual attempts are charged before transport. Completed
request ledgers and results are hashed and checked on restart. Preserve the
output directory rather than cleaning individual ledgers or checkpoints.

An interrupted optimizer stage is preserved and abandoned; the next cycle uses
the last fully retained research checkpoint. An interrupted bootstrap uses a fresh
numbered attempt, with the same failure bound. A completed bootstrap can be
reused after an interruption before its journal event was written. Storage or
request exhaustion stops cleanly instead of silently deleting evidence.
Screened fresh rows are retained for replay even when a subsequent training stage
fails; the checkpoint rolls back independently. An interrupted, unjournaled
cycle remains an inspection artifact and does not automatically enter replay.

## Acceptance and interpretation

Both exported FP32 and INT8 development policies must pass the existing gates:
at least 95% edit precision, at most 2% changes to clean controls, no inference
failures and at least 25 predicted edits. Among qualified candidates, select the
best minimum human ErAConD FP32/INT8 F0.5 score only when it exceeds the accepted
score by `--min-gain` (default .001). An unqualified initial checkpoint has a
qualification score of zero. Failed or inferior candidates retain that pointer.

The separate `training` pointer allows research to progress before the model
qualifies. It advances only when the minimum FP32/INT8 **unconstrained human
development F0.5** improves by the same margin, choosing between the matched
supervised and RL candidates. Otherwise it retains its prior weights. Both arms
in the next cycle start from that research pointer. This avoids restarting from
the bootstrap forever when release precision is still inadequate; it does not
relax the qualification gates or authorize deployment.

A supervised winner is recorded as supervised; it is not evidence
that RL helped. Each result records the RL-minus-supervised human development
F0.5 difference for both the gated policies and unconstrained research behavior,
even when neither arm qualifies.

The test population is hashed and used locally for overlap exclusion, but never
inferred during this adaptive loop. Repeated development selection can overfit
the development set. Any quality improvement here requires a fresh independent
human confirmation experiment before a generalization claim or release.

Generation and admission use the same model, so correlated mistakes remain
possible. The calibration cases are agent-authored diagnostic examples, not
human gold. Generated text is explicitly `provider-terms-unverified`,
unreviewed, machine-generated and nonpublishable. Data manifests retain the
anchor hash, generated request/spec provenance, license counts and origin
counts. All generated text, checkpoints and receipts stay in ignored paths.
Human correction references, calibration texts and product users' typed text
are excluded from generated training sources; human references and typed text
are never included in generation or reward prompts. Calibration itself sends
only its declared fictional diagnostic cases.

The loop never replaces browser assets or publishes weights. It can run without
finding a qualified improvement; continued generation is an experiment, not a
guarantee of better writing corrections.

## Live validation, 2026-10-03

The 25-case fictional judge diagnostic passed two independent live rounds:
98.4% flag agreement in each round, 100% repeat agreement and no critical false
positives. The article case `calib13` remained misjudged. These expectations are
agent-authored and unreviewed, so this is not a grammar-accuracy measurement.

An initial larger-batch trial completed one matched cycle: 32 generated pairs,
22 representable pairs screened, 21 admitted, 63 optimizer updates in each arm,
and four nonzero RL advantages across 21 reward sources. Both arms used the same
3,993-row anchor plus those 21 pairs and the same warm-start weights.

| Human ErAConD development metric | Supervised control | Judge RL |
| --- | ---: | ---: |
| FP32 F0.5, out of 100 | 70.91 | 71.30 |
| INT8 F0.5, out of 100 | 70.51 | 70.91 |
| FP32 edit precision | 98.73% | 98.75% |
| FP32 predicted edits | 79 | 80 |
| Clean-control change rate | 0% | 0% |

The 0.39-point FP32 difference is one paired development observation, not a
consistent RL benefit or independent generalization result. That trial used
eight-pair generation and sixteen-pair judging. Other cycles timed out during
generation or judging and retained their checkpoints. The final defaults use
four-pair generation and eight-pair judging. A subsequent small-batch run
completed all eight generation requests (32 pairs) before an intentional STOP
during screening; its partial evidence and charged attempts were retained.
The full local training suite passed 286 tests, including restart, conservative
request reservations, source-pool identity, calibration exclusion, research
progress without release qualification, and data retention after trainer failure.

Generated text, human references, model weights and private receipts are not
included in this public report. The continual experiment still needs longer
observation and a fresh independent human holdout before a quality claim.
