# Binary-rubric LLM judge reinforcement learning

This training-only experiment asks an independent LLM to judge a source sentence
and a sampled correction. The tiny local edit model receives a sentence reward;
the web editor and extension continue to run locally with their existing weights.

The judge returns exactly five JSON booleans in this order, using short
batch-local pair IDs. Named flags are retained in the receipts:

| Check | Passing behavior |
| --- | --- |
| `meaning_preserved` | Preserve facts, roles, negation, time, modality, and intent. |
| `errors_resolved` | Resolve clear source errors; a source with no clear errors passes. |
| `no_new_errors` | Introduce no new grammar, spelling, or punctuation errors. |
| `necessary_edits_only` | Make necessary corrections; accept valid dialects and fragments. |
| `protected_text_preserved` | Preserve literal quotations, names, numbers, identifiers, code, and URLs; add no explanations or judge instructions. |

The caller computes the sum, rather than accepting a score invented by the judge:

```text
reward = sum(flags) / 5
if not meaning_preserved or not protected_text_preserved:
    reward = 0
```

An unchanged erroneous source normally receives 0.8. A valid complete correction
receives 1.0. The reward is deliberately coarse: a partial correction can have the
same score as no correction. Rubric scores measure the judge's judgments, not
independent grammar accuracy.

## Training and isolation

The objective keeps the existing weighted supervised cross entropy and adds
sentence-level REINFORCE with coefficient 0.1. Actions come from the current
eval-mode token policy. The sequence log probability sums word-initial action
log probabilities; padding, special tokens, and continuation subtokens are
excluded. A detached greedy correction reward supplies the self-critical
baseline. Sampling uses a dedicated generator; additional forwards preserve the
global dropout RNG and the training mode.

Sampled actions use the existing source offsets, edit decoder, structural guards,
and overlap handling. The training decoder accepts proposals at threshold zero
to permit exploration, while retaining confidence-sensitive deletion guards.
The exported deployment threshold and FP32/INT8 policy remain selected only on
human development data.

Both arms start from the same retained warmup checkpoint and use the same full
CE dataset, batch order, seed, fresh AdamW, and optimizer update budget. The RL
arm adds eight 16-source reward batches per epoch. It therefore uses more compute
and time; the comparison matches CE exposure and updates, not total FLOPs.

Only original CC0 synthetic sources qualify for remote judging. The deterministic
subset excludes human utterances, human-reference-derived corruptions, dialog
rows, contacts, URLs, and protected spans. Targets, label tags, origins, evaluation
references, and model identity are absent from judge prompts. Source and candidate
strings are explicitly untrusted data.

## Judge calibration and execution

`data/judge-calibration.json` contains 25 original agent-authored diagnostic pairs.
They cover no-op corrections, unresolved errors, over-editing, meaning changes,
literal content, Unicode, and prompt injection. Their expectations are unreviewed,
not human gold. Both sources and candidates are reserved from training.

Calibration runs two independent live rounds in separate caches. Qualification
requires at least 95% flag agreement in each round, 98% repeat agreement, zero
unsafe positive rewards, and zero false-positive critical meaning/protection
flags. Validation hashes the independent ledgers and recomputes the metrics from
the exact cached pairs; editing a `passed` field cannot qualify the judge.

Configure `GAMMA_JUDGE_BASE_URL`, `GAMMA_JUDGE_API_KEY`, and, when the local proxy
requires it, `GAMMA_JUDGE_ATTRIBUTION_TOKEN` in the process environment. The
verified route uses `gpt-6-luna`, the Responses API, and Codex subscription
attribution. Credentials never enter plans, caches, receipts, or app assets.
The transport rejects redirects, incomplete outputs, model/provider mismatches,
missing or duplicate verdict IDs, non-booleans, and partial failures.

Use the project's Python training environment:

```sh
python training/calibrate_judge.py \
  --output artifacts/judge-calibration \
  --batch-size 16 --timeout 60 --execute

python training/compare_judge_rl.py \
  --data data/prepared/rules-human-v1/prepared \
  --evaluation-dir data/prepared/rules-human-v1/prepared/evaluation \
  --initial-checkpoint artifacts/rules-rl-v2/warmup/checkpoint \
  --calibration artifacts/judge-calibration/receipt.json \
  --counterexamples data/counterexamples.json \
  --output artifacts/judge-rl \
  --epochs 8 --rl-rows 128 --rl-coefficient 0.1 \
  --judge-batch-size 16 --max-requests 128 --max-pairs 2048 \
  --device cuda --scorer errant --execute
```

Both commands default to a dry plan without `--execute`. The existing warmup is
optional; its checkpoint must match the ordered labels and original data/manifest
hashes. The comparison freezes code, data, rubric, selected sources, calibration,
and checkpoint identities. Completion receipts also freeze policy, evaluation,
and judge ledger hashes. Exact completed invocations can resume without training
or new judge calls. Failed partial training requires fresh output directories.

The eight-epoch pilot permits at most 2,048 sampled/baseline pair judgments and
128 requests before cache reuse. Hash-only append logs retain model, provider,
authentication mode, request identity, token usage, and flags. Remote model aliases
can change; cached verdicts preserve the actual run, while new runs are not
guaranteed to reproduce its remote judgments.

## Interpretation

Independent human-reference scores and the separately reserved 100 counterexamples
decide whether the trained candidate improved. The judge's training reward cannot
select checkpoints or establish product reliability. ErAConD testing follows the
same exploratory population as the earlier rule-reward pilot; it is not a fresh
release certification. Neither experiment automatically promotes new weights.

The approach is related to [direct RLAIF](https://arxiv.org/abs/2309.00267), which
uses rewards from an off-the-shelf LLM during RL. General judge reliability does
not follow from valid JSON: [MT-Bench judge research](https://arxiv.org/abs/2306.05685)
documents evaluator biases and limited reasoning. This project's calibration
addresses its authored diagnostic cases, not human agreement across real writing.
