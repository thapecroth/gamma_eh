# Rule-generated data and reinforcement learning

This experiment combines controlled synthetic errors with a small human-annotated
conversational corpus, then compares supervised continuation with supervised-anchored
reinforcement learning. It does not replace the shipped browser model. Agent-written
examples are explicitly unreviewed; their results are diagnostic, not natural-writing
accuracy or independent human review.

## Data and provenance

`training/import_eracond.py` downloads the authors' [ErAConD data repository](https://github.com/yuanxun-yx/eracond)
at revision `8401d3601f58170b55f0b1ca3773329c56b116f9`. Its data-only repository
carries the MIT license, copyright 2022 YUAN Xun. The importer checks the archive hash,
retains the license and attribution, and reads aligned `orig`/`corr` text plus optional
second human corrections. These files contain 186 dialogs and 1,732 utterance lines;
an utterance can contain several sentences. This is not the paper's 2,454-sentence,
severity-3 evaluation protocol.

Dialogs receive deterministic 80/10/10 hash assignments before augmentation. Every
row whose source or any reference also occurs in another split is removed from all
affected splits. Dialog identity is available; author identity is not, so this does
not establish author separation. Training uses the first supplied correction;
evaluation preserves all supplied alternatives and all annotation severities.

`training/rule_data.py` reserves the human development/test sources and references,
the counterexample sources and targets, and supplied JFLEG/CWEB populations before
selecting any training pair. It creates weak corruptions from human training references
and 100 original agent-written clean sentences in `data/rule-seeds.jsonl`. Narrow rules
introduce known nonword misspellings, sentence-initial pronoun agreement errors, and
duplicated determiners. Arbitrary tense/preposition/article substitutions and generic
word repetition are excluded because they can remain grammatical. Quoted text, code,
and URLs are not corrupted.

Unchanged clean examples remain in training. Original CC0 templates fill the bounded
mixture; ambiguous `read` to `reads` labels are excluded. Conflicting source/target
pairs are removed. The schema-2 edit vocabulary comes only from training; alignment,
vocabulary and context-budget rejection counts accompany results. The
`prepare_pairs.py --train-only` option preserves human training provenance without inventing development/test
examples from an already assigned training corpus.

Development and test contain human utterances plus separately labeled clean controls
derived from their corrected references. The joint population is used for calibration;
original utterances and reference controls must also be reported separately. Clean
reference controls are useful checks but are not independently collected clean writing.
JFLEG and CWEB remain separately licensed, evaluation-only resources. Never use their
sentences, corrections, predictions or derived corruptions as training data or rewards.

`data/counterexamples.json` contains 100 unreviewed agent-written cases: 84 valid
sentences to leave unchanged and 16 clear corrections. They cover legitimate repetition,
subjunctives, modals, questions, tense ambiguity, compound subjects, names, quoted
errors, URLs/code, Unicode and article sound exceptions. They are excluded from training
and development selection and scored only as a diagnostic after the experiment is frozen.

## Reproduce

Use the pinned dependencies in `training/requirements.txt` and, for actual ERRANT,
`training/requirements-evaluation.txt` in an isolated environment. Run jobs sequentially.
Generated corpora, candidate weights and detailed reports remain under Git-ignored
`data/imported`, `data/prepared`, `models/checkpoints` and `artifacts` directories.

```sh
.venv/bin/python training/import_eracond.py --execute \
  --output data/imported/eracond
.venv/bin/python training/public_benchmark.py prepare --execute \
  --output data/imported/public-benchmark
.venv/bin/python training/rule_data.py \
  --human-corpus data/imported/eracond \
  --reserved-evaluation data/imported/public-benchmark/jfleg \
  --reserved-evaluation data/imported/public-benchmark/cweb \
  --output data/prepared/rules-human-v1 --max-rows 4096 --max-labels 512
.venv/bin/python training/compare_rl.py \
  --data data/prepared/rules-human-v1/prepared \
  --evaluation-dir data/prepared/rules-human-v1/prepared/evaluation \
  --counterexamples data/counterexamples.json \
  --output artifacts/rules-rl-v1 --device cuda --scorer errant
# Review the frozen plan, then repeat with --execute to run sequentially.
```

The comparison defaults to two warmup epochs and two continuation epochs per arm;
`--warmup-epochs 8 --epochs 8` selects the longer pilot below. Use a new output directory
for a different schedule. Remote base weights/tokenizer must already be cached at the
pinned revision; the runner uses `--local-files-only`. To populate that cache, load
the pinned model/tokenizer once through Transformers before running the comparison.
The initial checkpoint's label IDs must exactly match the training vocabulary before
loading. Local warm-start weights remain explicitly nonpublishable; provenance is
recorded without asserting independent release qualification.

Execution freezes source, labels, training/evaluation files, license notices and shared
checkpoint hashes. Successful stages also retain immutable hashes of policy and
evaluation metadata. An identical invocation can reuse completed stages; changed
inputs, assets, thresholds, metrics or schedules fail closed. Incomplete stages require
a fresh directory rather than silently resuming an unknown optimizer state. MIT notices
travel with the raw mixture, prepared data, checkpoint and candidate exports.

## Measured pilot, 2026-10-01

The pinned corpus provided 1,217 train, 118 dev and 188 test utterances after cross-split
overlap/duplicate removal. The 4,096-row mixture selected 1,149 original human pairs,
1,024 weak rule/reference-control pairs, and 1,923 original CC0 templates. Alignment
retained 3,993 training rows and 505 labels; 103 rows were unrepresentable and no retained
taggable rows were lost to the label limit. All training rows fit the 96-WordPiece budget.
Training and inference used an RTX 3070 and four CPU threads.

Development had 118 original utterances plus 118 reference controls. Test had 188
original utterances plus 191 reference controls, with 346 reference edits and 223 clean
sentences across the joint population. The scoring method uses actual ERRANT 3.0.2
token-span edits, choosing the best supplied reference by sentence F0.5. It is not an
official ErAConD paper score or a full-engine/browser benchmark.

The two-epoch smoke comparison produced no edits in either arm. The longer schedule
was chosen because development also showed no edits; its test set was reused as
exploratory confirmation. That repeated use does not provide a fresh final certification.

| Longer continuation | Test edit precision | Test edit recall | Test F0.5 | TP / FP / missed | Clean sentences changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervised | 87.39% | 28.03% | 61.39 | 97 / 14 / 249 | 2 / 223 |
| Anchored REINFORCE | 87.04% | 27.17% | 60.41 | 94 / 14 / 252 | 2 / 223 |

Both arms received 504 continuation updates and 31,944 examples, after the same
504-update warmup. Development selected epoch 2 for both continuations and a shared
FP32/INT8 threshold of 0.6. Both met development constraints, but their held-out precision
fell below the 95% target. `training_seconds` was 11.71 supervised and 13.02 RL; this
field includes pre-export selection/evaluation and excludes the later export checks.
The observed difference in this one-seed run is not evidence that RL is generally worse.

On original human utterances alone, F0.5 was 61.70 supervised and 60.72 RL. Most correct
test edits were punctuation (66) and contractions (22); no agreement corrections were
credited. Both models missed all 16 diagnostic corrections and changed 1 of the 84 valid
agent-written cases. The diagnostics reveal poor transfer to these requested grammar
boundaries despite the conversational test score.

**Do not promote these weights.** Neither model establishes the requested precision
or broad grammar coverage. Keep the currently shipped model intact; use the pipeline
and failure cases to guide reviewed data collection and further development experiments.
Full aggregate receipts and hashes are in [pilot results](rule-data-and-rl-results.json);
the ignored `artifacts/rules-rl-v1` and `artifacts/rules-rl-v2` directories retain detailed
reports, frozen plans, stage receipts, checkpoints and FP32/INT8 candidates.

The follow-up [binary-rubric LLM judge RL pilot](llm-judge-rl.md) uses the same
mixture and warmup. It also found no consistent improvement: joint ErAConD test
F0.5 was 61.39 supervised versus 61.07 judge RL, with separately reserved public
benchmarks reported in that page.

Validation: 119 Python tests passed, followed by 28 focused data/comparison tests
including the new metadata-tampering checks. `npm run check` passed lint, typecheck,
213 app tests, release and agent-harness tests, and both app builds. No browser automation
or production deployment was run for this training-only change.

## Reliability criteria

REINFORCE optimizes sampled edit tags against training labels. A supervised loss
anchors the model, and a detached expected-reward baseline reduces sampling variance.
Correct edits earn reward; wrong edits, missed corrections and changes to clean tokens
incur penalties. These rewards inherit annotation and synthetic-label limitations.
They do not independently verify meaning, fluency or grammaticality. This is not
RLHF, PPO or a reasoning-model GRPO implementation.

The experiment must start both continuation arms from identical checkpoint bytes,
ordered labels and training data, with the same batch order, dropout seed, learning
rate and number of optimizer updates. Action sampling has a separate RNG. Equal
updates do not imply equal computation; report elapsed time and examples seen.

All checkpoint selection and threshold calibration use development only. Existing
precision, clean-text-change and minimum-edit-support constraints remain in force;
when no policy qualifies, model edits stay disabled. Both FP32 and INT8 exports must
pass the same development policy. Report the qualified policy and the unconstrained
diagnostic separately: an identity-output model is not a successful corrector merely
because its clean-text change rate is zero.

Treat a small matched run as evidence about that run. A reliability claim requires
new untouched real writing, human review of edit quality and preserved meaning,
multiple seeds, and tests in the intended product domains. These are empirical gates,
not properties supplied by the words "rules" or "reinforcement learning".
