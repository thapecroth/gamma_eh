# Controlled model tuning

The tuning workflow compares KEEP loss weights and learning rates on one frozen,
publishable corpus. `training/tuning_data.py` combines the original CC0 corpus,
the harder CC0 augmentation, and original clean controls for homophones, modals,
questions, embedded clauses and past-tense `read`. It removes ambiguous
`read → reads` supervision. Normalized challenge inputs/targets and all supplied
natural evaluation inputs/references are excluded before training. These natural
sentences are exclusion keys only, never training examples.

Clean-target grouping preserves the original corpus's split assignments. After
merging, an audit requires normalized sources and targets to be disjoint across
train/dev/test. Conflicting source labels are removed. The label vocabulary comes
from training only. The frozen mixture contains 58,479 training pairs, 7,202
development pairs and 7,490 test pairs, including 18,556 clean training examples
and 66 labels. Dataset hashes are recorded in the results receipt.

The harder 512-case challenge has already been inspected and is a regression
audit, not a blind test. Its exact texts and reserved vocabulary remain excluded.
The new controls share some grammatical constructions with its families, so the
old claim that every structure was reserved no longer applies to tuned students.

The completed study's decoder source is preserved at commit `eeecf8e`. Main's
quality pipeline landed at `310f10a3` while that study ran. Integration retains
its schema-2 edits, per-category floors, disabled policies and multi-pass history.
The new restrictions apply to the schema-1 classifier used by the bundled weights;
rich edit calibration keeps its existing contract. Python's schema-1 evaluator
mirrors the browser restrictions using the bundled dictionary and a shared 30-case
fixture. An additional actual-browser test comparison checks the final integrated
engine against `310f10a3`, independently of the completed-study receipts.

## Reproduction

Use the dependencies in `training/requirements.txt` and a pinned cached base model.
Natural evaluation files remain local and ignored by Git; the JFLEG snapshot used
here is evaluation-only under CC BY-NC-SA 4.0, pinned at
`ee06ff806a208aba815ac45313f4e750a48330a5`. No learner sentences or references are
bundled into the app. Only aggregate metrics and population hashes are committed.

```sh
python training/tuning_data.py --output data/generated/tuning-v1 --evaluation-dir PATH_TO_JFLEG
python training/tune.py --data data/generated/tuning-v1 --output artifacts/tuning-v1
python training/tune.py --data data/generated/tuning-v1 --output artifacts/tuning-v1 --device cuda --execute
```

The four profiles use KEEP weights 0.3/1.0 and learning rates 0.0005/0.0002, with
8 epochs, batch size 128, context 64 and seed 42. They start independently from
the same pinned Apache-2.0 Google BERT-Tiny base, without teacher checkpoints.
Jobs run sequentially. Checkpoints are selected by synthetic development F0.5
at threshold 0.8, then the retained checkpoint receives a development threshold
sweep. Synthetic test metrics do not select profiles.

The completed study used `training/train.py` before the richer quality pipeline
was merged. Its dedicated trainer now lives in `training/train_synthetic.py`;
the main decoded/natural trainer remains intact. Historical plan/source hashes
retain the executed filename. Future tuning plans freeze the dedicated trainer.

The driver freezes input, trainer, calibration and runner hashes plus device and
settings. Resume checks validate receipt settings and exported asset hashes.
The first completed comparison used a schema-1 plan, before the calibration and
runner hashes and device were added; its original plan is retained in the results
receipt. The independently checked FP32 exports all had word-position argmax
agreement 1.0 with PyTorch. INT8 is diagnostic and is not the browser serving graph.

## Actual browser evaluation

```sh
node scripts/evaluate-tuning.mjs --input PATH_TO_JFLEG/dev.jsonl --model MODEL_DIRECTORY --output artifacts/dev-browser.json --thresholds 0.6,0.8,0.85,0.95,0.995
python training/score_tuning.py --input PATH_TO_JFLEG/dev.jsonl --report artifacts/dev-browser.json --output artifacts/dev-score.json
```

Install `training/requirements-scoring.txt` into a separate environment for the
second command. The scorer uses actual ERRANT 3.0.2 and spaCy 3.8.7. It retains all
754 development and 747 test rows and all four references, choosing the reference
with highest sentence edit F0.5, then TP and fewer FP/FN. This custom metric is
**not official JFLEG GLEU**. Clean identity uses NFC and collapsed whitespace,
preserving case. Population hashes and exact output ID/source order are checked.

The browser runner serves local bundled engine/model assets, verifies model asset
hashes, requires JAX/WASM, and fails on nonlocal requests or execution errors.
It collects guarded neural proposals at threshold zero, filters them through a
fixed threshold grid, and applies the engine's rule priority. Public `analyzeText`
calls at both grid ends for the first item in every batch verify equivalence.
Its explicit diagnostic override can evaluate disabled candidates; this does not
change model files or qualify an activation policy. Reports distinguish neural-only
and full-engine predictions from rules alone.
This diagnostic runner accepts only schema-1 models and fixes inference to one
pass; use the richer quality workflow for schema-2 calibration. `--verify-all`
checks both threshold endpoints against the public API for every sentence.

New weights require development precision ≥95%, clean-sentence change rate ≤2%,
and at least 25 predicted edits before promotion. No threshold satisfying these
requirements means an explicit disabled-edit policy, rather than falling back to
the highest unsafe score. Passing synthetic token-label calibration alone never
qualifies natural-text behavior. A zero-edit result cannot meet the support gate.

## Legacy decoder restrictions

The comparison exposed high-confidence out-of-domain replacements such as
unfamiliar nouns becoming `the`, and `be` becoming `because`. Confidence alone
cannot establish a relationship between a source word and a finite output label.
The schema-1 decoder now permits same-family verb agreement only with its existing bounded
subject evidence, and `a ↔ an` changes only for known sound classes. Other
replacements require a known canonical typo or an unknown source near a known
dictionary word (distance ≤1 for short words, ≤2 otherwise). Known words and
homophones abstain; this intentionally loses real-word correction coverage.
The complete dictionary and technical-word set protect valid words.

Whole spelling-token boundaries also prevent edits inside identifiers, compounds,
or Unicode words. Unknown names, acronyms and mixed-case tokens follow the spelling
scanner's exclusions for neural replacements; the existing explicit typo rules have
their separate casing behavior. Duplicate deletions require a whole preceding word
separated only by horizontal whitespace; the first word is retained even when both
receive DELETE logits. Article insertion is restricted to a supported simple subject
with correct `has/have` and a closed, known singular countable object with the
appropriate sound class; it cannot insert an
article into arbitrary phrases such as `I read book reviews.` The dictionary can
still contain errors, and bounded distance does not resolve every ambiguous typo.

See [the tuning results](model-tuning-results.json) for measured outcomes. New
checkpoint qualification and improvements to the existing decoder are reported
separately. The model remains an experimental writing assistant with narrow grammar
coverage, not a general grammar corrector.

## Measured result

All four students completed eight epochs on the frozen mixture. Increasing the
KEEP weight reduced out-of-domain suggestions, but **none qualified** on natural
development data. Their retained local manifests explicitly disable model edits.
No new weights are bundled. The existing FP32 checkpoint and threshold 0.85 are
unchanged; the shipped improvement is the general decoder restriction.

The separate test population was evaluated after fixing the decoder policy. Both
conditions ran the actual browser graph and verified public-API equivalence for
every sentence. Scores below use a 0–100 scale for precision/F0.5.
Repeating the comparison after integrating main's quality pipeline reproduced
these test results; the `integration_check` receipt binds both engine source hashes.

| JFLEG test, 747 sentences | Before | After |
| --- | ---: | ---: |
| Full-engine edit precision | 55.30 | 71.90 |
| Full-engine edit F0.5 | 38.02 | 43.56 |
| Full-engine true / false edits | 261 / 211 | 261 / 102 |
| Full-engine changed clean sentences | 24 / 182 | 2 / 182 |
| Neural-only true / false edits | 7 / 131 | 7 / 1 |
| Neural-only changed clean sentences | 23 / 182 | 0 / 182 |

The neural-only precision after filtering is 87.5%, from just eight predictions;
its recall is only 0.47%. This is evidence for rejecting harmful suggestions,
not sufficient support for a broadly accurate model. On development, full-engine
F0.5 improved from 36.63 to 41.50, with unchanged 247 true edits and false edits
falling from 202 to 103. Neural-only development retained 10 of 11 true edits,
and false edits fell from 126 to 1. Rules/dictionary still changed 38 of 216
development clean sentences, so their natural false corrections remain a follow-up.

The inspected 512-case synthetic regression audit preserved all 272 clean texts,
up from 271, and still completely corrected 51/240 erroneous texts. Exact matches
rose from 322/512 to 323/512 (63.09%). Eighty erroneous cases remain outside the
model's finite label vocabulary. These scores do not measure general grammar accuracy.
