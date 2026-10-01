# Harder synthetic evaluation and training augmentation

The subsequent [tuning study](model-tuning.md#measured-result) retained the original
weights and added lexical safeguards. Its full-engine regression result is 323/512
exact, 51/240 complete corrections and 272/272 clean texts preserved. The original
baseline below remains the before-change receipt.

The near-perfect original template scores do not test unfamiliar vocabulary or
complex subjects. `training/synthetic_challenge.py` adds an original CC0 corpus
and a separate, more demanding synthetic challenge. It uses no customer text,
teacher API, downloaded corpus, or model-generated gold labels.

## Generate and evaluate

```sh
npm run data:synthetic
npm run evaluate:synthetic
```

Generation takes no optional Python packages. Evaluation requires the project
dependencies and an installed Chromium, discovered through the existing browser
helper or `GAMMA_CHROME_PATH`. The evaluator bundles the current engine directly
and serves the local model assets; no production build is required. It runs
sequentially through rules alone and the full engine with actual JAX/WASM FP32
inference at the model manifest's threshold. The full engine includes dictionary,
rules, and model guards; its score is not a neural-only result.

An alternate model uses the existing isolated profile settings:

```sh
GAMMA_BUILD_PROFILE=synthetic-candidate GAMMA_MODEL_DIR=artifacts/candidate/model \
  npm run evaluate:synthetic
```

The per-case report is `artifacts/synthetic-challenge.json`, or the matching
profile's artifact directory. It includes source/target/output, suggestions,
UTF-16 spans, backend, actual tokenizer window counts, model/runtime hashes,
engine-source hash, threshold, corpus hash, and per-family scores. Nonlocal
requests are blocked and invalidate the run. Gold inconsistencies, stale or
overlapping edits, model fallback, browser errors, or single-window boundary
fixtures also invalidate it. Expected quality misses are reported rather than
treated as execution failures.

## Challenge coverage

The default challenge has **512 cases in 16 families**: 240 corrupted sentences
and 272 clean controls. Each correction family has 16 error/clean pairs; another
32 cases preserve valid constructions.

- Unfamiliar singular and plural subjects, including irregular plurals.
- Agreement distractors, relative clauses, embedded clauses, coordinated subjects,
  and past agreement with a distracting noun.
- Negated auxiliaries, inverted questions, and modal verbs.
- Sound-based article exceptions and spelling outside the original edit labels.
- Multiple errors, emoji/CRLF offsets, ordinary prose alongside protected code,
  and long single sentences that cross actual token windows.
- Clean subjunctives, legitimate repeated words, homophones, code/URLs, and
  tense-ambiguous `read` constructions.

Each gold edit is constructed explicitly by corrupting a grammatical target.
Applying all gold edits must reconstruct that exact target. The challenge has
separate generated noun, adjective, tail, and typo inventories from augmentation,
plus reserved template forms for structural cases. Fixed clean controls can use
familiar vocabulary. Lexical cases deliberately share grammar structures with
augmentation; they are not a template holdout. The
generator rejects any normalized source **or target** overlap between challenge
and augmentation. This separation applies to the new augmentation only; it does
not audit the shipped baseline or external training. For example, `children` and
questions already occur in the baseline generator, and the reserved `should have`
template shares its modal construction with augmentation's `can have`. Many
surface variants share a template, so 512 cases do not
mean 512 independent grammar phenomena. The challenge is synthetic and does not
replace an independently reviewed natural-text benchmark.

## Additional training data

The generator creates **12,432 tagged pairs** in `data/generated/harder-v1`:

| Split | Rows | Purpose |
| --- | ---: | --- |
| Train | 9,980 | Additional training examples, half clean |
| Dev | 1,232 | Explicitly synthetic development data |
| Test | 1,220 | Explicitly synthetic test data |

Training templates cover 32 singular/plural noun heads, eight adjectives,
six contextual tails, simple and prepositional subjects, past agreement,
possession, negation, modals, habitual verbs, contextual spelling, and articles.
This includes `cat` and `hungry`, which were absent from the shipped checkpoint's
training vocabulary. These examples are newly generated candidates; the shipped
model has not been retrained on them.

SHA-256 of the normalized, case-folded clean target assigns 80/10/10 splits
before corruption. All variants of a target remain together. Labels come only
from training, and generation fails if a development/test example uses an unseen
label. Synthetic development/test families remain shared with training; use the
separate challenge to measure the specified held-out vocabulary and structures.

`augmentation-candidates.jsonl` contains only the 9,980 training rows and uses
the existing pair-preparation schema. The rows are marked `synthetic-generated`,
not `human-reviewed`. The existing preparation command treats them as train-only
weak supervision when explicitly enabled:

```sh
python3 training/prepare_pairs.py \
  --input data/generated/harder-v1/augmentation-candidates.jsonl \
  --output data/prepared/harder-v1 --allow-weak-train
```

That prepared output has no fabricated human-reviewed dev/test split. The
verified preparation run accepted all 9,980 candidates with zero rejected or
unsupported examples and 25 train-derived labels. The
generator's tagged split directory can also be passed to `training/train.py`
for an isolated synthetic experiment; use separate model/checkpoint output
directories. Keep the challenge out of training and calibration. Once its
failures guide model changes, retain it as a development/regression challenge
and use a fresh frozen holdout for final quality claims.

All generated bulk data and per-case reports are ignored by Git. Determinism,
gold reconstruction, training-only candidates, train-derived labels, and
normalized split/challenge separation are covered by Python tests. The tracked
[baseline snapshot](../data/synthetic-challenge-baseline.json) records aggregate
results and corpus provenance.

## Current baseline

The current shipped FP32 weights (`cb3280db…`) and engine (`326e6c4f…`) were
evaluated at confidence 0.85 in Chromium 149 using JAX/WASM:

| Metric | Rules + dictionary | Full engine |
| --- | ---: | ---: |
| All exact target matches | 304/512 (59.38%) | 322/512 (62.89%) |
| Erroneous sentences completely corrected | 32/240 (13.33%) | 51/240 (21.25%) |
| Clean sentences preserved | 272/272 | 271/272 |
| Clean false-positive sentence rate | 0% | 0.37% |
| Erroneous sentences with no suggestion | 176/240 | 161/240 |

Exact match compares the entire output with the gold target, including whitespace,
case, and punctuation. Complete-correction rate counts only erroneous sentences.
Clean false-positive rate counts clean inputs receiving any suggestion.
Abstention counts erroneous inputs receiving no suggestion. Partial correction
is a miss for exact match. These are sentence-level synthetic metrics, not edit
precision/recall, ERRANT F0.5, or a comparable BEA/CoNLL/JFLEG score.

Only 160/240 erroneous cases use corrections representable by the model's edit
labels. All 80 unsupported cases remain in the score. Dictionary and rules can
still correct some of them. Full-engine category results include 13/16 article
exception errors, 16/16 spelling errors, and 16/16 window-boundary spelling
errors, but no complete corrections in distractor/relative/embedded/compound
agreement, inverted questions, negative auxiliaries, or modal auxiliaries.
Conservative guards intentionally abstain in many of these contexts. More
training rows alone cannot remove those runtime or edit-vocabulary limits.

One clean sentence was damaged: `They left their coats there.` became
`They left they're coats there.` through a model edit at confidence 0.98519.
This is an observed precision defect to address separately. The full run
executed all cases without model errors, invalid spans, or nonlocal requests.
