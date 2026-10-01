# Model quality results: 2026-10-01

Mixed training data and a larger student improved independent neural edit scores,
but **none of the four students qualified for release**. All candidate exports
explicitly disable neural edits. The bundled checkpoint remains an experimental,
opt-in model; its weights were not replaced. This comparison establishes useful
training and measurement infrastructure, not production grammar accuracy.

The [aggregate results](model-quality-results.json) contain population, dataset,
code, model and report hashes, all 108 browser conditions, error counts for the
single-pass neural profiles and baseline rules, and toolchain versions. Raw
sentences, teacher outputs, checkpoints and browser predictions remain local.
The [workflow](model-quality.md) describes reproduction and scoring.

## Completed data

| Corpus | Accepted pairs | Role |
| --- | ---: | --- |
| GLM `glm-5.3-flash` | 10,436 | Unreviewed, train-only teacher supervision |
| Pinned C4 correction stream | 10,000 | Synthetic, train-only weak supervision |
| JFLEG development/test | 754 / 747 | Evaluation only, four human references each |

The GLM corpus is available locally at `data/teacher/glm-quality/candidates.jsonl`,
with its merge provenance and hash in `manifest.json`. It merges three resumable
CLIProxyAPI runs: 10,441 accepted input rows, five duplicate sources, no conflicting
targets, and 10,436 final pairs. The final generation run completed all 154 jobs;
the earlier partial ledgers were snapshotted rather than discarded.
The final JSONL SHA256 is
`1f2992c0741813030d48027b9832f1cc23b0b6d4b3823c4fa31a3aa77442a8ee`.

| GLM category | Pairs |
| --- | ---: |
| Clean controls | 2,590 |
| Verb tense | 1,225 |
| Agreement | 1,199 |
| Articles | 1,150 |
| Spelling | 1,100 |
| Prepositions | 1,098 |
| Punctuation | 1,050 |
| Word order | 1,024 |

The provider generated fictional examples; no typed user text was submitted.
Structural validation checks JSON, categories, clean controls, duplicate sources,
and contact/secret patterns. It does **not** verify the correction itself.
Provider terms remain unverified, so the corpus and derived mixed-data weights
are private and nonpublishable. C4 retains its attribution and source notices.

Training froze a **5,249-pair** teacher snapshot before generation finished,
at `data/teacher/glm-quality-training`. Its SHA256 is
`9860b11ada7cf46bc5aa5f3a3caabb1cb61957a0ddd4955a606898c455ba6338`.
The experiments selected 5,000 raw GLM rows from that pool. The additional rows
in the completed corpus were **not trained in this comparison**.

On the same fixed 1,000 C4 rows, schema 2 increased supported pairs from 378 to
596. This measures representability before deployment guards, not accuracy.

## Controlled training

Each arm began with 20,000 raw pairs and trained **15,937 supported rows**,
using eight epochs, batch size 32, learning rate 0.0005 and seed 42. Stable hash
selection, a shared 64-WordPiece training budget, and deterministic downsampling
held supported row counts constant. Every mixed arm trained identical rows:

| Origin | Raw selected | Supported training |
| --- | ---: | ---: |
| Templates | 10,000 | 10,000 |
| C4 | 5,000 | 1,647 |
| GLM | 5,000 | 4,290 |

The template control used 15,937 template rows. Tag alignment and vocabulary
filtering explain most of the mixed-data loss. Labels are train-derived: 42 for
the template control and 2,048 for mixed students. This is one seed and schedule,
without uncertainty estimates. All training rows fit 64 WordPieces, so the
128-token arms test inference context rather than long-context training quality.
Tiny bases use the pinned 2-layer/128-hidden BERT revision; the larger arm uses
the pinned 4-layer/256-hidden BERT revision. Both revisions and training hashes
are in the aggregate report.

## Actual browser scores

These scores use all 754 JFLEG development and 747 test sentences. The scorer
uses ERRANT 3.0.2 with spaCy 3.8.7, selecting the best reference for each sentence
by edit F0.5, then TP and fewer FP/FN. This is a custom extraction-based metric,
**not official JFLEG GLEU**, official ERRANT comparison, or universal grammar
accuracy. JFLEG also contains stylistic rewrites. No rows were filtered for
alignment, vocabulary, or inference.

The following neural-only results use one correction pass through the actual
bundled JAX JS FP32 runtime in Chromium/WASM. Candidate diagnostic manifests
deliberately enable an unsafe private policy at threshold 0.60; the bundled
baseline uses its existing 0.85 threshold. Diagnostic manifests are separate
from the disabled candidate exports.

| Neural profile | Dev precision | Dev F0.5 | Dev clean changed | Test precision | Test F0.5 | Test clean changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bundled baseline | 8.03% | 2.82% | 26/216 | 5.07% | 1.72% | 23/182 |
| Templates / Tiny / 64 | 11.02% | 3.41% | 22/216 | 3.60% | 1.23% | 15/182 |
| Mixed / Tiny / 64 | 26.36% | 8.69% | 26/216 | 14.06% | 4.48% | 22/182 |
| Mixed / Tiny / 128 | 22.38% | 7.98% | 35/216 | 17.72% | 6.55% | 25/182 |
| Mixed / Larger / 128 | 26.50% | 11.64% | 45/216 | 19.77% | 7.78% | 28/182 |

Mixed/Tiny/64 beats the matched template control, and the larger arm has the
best development F0.5. Its test result contains 35 matching edits and **142
unmatched edits**. A roughly 4.5-fold increase over the bundled neural test F0.5
still leaves precision far below the release gate.

The current rules engine includes the dictionary merged on main. At one pass:

| Engine | Dev precision | Dev F0.5 | Dev clean changed | Test precision | Test F0.5 | Test clean changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules | 70.49% | 41.43% | 38/216 | 71.90% | 43.56% | 2/182 |
| Rules + bundled neural | 55.01% | 36.63% | 54/216 | 55.30% | 38.02% | 24/182 |

Active neural diagnostics lower combined F0.5 relative to rules on both splits.
Disabled students produce no neural suggestions, and their combined outputs
equal rules exactly. **Abstaining neural policies do not certify combined
quality**: the rules engine itself changes 17.59% of accepted development
sources, above the 2% clean-text gate. The much lower test clean-change rate
does not replace development evidence.

## Promotion and runtime evidence

A candidate needs at least 95% development edit precision, at most 2% changed
clean development sources, at least 25 predicted development edits, and zero
failures in **both FP32 and INT8** exports under a shared policy. Actual JAX
browser development outputs must also qualify. The 25-edit minimum is an
operational evidence floor, not statistical certification. A prior matched
trial qualified on only 1–5 edits; all four arms were rerun with the floor on
the same frozen rows. None qualified. Test did not select checkpoints or policy.
`calibration.best_unconstrained` contains INT8 metrics; compare actual JAX
results with the corresponding `fp32_candidates` entry instead.

The matrix ran nine profiles (baseline, four disabled policies, four private
diagnostics), two populations, three modes, and one/two passes: **18 reports,
108 conditions, 81,054 predictions**. There were zero startup or inference
failures, zero unexpected network requests, and 432 unchanged protected-text
canary checks. Disabled policies fetched no weights and performed zero neural
inference runs. Report hashes and the engine bundle hash identify the execution.
An additional decoded-text parity check matched all **7,505** single-pass
neural predictions between JAX browser FP32 and ONNX Runtime CPU FP32, covering
the baseline and four diagnostic students on both complete populations.

| Neural profile | FP32 bytes | Test startup ms | Test p50 ms | Test p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Bundled baseline | 17,552,569 | 184.1 | 1.5 | 27.8 |
| Templates / Tiny / 64 | 17,540,701 | 193.3 | 1.5 | 27.4 |
| Mixed / Tiny / 64 | 18,575,800 | 199.5 | 2.3 | 31.0 |
| Mixed / Tiny / 128 | 18,575,800 | 206.1 | 2.3 | 31.2 |
| Mixed / Larger / 128 | 46,598,812 | 314.2 | 9.5 | 70.6 |

These are local loaded-engine sentence timings, including first-use JIT for new
sequence shapes. Each profile starts a fresh browser, but later conditions reuse
caches. Lower second-pass or combined-mode timings do not establish speedups.
Startup excludes Chromium launch. This matrix used WASM; the separate current
web-worker/MV3 suite also executed WebGPU through a software adapter. Physical
GPU performance remains unmeasured.

## Next quality bottlenecks

The completed teacher corpus is available for a later training round, but more
weak labels alone do not solve precision. The measured bottlenecks are label
quality, the large train-derived payload inventory, clean-text changes, and
rules false positives on development. A reviewed, licensed correction subset
and targeted development-error inspection are the next useful experiments.
Keep test frozen, preserve weak-label provenance, and require the same release
gates for any future model.
