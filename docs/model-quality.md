# Independent model quality workflow

The template baseline generalizes poorly to the independent JFLEG reference
population. Improve data coverage and measure the deployed engine before
promoting a student. Synthetic classifier scores remain synthetic scores.
Raw corpora, checkpoints, predictions, and provider-derived weights stay in
Git-ignored directories. Typed user text is never sent to a teacher.

The [completed comparison](model-quality-results.md) records 10,436 GLM pairs,
10,000 C4 pairs, four trained students, and 81,054 actual browser predictions.
Mixed data and model capacity improved diagnostic scores, but no student met
the release gates. The original bundled weights remain in place.

## Frozen evaluation population

`training/import_jfleg.py` imports the official JFLEG repository at
`ee06ff806a208aba815ac45313f4e750a48330a5`: all 754 development and 747 test
sentences, each with four human references. It saves original files, hashes,
license, and provenance. JFLEG is a **fluency** benchmark, including rewrites;
these scores do not describe universal grammar accuracy or customer acceptance.
Its CC-BY-NC-SA-4.0 evaluation license is separate from training. Never include
this benchmark in training or browser assets.

Development chooses checkpoints and confidence policies. Test is evaluated
afterward. Both training fields are checked against the normalized union of
all heldout sources and references. Preparation retains complete evaluation
JSONL before tag alignment or vocabulary filtering; unsupported references and
failed inference remain in the decoded score denominator.

```sh
uv venv artifacts/training-env --python python3.10
uv pip install --python artifacts/training-env/bin/python \
  -r training/requirements.txt -r training/requirements-evaluation.txt
artifacts/training-env/bin/python training/import_jfleg.py \
  --output data/imported/jfleg-evaluation --execute
```

The optional scorer uses ERRANT 3.0.2, spaCy 3.8.7, the pinned English model
3.8.0, and RapidFuzz 3.14.5. Click/Typer are pinned because spaCy's CLI imports
Click and newer Typer releases no longer supply it. The stdlib fallback is
explicitly called **approximate**, not ERRANT. Our multi-reference policy chooses
the highest sentence edit F0.5, then TP, then fewer FP/FN. It is a custom use of
ERRANT extraction, not official JFLEG GLEU or the official ERRANT comparison CLI.
Reports include scorer versions, per-error-type counts, clean-text false
positives, and exact reference matches.

## Edit schema 2

Legacy schema-1 models still use their existing labels. Schema 2 adds exact-case
replacement, append and prepend payloads of at most four tokens, reusable case
operations, and six suffix transformations. This supports initial insertions,
multiword substitutions, and punctuation changes. Roundtrip checks reject
unsupported case/spacing or payloads instead of silently relabeling them KEEP.

On the exact same fixed 1,000-row C4 sample, preparation accepted **378 rows with
schema 1 and 596 with schema 2**. This is training representability, before
deployment guards. It is not correction accuracy. Schema 2 still rejected 404
rows: 379 case/spacing mismatches and 25 payload-limit failures.

The browser decoder retains source whitespace and UTF-16 offsets. Article
sound-class guards, protected code/URLs, conservative deletions, and numeric/
symbol protection remain in effect. Python guarded evaluation mirrors these
operations and the current full-source verb agreement guard. Every article in
an exact payload is checked against its following payload token or source
neighbor; unknown sound classes abstain. Punctuation-to-word replacements add
exterior spacing without changing source offsets. Shared edit and pinned BERT tokenizer fixtures cover case,
contractions, accents, Chinese isolation, and Unicode offsets. Training rejects
incompatible BERT preprocessing configurations.

An optional `maxPasses` of 1–3 lets a later pass repair newly inserted text.
`EditHistory` composes accepted edits against the original source; cycles and
unchanged passes stop early. Insertions and composed suggestions carry a source
snapshot so an empty insertion span cannot accept stale text. The default is
one pass. `mode: 'rules' | 'model' | 'combined'` makes ablation explicit;
rules-only checks do not load a model.

## Licensed and teacher training data

```sh
artifacts/training-env/bin/python training/import_c4.py \
  --limit 10000 --max-scanned 15000 \
  --output data/imported/c4-quality-v2 --execute

# Provide TEACHER_API_KEY securely in the environment.
artifacts/training-env/bin/python training/generate_llm.py \
  --base-url http://127.0.0.1:8317/v1 --model glm-5.3-flash \
  --pairs 10000 --batch-size 50 --concurrency 4 --max-tokens 24576 \
  --reasoning-effort low --json-mode --timeout 300 --retries 2 \
  --seed 46 --output data/teacher/glm-new-run --execute
```

The teacher generates fictional examples across 16 domains, seven error
categories, and clean controls. Structural checks reject malformed, truncated,
wrong-category, secret/contact-pattern, and inconsistent clean outputs. They do
**not** independently verify grammatical correctness. Avoid preference-only
adverb movement and preserve facts in the teacher prompt.

SQLite commits each completed job. Periodic atomic snapshots expose committed
pairs during long runs. Repeat identical settings to resume unfinished requests;
concurrency and snapshot frequency may change without repeating completed jobs.
Change generation settings or prompt into a new directory. The same command
with `--snapshot-only` exports an interrupted ledger without calling a provider.
Each new run saves `prompt.txt`; use `--prompt-file RUN/prompt.txt` to resume
after the default prompt changes.
Completed-request token usage excludes failed/interrupted attempts.

`training/merge_teacher.py --input RUN_A --input RUN_B --output NEW_DIRECTORY`
verifies snapshot hashes and row provenance, removes conflicting sources, and
merges deterministically. Teacher inference itself is not bit reproducible.
Dataset assembly, seeds, population hashes, and completed-request resume are.
CLIProxy may translate GLM requests through a Claude-compatible route. On that
route, `response_format` is not forwarded, and `max_tokens` must be used instead
of `max_completion_tokens`. Prompt and response validation remain required;
`--json-mode` does not prove upstream JSON mode. Higher request concurrency
triggered provider 429/503 limits in the live run;
four concurrent requests restored progress. Do not infer endpoint readiness
from the model catalog alone.

Teacher pairs remain train-only weak supervision with unverified provider terms.
Prepared and exported manifests block publication of their data and weights.
C4 attribution, revision, URLs, input hashes, and source-corpus notice scope travel
with the prepared artifact. Only the two verified Apache-2.0 BERT base revisions
receive automatic model-license attribution; arbitrary compatible bases remain
nonpublishable until their provenance is added and reviewed.

## Controlled students and deployment calibration

```sh
artifacts/training-env/bin/python training/data.py
artifacts/training-env/bin/python training/experiments.py \
  --template data/generated \
  --weak data/imported/c4-quality-v2/candidates.jsonl \
  --weak data/teacher/glm-quality-training/candidates.jsonl \
  --evaluation-dir data/imported/jfleg-evaluation \
  --output artifacts/model-quality-v2 --rows 20000 --max-labels 2048 \
  --epochs 8 --batch-size 32 --scorer errant \
  --common-token-budget 64 --tokenizer-model models/browser --prepare
artifacts/training-env/bin/python training/experiments.py \
  --run-plan artifacts/model-quality-v2/plan.json --execute
```

The four predeclared arms compare templates/Tiny/64, mixed/Tiny/64,
mixed/Tiny/128, and mixed/BERT-4-layer-256/128. The mixed raw composition is half
templates, one quarter C4, one quarter GLM. Stable hash sampling, a shared
WordPiece budget, and deterministic downsampling equalize supported training
counts. Every mixed arm trains identical rows with the same schedule and seed.
Label inventories remain train-derived and can differ between template/mixed
datasets. Because all training rows fit 64 WordPieces, the 128 comparison tests
inference context; it does not establish long-context training quality.

The completed run used the frozen `glm-quality-training` pool of 5,249 pairs,
selecting 5,000 before tag/vocabulary filtering. Its mixed arms ultimately
trained 10,000 template, 1,647 C4 and 4,290 GLM rows. The larger completed
`glm-quality` corpus was generated concurrently and is available for future
experiments; using it above produces a new comparison rather than reproducing
the frozen results. Final exports are under `artifacts/model-quality-v3/`.

Run arms sequentially. The plan verifies dataset, label, manifest, and evaluation
hashes before each run. Checkpoint selection uses decoded natural development
predictions. The browser executes FP32 `model.onnx` through bundled JAX JS. ONNX FP32 and INT8 exports each run the complete development
population; their shared threshold must meet both **95% edit precision** and
**at most 2% clean-sentence changes**, with at least **25 decoded development edit predictions** in each export
and zero inference failures. The support minimum is an operational evidence
floor, not statistical certification. A first matched trial produced policies
with just 1–5 development edits; all four arms were rerun on the same frozen
rows after adding this support floor. Optional category thresholds only tighten the policy. If no shared
active policy qualifies, `disableModelEdits: true` explicitly suppresses edits and skips weight loading
and runtime initialization;
the diagnostic unconstrained score is reported separately. No-edit precision is
not evidence of correction quality. ONNX calibration is a prerequisite; complete
actual JAX browser development predictions must also qualify before promotion.
INT8 remains a portable diagnostic export, not the current browser weights. Test never selects the deployment policy.

## Actual browser evaluation

```sh
node scripts/evaluate-browser.mjs \
  --input data/imported/jfleg-evaluation/test.jsonl \
  --model-dir artifacts/model-quality-v3/mixed-tiny64/model \
  --output artifacts/browser-quality/mixed-tiny64-test.json \
  --modes rules,model,combined --passes 1,2
artifacts/training-env/bin/python training/evaluate.py \
  --predictions artifacts/browser-quality/mixed-tiny64-test.json \
  --evaluation-dir data/imported/jfleg-evaluation --split test --scorer errant \
  --output artifacts/browser-quality/mixed-tiny64-test-errant.json
```

Browser evaluation bundles the actual JAX engine with the same Windows adapter
guard as production, records its runtime versions
and bundle hash, checks model asset hashes,
serves only local assets, blocks unexpected network requests, and checks protected
text canaries outside the benchmark denominator. It records actual corrected
text, suggestions, inference counts, startup, and p50/p95 analysis latency.
Failures remain represented. Latency samples are post-initialization and
include first-use shape compilation; condition order warms caches. `--limit` is a smoke subset and cannot be scored as
the complete benchmark. Default measurements use WASM; `--webgpu` records the
actual backend and may fall back to WASM. Software adapters do not prove physical
GPU performance. This engine evaluation is separate from web-worker/MV3 browser
verification (`npm run test:browser`).

All four completed candidate manifests disable neural edits. To inspect a
candidate's unconstrained behavior, use a separate private diagnostic directory
with the development diagnostic threshold and `disableModelEdits: false`;
never modify the calibrated export or treat these diagnostic profiles as a
deployment recommendation. The public aggregate report records both policies
and diagnostics, with no-edit precision represented as null.

## Sources

- [JFLEG repository, references, and license](https://github.com/keisks/jfleg)
- [ERRANT extraction and comparison methodology](https://github.com/chrisjbryant/errant)
- [GECToR edit tagging and iterative correction](https://aclanthology.org/2020.bea-1.16/)
- [Original C4_200M correction dataset](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction)
- [GLM thinking behavior](https://docs.z.ai/guides/capabilities/thinking-mode)
