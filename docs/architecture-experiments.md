# Grammar architecture experiments

This research harness compares local edit tagging and generation using natural
human corrections. It never replaces the bundled model or enables a candidate
that fails its development quality gates. Results and the architecture decision
are added below after the sequential run completes.

## Protocol

The WI+LOCNESS v2.1 original development documents are split by author into a
development set (2,332 correction-evaluable sentence windows, 856 clean) and a
research test set (1,670 windows, 523 clean). Training excludes 183 documents
whose authors occur in either holdout. Six test windows overlapping development
source/reference text are removed before model selection. Detection-only null
annotations and corrections crossing sentence boundaries are excluded before
freezing; counts and exact input hashes are retained. This is a research holdout,
not the official BEA blind test. JFLEG's 747 test sentences and four references
provide a separate fluency-oriented evaluation; neither JFLEG split is training
data. Exact source/reference matches are excluded from all training sources.

The deterministic 20,000-pair pool contains 10,000 human learner pairs, 5,000 C4
synthetic pairs, and 5,000 original synthetic templates. Basic and richer edit
formats use the same 11,689 supported pairs. A template-only arm matches that
row count; a separate full-rich arm tests the additional representable pairs.
Richer tags include case and suffix transforms, leading insertion, punctuation,
and up to four-token payloads. Vocabulary derives only from training.

Tiny, Mini, and the pinned `nreimers/MiniLM-L6-H384-uncased` mirror use identical
rows, seed 42, AdamW learning rate 0.0001, batch 32, and four epochs. The mirror
retains alternating layers of the 12-layer MiniLM checkpoint; it is not evidence
for every model sold as MiniLM. Twelve-epoch sensitivity runs check whether the
short budget unfairly favors a backbone. Context arms use 64, 128, and 256
WordPieces and preserve each supervised word once in whole-word windows.
Window counts and optimizer updates can differ with context; equal epochs do
not mean equal training compute.

Decoded ERRANT precision, recall and F0.5 use the entire frozen population,
including unsupported corrections and inference failures. A threshold qualifies
only on development with at least 95% point precision, at least 30 predicted
edits, at least 50 clean controls, at most 2% clean-text changes, and no failures.
Wilson precision intervals are descriptive and are not the qualification rule.
If no threshold qualifies, the model is explicitly disabled. Diagnostic scores
use a development-selected threshold and remain separately labeled. Browser
diagnostics use private derived manifests; original candidate manifests retain
the disabled policy. Basic/rich inference differs in the legacy verb guard as
well as representation, so their quality difference is not a pure tag ablation.

JFLEG additionally uses its pinned official GLEU implementation with 500 seeded
reference draws. Custom best-sentence-reference ERRANT scores are diagnostic
and do not replace official GLEU. Its reference-sampling interval is not a
population confidence interval.

The T5-small comparison uses the same matched pairs for full-sentence correction
and source-aware span repair. Original suspect text remains between markers.
Clean masked words teach copying after false detection. Span inference uses
`1-P(KEEP)` from a frozen, development-selected edit model, with thresholds chosen
on development; test reference spans are never input to the model. One bounding
span may encompass distant detections, so this is a bounded prototype rather
than a reproduction of EdiT5 or DeCoGLM.

All typed text, research data, inference and training stay local. WI+LOCNESS and
JFLEG have noncommercial restrictions. Research weights and raw corpora remain
ignored and cannot be promoted into the commercial extension. Base checkpoint
licenses do not override training-corpus terms.

## Reproduction

Install Python training dependencies plus `training/requirements-evaluation.txt`
and `training/requirements-generation.txt`
in an isolated environment. Download the official archive and pinned JFLEG/C4
sources into ignored `data/imported/`, retaining their licenses. Freeze once:

```sh
python training/architecture_data.py \
  --archive data/imported/wi-locness-v2.1.tar.gz \
  --templates data/generated/train.jsonl \
  --c4 data/imported/c4-quality-v2/candidates.jsonl \
  --jfleg data/imported/jfleg-evaluation \
  --output data/prepared/architecture-sweep
python training/architecture_sweep.py --prepare
python training/architecture_sweep.py --epochs 4
python training/benchmark_sweep.py
```

Run generation separately, then export and benchmark each trained objective:

```sh
python training/generation_pilot.py --objective span \
  --detector artifacts/architecture-sweep/models/SELECTED-ON-DEV \
  --output artifacts/architecture-sweep/generation-span
python training/generation_export.py --model artifacts/architecture-sweep/generation-span
node scripts/benchmark-generation.mjs \
  --model-dir artifacts/architecture-sweep/generation-span/browser \
  --output artifacts/architecture-sweep/generation-span/browser-results.json
```

Repeat with `--objective full`. Browser generation measures single-thread WASM
model computation on 32 fixed, evenly spaced JFLEG inputs. It excludes tokenization,
detection and UI, and uses an uncached decoder. This is not a production latency
claim. Tagger evaluation measures complete engine analysis separately.

Run all heavy jobs sequentially. An existing completed evaluation receipt is
retained; partial directories must be preserved or moved before retrying. No
dataset or model files are committed.

## Primary sources

- [Official WI+LOCNESS data and licenses](https://www.cl.cam.ac.uk/research/nl/bea2019st/)
- [JFLEG corpus and official scorer](https://github.com/keisks/jfleg)
- [MiniLM mirror model card](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased)
- [GECToR](https://aclanthology.org/2020.bea-1.16/)
- [EdiT5](https://aclanthology.org/2022.findings-emnlp.156/)
- [DeCoGLM](https://aclanthology.org/2024.acl-long.96/)
