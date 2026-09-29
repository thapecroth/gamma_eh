# Dataset and training

The first corpus is original, template-generated English data, dedicated to the
public domain under [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/).
It contains no customer text, scraped text, LLM-generated prose, or external GEC
corpus. The generator source is `training/data.py`.

## Construction

Clean sentences cover pronoun/noun-subject agreement, present/past copulas,
possession, articles, familiar present-tense verbs, and a small spelling lexicon.
Corruptions introduce one or two controlled errors, duplicate a token, or omit
an article. Every clean sentence is also included with `KEEP` labels. Reversible
alignment constructs word-level tags; pairs that the tag vocabulary cannot
represent are rejected.

Each clean sentence gets a SHA-256 group identifier. Its hash assigns train/dev/
test at 80/10/10 **before** corruptions. All variants stay together. A fixed
random seed and sorted iteration make repeated generation byte-identical. Tests
check reconstruction and disjoint clean groups and source strings across splits.

The manifest records seed, license, origin, split counts, tag counts, and file
hashes. Labels are a fixed edit ontology derived from the generator, not an
open-ended decoder. Full generated JSONL is excluded from Git; regenerate it
from the source. The trained artifact includes the dataset manifest.

## Reproduce locally

```sh
uv venv --python 3.10
uv pip install --python .venv/bin/python -r training/requirements.txt
.venv/bin/python training/data.py
.venv/bin/python -m pytest training/test_data.py
.venv/bin/python training/train.py --epochs 8
npm ci
npm run check
```

CUDA is selected when available, otherwise CPU. The verified first machine has
an RTX 3070 with 8 GB VRAM. Training uses batch size 128, sequence length 64,
AdamW at 5e-4, gradient clipping, and a lower `KEEP` loss weight because correct
tokens dominate. CPU threads are capped at four to respect the shared host.
Epoch count and other hyperparameters are configurable. A new run replaces
only the chosen output/checkpoint folders; use alternate paths for comparisons.

The best checkpoint is selected by development edit F0.5 at confidence 0.8.
Calibration then tests a fixed development threshold grid, preferring at least
95% edit precision and at most 2% clean-sentence false positives when feasible.
The final threshold maximizes development F0.5 among eligible thresholds. The
test split is evaluated only after selection. The export step checks ONNX
structure and evaluates FP32 and INT8 files against the same complete test split.

## Evaluation meaning

`models/browser/evaluation.json` reports correct edit precision, recall, F0.5,
sentence tag accuracy, clean-sentence false positive rate, export argmax
agreement, and logit differences. Accuracy includes `KEEP`; precision/recall
excludes it. This is edit-tag evaluation, not ERRANT/M2/GLEU benchmark scoring.

All splits share template families and vocabulary. Disjoint clean strings stop
exact pair leakage but do not test cross-domain or unseen-template generalization.
High scores on this corpus must not be advertised as general grammar accuracy.
A separate human-written regression set exercises common corrections and clean
counterexamples. It remains a small smoke test, not a general quality benchmark.

## Larger data candidates

[Google C4_200M](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction)
provides a CC-BY-4.0 synthetic corpus made by a tagged corruption model; see
[Google's original description](https://research.google/blog/the-c4_200m-synthetic-dataset-for-grammatical-error-correction/).
It is the best next broad-context synthetic source to assess. It is not included
in the first checkpoint. Sample/stream a bounded shard, preserve attribution
and revision, filter unsuitable examples, and report the fraction representable
by the edit vocabulary before training. Do not download the complete corpus
by default or silently mix it with the original CC0 material.

[JFLEG](https://github.com/keisks/jfleg) provides a human-written fluency benchmark
under CC-BY-NC-SA-4.0. Keep it as a separately licensed noncommercial evaluation
resource; it is not a permissive commercial training corpus. W&I+LOCNESS, FCE, NUCLE, and Lang-8 need separate license and access
review; public availability alone does not authorize redistribution. This
repository does not bundle those datasets or claim scores on them.
