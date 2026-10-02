# FCE training source

[fce upstream](https://www.cl.cam.ac.uk/research/nl/bea2019st/) supplies `human-annotated` supervision.
The adapter imports only the original training population and keeps benchmark
sources/references excluded under both raw and canonical sentence keys.

## Source terms and provenance

- Format: `m2-tar`.
- Source revision: `FCE v2.1 BEA2019`.
- License: `LicenseRef-Cambridge-FCE`.
- Scope: Cambridge FCE v2.1 noncommercial research/education only; retain licence.txt; no unrestricted data/weight publication.
- Unrestricted data/weight publication: blocked pending source-rights review.

- Input SHA256: `c574c1cdba6d3ab5a87280f180133cdcf0609848f5dc87cfa2c3f4b0c07ec67e`.

## Reproduce

```sh
python training/corpus_import.py fce \
  --output data/imported/corpora/fce-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb \
  --limit 10000 --max-scanned 28350
```

This is a dry run. Add `--execute` to import and choose a fresh output directory.
Read the [shared workflow](../training-corpora.md) before changing the bounds.

## Run status

The bounded four-epoch pilot completed on **CUDA** using the pinned tiny-BERT encoder,
seed 42, batch 32, context 128 and a source-specific edit vocabulary (up to 4,096 labels).

| Measurement | Result |
| --- | ---: |
| Scanned corpus records | 28,350 |
| Selected original pairs | 9,979 |
| Added clean controls | 1,654 |
| Imported pairs | 11,633 |
| Pairs actually trained | 11,443 |
| Full JFLEG test diagnostic F0.5 (0–100) | 2.24 |
| Diagnostic edit precision | 41.18% |
| Diagnostic proposed edits | 17 |
| Reference-accepted test sources changed | 2 / 182 |
| Export quality gate | Failed: model edits disabled |

The diagnostic threshold is selected on all 754 development sentences. All 747
test sentences are scored with sentence-wise best-reference ERRANT; this is
not official JFLEG GLEU or a browser benchmark. The exported policy retains the
95% development precision, 2% clean-text-change and 25-edit-support gates across
both FP32 and INT8. No shipped model is changed.

Candidate input SHA256: `624690ba8307589d11602a17d6b97d14573e2ea7a174519b211e01d1ec761db3`.
Model-manifest SHA256: `51dd73ec4502296944c9fcbe4d4b9c1c0a67b8e6c698e49033b4f6c6f14cd594`.
Evaluation SHA256: `f8daefcfff3c3382ab0c4287a279ecbab60ff5593cb32fb8ca7c7586d24f35e1`.
[Exact pilot implementation](https://github.com/thapecroth/gamma_eh/tree/8d97e2af2a77c46ef3cb2391c29035fa896cbb20)
for score reproduction; newer teacher infrastructure is preserved by the PR.

Data attribution: [Yannakoudakis, Briscoe and Medlock (2011)](https://aclanthology.org/P11-1019/).

All weights remain local pending source-rights review.
