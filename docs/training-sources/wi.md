# Write & Improve training source

[wi upstream](https://www.cl.cam.ac.uk/research/nl/bea2019st/) supplies `human-annotated` supervision.
The adapter imports only the original training population and keeps benchmark
sources/references excluded under both raw and canonical sentence keys.

## Source terms and provenance

- Format: `m2-tar`.
- Source revision: `W&I v2.1 BEA2019`.
- License: `LicenseRef-Cambridge-WI`.
- Scope: Cambridge W&I v2.1 noncommercial research/education only; retain licence.wi.txt; no unrestricted data/weight publication. LOCNESS is not a training split.
- Unrestricted data/weight publication: blocked pending source-rights review.

- Input SHA256: `d5cbf68cda3da0c3af69dd672614d07287bfe996b87da0c75051d5349d76c666`.

## Reproduce

```sh
python training/corpus_import.py wi \
  --output data/imported/corpora/wi-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb \
+  --limit 10000 --max-scanned 34308
```

This is a dry run. Add `--execute` to import and choose a fresh output directory.
Read the [shared workflow](../training-corpora.md) before changing the bounds.

## Run status

The bounded four-epoch pilot completed on **CUDA** using the pinned tiny-BERT encoder,
seed42, batch32, context128 and a source-specific edit vocabulary (up to4,096 labels).

| Measurement | Result |
| --- | ---: |
| Scanned corpus records | 34,308 |
| Selected original pairs | 9,992 |
| Added clean controls | 1,723 |
| Imported pairs | 11,715 |
| Pairs actually trained | 11,443 |
| Full JFLEG test diagnostic F0.5 (0–100) | 10.72 |
| Diagnostic edit precision | 54.93% |
| Diagnostic proposed edits | 71 |
| Reference-accepted test sources changed | 9 / 182 |
| Export quality gate | Failed: model edits disabled |

The diagnostic threshold is selected on all754 development sentences. All747
test sentences are scored with sentence-wise best-reference ERRANT; this is
not official JFLEG GLEU or a browser benchmark. The exported policy retains the
95% development precision,2% clean-text-change and25-edit-support gates across
both FP32 and INT8. No shipped model is changed.

Candidate input SHA256: `f6af4090431561cf8c5e66ea14a5640c8d6db58c0a2a79bf03fd31066779464d`.
Model-manifest SHA256: `2ba291bfa679db2959bdd7499e5a8c07c1df0f6843f4fb70e4d7cf2d1c98e413`.
Evaluation SHA256: `4098099b020e21f00e0fedeabb313f46f87984684b63780950a802ffb5f5d3db`.
[Exact pilot implementation](https://github.com/thapecroth/gamma_eh/tree/8d97e2af2a77c46ef3cb2391c29035fa896cbb20)
for score reproduction; newer teacher infrastructure is preserved by the PR.

Data attribution: [Bryant et al. (2019)](https://aclanthology.org/W19-4406/).

All weights remain local pending source-rights review.
