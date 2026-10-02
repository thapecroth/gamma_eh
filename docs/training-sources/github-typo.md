# GitHub Typo training source

[github-typo upstream](https://github.com/mhagiwara/github-typo-corpus) supplies `natural-revision-auto-typo-filter` supervision.
The adapter imports only the original training population and keeps benchmark
sources/references excluded under both raw and canonical sentence keys.

## Source terms and provenance

- Format: `github-dolt`.
- Source revision: `ohkqisf5r8i8ht82gtt0kafuhs7bokql`.
- License: `LicenseRef-Repository-Terms`.
- Scope: Each source repository/commit retains its terms. Root-license proof does not resolve file-specific exceptions; candidate weights remain private pending source review.
- Unrestricted data/weight publication: blocked pending source-rights review.

- Sampling: English automatically classified typo edits in md/rst/txt documentation of explicitly selected repositories; root license verified at original commit.
- Mirror: [https://www.dolthub.com/repositories/dolthub/github-typos](https://www.dolthub.com/repositories/dolthub/github-typos).
- [Author permission for the transformed mirror](https://github.com/mhagiwara/github-typo-corpus/issues/1#issuecomment-561297197); not a claim of byte identity with the unavailable original gzip.

## Reproduce

```sh
python training/corpus_import.py github-typo \
  --output data/imported/corpora/github-typo-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb \
+  --limit 500 --max-scanned 500
```

This is a dry run. Add `--execute` to import and choose a fresh output directory.
Supply `--license-cache` from `training/audit_github_licenses.py`.
Read the [shared workflow](../training-corpora.md) before changing the bounds.

## Run status

The bounded four-epoch pilot completed on **CPU** using the pinned tiny-BERT encoder,
seed42, batch32, context128 and a source-specific edit vocabulary (up to4,096 labels).

| Measurement | Result |
| --- | ---: |
| Scanned corpus records | 500 |
| Selected original pairs | 370 |
| Added clean controls | 70 |
| Imported pairs | 440 |
| Pairs actually trained | 440 |
| Full JFLEG test diagnostic F0.5 (0–100) | 0.00 |
| Diagnostic edit precision | Not estimable: no edits |
| Diagnostic proposed edits | 0 |
| Reference-accepted test sources changed | 0 / 182 |
| Export quality gate | Failed: model edits disabled |

The diagnostic threshold is selected on all754 development sentences. All747
test sentences are scored with sentence-wise best-reference ERRANT; this is
not official JFLEG GLEU or a browser benchmark. The exported policy retains the
95% development precision,2% clean-text-change and25-edit-support gates across
both FP32 and INT8. No shipped model is changed.

Candidate input SHA256: `21a3c8cf839fa6b001ded03cf8a252a5e767743f18b02254fc7b5eac67deb995`.
Model-manifest SHA256: `e7953752ff68fd47964ee45df373da9e7626c7598cbee042c97f252bc145261e`.
Evaluation SHA256: `c0809bdbe6ad9ad36b5db7847c681024ecd07dd620c3b884ee775693e95b152c`.
[Exact pilot implementation](https://github.com/thapecroth/gamma_eh/tree/a967f536ab3cb92911209b5d17d3911543cafa45)
for score reproduction; newer teacher infrastructure is preserved by the PR.

Data attribution: [Hagiwara and Mita (2020)](https://aclanthology.org/2020.lrec-1.835/).

All weights remain local pending source-rights review.
