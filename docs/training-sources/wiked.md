# WikEd training source

[wiked upstream](https://github.com/snukky/wikiedits) supplies `natural-revision-unreviewed` supervision.
The adapter imports only the original training population and keeps benchmark
sources/references excluded under both raw and canonical sentence keys.

## Source terms and provenance

- Format: `wiked-prefix`.
- Source revision: `2e307bf1cd43e0b3247e4968624bbd5eafc2272a`.
- License: `CC-BY-SA-3.0`.
- Scope: Wikipedia 2014 text inherits CC BY-SA 3.0; retain dump, page/revision metadata and source attribution; no unrestricted weight publication claim.
- Unrestricted data/weight publication: blocked pending source-rights review.

- Input SHA256: `aec9861b1ecda255e536c36b3deab124b8aef3995f35520fe7f3ffb4272edab3`.
- Sampling: Exact 16MiB compressed raw-archive prefix, one nested member; archive is not fully downloaded. Hash covers the prefix, not the whole corpus.

## Reproduce

```sh
python training/corpus_import.py wiked \
  --output data/imported/corpora/wiked-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb \
+  --limit 10000 --max-scanned 50000
```

This is a dry run. Add `--execute` to import and choose a fresh output directory.
Read the [shared workflow](../training-corpora.md) before changing the bounds.

## Run status

The bounded four-epoch pilot completed on **CPU** using the pinned tiny-BERT encoder,
seed42, batch32, context128 and a source-specific edit vocabulary (up to4,096 labels).

| Measurement | Result |
| --- | ---: |
| Scanned corpus records | 50,000 |
| Selected original pairs | 9,773 |
| Added clean controls | 1,849 |
| Imported pairs | 11,622 |
| Pairs actually trained | 10,139 |
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

Candidate input SHA256: `03f55e99bec8470f722d6d3d2b6791d6a20a85da3baa3aeacdba0fb4b3d4b0b5`.
Model-manifest SHA256: `0c62d92b93bfb6d6df0fcf0797d0f1cf68c10892e8f692e6812a4f8b125c7dfb`.
Evaluation SHA256: `c8bc61e2c40f2f70f44964382b0b083b6f0a8570f67a2e9755a526ac13db9895`.
[Exact pilot implementation](https://github.com/thapecroth/gamma_eh/tree/a967f536ab3cb92911209b5d17d3911543cafa45)
for score reproduction; newer teacher infrastructure is preserved by the PR.

Data attribution: [Grundkiewicz and Junczys-Dowmunt (2014), author repository](https://github.com/snukky/wikiedits/tree/2e307bf1cd43e0b3247e4968624bbd5eafc2272a).

All weights remain local pending source-rights review.
