# fce training source

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
  --heldout data/imported/public-benchmark/cweb
```

This is a dry run. Add `--execute` to import; owner-provided sources also require
`--input`. GitHub Typo requires `--license-cache` produced by the bounded
`training/audit_github_licenses.py` commit-level check. Read the
[shared workflow](../training-corpora.md) before increasing scan/sample bounds.

## Run status

The specification is ready for the bounded local training pilot; run results are recorded after execution.

All candidates remain local research artifacts. No shipped model is changed.
