# nucle training source

[nucle upstream](https://www.comp.nus.edu.sg/~nlp/corpora.html) supplies `human-annotated` supervision.
The adapter imports only the original training population and keeps benchmark
sources/references excluded under both raw and canonical sentence keys.

## Source terms and provenance

- Format: `m2-tar`.
- Source revision: `NUCLE BEA2019 standardized train archive; actual owner-provided file SHA256 recorded`.
- License: `LicenseRef-NUCLE`.
- Scope: Owner-provided NUCLE agreement; noncommercial research/trials only; preserve archive agreement.
- Unrestricted data/weight publication: blocked pending source-rights review.


Request the official BEA2019 standardized archive and provide its local path.

## Reproduce

```sh
python training/corpus_import.py nucle \
  --output data/imported/corpora/nucle-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb
```

This is a dry run. Add `--execute` to import; owner-provided sources also require
`--input`. GitHub Typo requires `--license-cache` produced by the bounded
`training/audit_github_licenses.py` commit-level check. Read the
[shared workflow](../training-corpora.md) before increasing scan/sample bounds.

## Run status

Owner-provided access is pending; no download or training run is claimed.

All candidates remain local research artifacts. No shipped model is changed.
