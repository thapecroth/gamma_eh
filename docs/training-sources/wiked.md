# wiked training source

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
  --heldout data/imported/public-benchmark/cweb
```

This is a dry run. Add `--execute` to import; owner-provided sources also require
`--input`. GitHub Typo requires `--license-cache` produced by the bounded
`training/audit_github_licenses.py` commit-level check. Read the
[shared workflow](../training-corpora.md) before increasing scan/sample bounds.

## Run status

The specification is ready for the bounded local training pilot; run results are recorded after execution.

All candidates remain local research artifacts. No shipped model is changed.
