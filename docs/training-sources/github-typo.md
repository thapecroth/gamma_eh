# github-typo training source

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
  --heldout data/imported/public-benchmark/cweb
```

This is a dry run. Add `--execute` to import; owner-provided sources also require
`--input`. GitHub Typo requires `--license-cache` produced by the bounded
`training/audit_github_licenses.py` commit-level check. Read the
[shared workflow](../training-corpora.md) before increasing scan/sample bounds.

## Run status

The specification is ready for the bounded local training pilot; run results are recorded after execution.

All candidates remain local research artifacts. No shipped model is changed.
