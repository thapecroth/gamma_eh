# wi training source

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
  --heldout data/imported/public-benchmark/cweb
```

This is a dry run. Add `--execute` to import; owner-provided sources also require
`--input`. GitHub Typo requires `--license-cache` produced by the bounded
`training/audit_github_licenses.py` commit-level check. Read the
[shared workflow](../training-corpora.md) before increasing scan/sample bounds.

## Run status

The specification is ready for the bounded local training pilot; run results are recorded after execution.

All candidates remain local research artifacts. No shipped model is changed.
