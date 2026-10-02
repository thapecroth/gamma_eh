# Lang-8 training source

[lang8 upstream](https://sites.google.com/site/naistlang8corpora) supplies `crowd-corrected` supervision.
The adapter imports only the original training population and keeps benchmark
sources/references excluded under both raw and canonical sentence keys.

## Source terms and provenance

- Format: `m2-tar`.
- Source revision: `Lang-8 BEA2019 standardized train archive; actual owner-provided file SHA256 recorded`.
- License: `LicenseRef-Lang8`.
- Scope: NAIST Lang-8 research/education only; commercial permission must come from corpus owner.
- Unrestricted data/weight publication: blocked pending source-rights review.


Request the official BEA2019 standardized archive and provide its local path.

## Reproduce

```sh
python training/corpus_import.py lang8 \
  --output data/imported/corpora/lang8-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb
```

This is a dry run. Add `--execute` to import and choose a fresh output directory.
Supply the owner-authorized BEA2019 standardized M2 archive with `--input`.
Read the [shared workflow](../training-corpora.md) before changing the bounds.

## Run status

Owner-provided access is pending; no download or training run is claimed.

All candidates remain local research artifacts. No shipped model is changed.
