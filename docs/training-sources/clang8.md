# cLang-8 training source

[clang8 upstream](https://github.com/google-research-datasets/clang8) supplies `gT5-generated-target` supervision.
The adapter imports only the original training population and keeps benchmark
sources/references excluded under both raw and canonical sentence keys.

## Source terms and provenance

- Format: `tsv`.
- Source revision: `0fa5bf90bfa7a038b62175c7b262c4b72023393d`.
- License: `CC-BY-NC-SA-4.0`.
- Scope: Data CC BY-NC-SA 4.0; Apache code does not override Lang-8 source and cLang-8 target terms.
- Unrestricted data/weight publication: blocked pending source-rights review.


Obtain original Lang-8 through owner access, run pinned official cLang-8 alignment recipe, provide English source/target TSV.

## Reproduce

```sh
python training/corpus_import.py clang8 \
  --output data/imported/corpora/clang8-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb
```

This is a dry run. Add `--execute` to import and choose a fresh output directory.
Supply the English source/target TSV from the pinned official alignment recipe with `--input`.
Read the [shared workflow](../training-corpora.md) before changing the bounds.

## Run status

Owner-provided access is pending; no download or training run is claimed.

All candidates remain local research artifacts. No shipped model is changed.
