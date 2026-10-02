# Natural-corpus training

The corpus importer and sequential runner support per-source research models and
a mixed-source candidate. These are bounded integration experiments. They do
not train on the entire C4/WikEd corpora, certify model quality, or replace the
shipped model. Corpus text, checkpoints and ONNX candidates stay Git-ignored.

## Sources and PR boundaries

Each source has its own specification and documentation change. Importer and
runner infrastructure is reviewed separately. Source PRs depend on the shared
infrastructure PR; their training results must distinguish completed runs from
missing archives. See the individual source pages:

- [Write & Improve](training-sources/wi.md)
- [FCE](training-sources/fce.md)
- [NUCLE](training-sources/nucle.md)
- [Lang-8](training-sources/lang8.md)
- [cLang-8](training-sources/clang8.md)
- [C4_200M](training-sources/c4.md)
- [GitHub Typo Corpus](training-sources/github-typo.md)
- [WikEd](training-sources/wiked.md)

The original source terms travel with every pair and prepared manifest.
`--allow-research` permits explicitly named research/share-alike corpus terms
without adding them to the permissive-license allowlist. Research-only and
unreviewed source-text rights block `publication_allowed` in both prepared
datasets and exported models. An Apache base model does not override corpus
restrictions. Publishing code or aggregate results does not publish the data
or authorize weight redistribution.

## Import

```sh
python training/corpus_import.py wi \
  --output data/imported/corpora/wi-v1 \
  --heldout data/imported/jfleg-evaluation \
  --heldout data/imported/public-benchmark/cweb
# Review the dry-run specification; --execute imports the bounded sample.
```

Automatic downloads require exact verified hashes. Owner-provided M2 archives
and aligned TSVs use `--input`; their actual file hashes remain recorded.
Neither missing access nor an unavailable upstream URL is a successful import.
Gated Lang-8/NUCLE archives must come from the owner-authorized download.
cLang-8 requires the original Lang-8 input and the pinned official preparation
recipe; a properly aligned English TSV can then be imported locally.

Imports select the lowest seeded pair hashes among a bounded scanned prefix.
This is reproducible, but prefix/selected-repository sampling is not a
representative full-corpus design. Counts distinguish scanned, valid, selected,
conflicting and rejected examples. All original training splits remain
train-only. Weak or transformed labels never become independent human test
data. Both raw and canonical sentence/reference keys are excluded against all
reserved evaluation inputs before selection.

M2 annotators remain separate; unresolved or overlapping annotations are
rejected. Text is rendered into edit-schema-2 canonical training spacing,
because raw spaCy-tokenized punctuation is incompatible with the tiny edit
decoder. Original evaluation text and offsets remain unchanged. Deterministic
clean controls use a subset of corrected training targets and retain the same
clean group and provenance; existing clean pairs are also retained.

## Train sequentially

Install the pinned training requirements and the separate scorer requirements
in an isolated Python 3.10 environment. Cache the pinned Google tiny-BERT base
before using `--local-files-only`.

```sh
python training/corpus_training.py \
  --corpus wi=data/imported/corpora/wi-v1 \
  --corpus fce=data/imported/corpora/fce-v1 \
  --evaluation-dir data/imported/jfleg-evaluation \
  --output artifacts/corpus-training/run-001 \
  --epochs 4 --batch-size 32 --max-length 128 --combined
# Add --execute to prepare and train each source, then the available-source mix.
```

The runner holds a repository-wide file lock and launches exactly one training
process at a time. It records inputs, hashes, schedule, toolchain, code hashes,
coverage and independent evaluation hashes. Completed arms resume only under
the identical plan and unchanged model manifest; interrupted arms are retained
and require a fresh run directory. This does not promise continuation from an
interrupted optimizer state.

If the execution device becomes unavailable, retain that receipt and use a fresh
output directory with `--device cpu --only-arm SOURCE` (repeat `--only-arm` for
other unfinished arms and `combined`). The new plan still pins all combined
inputs. Already completed arms need not be retrained. Aggregate publication
verifies each completed model under its original plan and records each arm's
actual device; CPU/CUDA results are not a controlled corpus ranking.

Fresh pretrained encoder initialization avoids silently reusing classifier
weights under a changed edit-label vocabulary. The current trainer loads the
bounded training population into memory. Million-scale training requires
tokenized streaming shards and a separate schedule; do not increase bounds
without measuring memory, throughput and edit coverage first.

Checkpoint and threshold selection use the complete development population.
The existing precision/clean-text/support constraints remain intact. A model
that does not qualify exports `disableModelEdits: true`; report that outcome
alongside development-selected diagnostic performance. Test scores use the
complete population, including corrections outside the edit representation.

The runner's ERRANT diagnostics use the repository's sentence-wise best-reference
policy and deployment guards. They are separate from official corpus JFLEG GLEU,
original-gold CWEB ERRANT and actual browser benchmark receipts. JFLEG has already
been used in prior model studies; it is a regression evaluation, not a newly
blind benchmark. The [public benchmark](public-benchmark.md) remains the final
browser confirmation before any candidate could replace the shipped model.
