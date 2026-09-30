# Scaling the English correction dataset

The first checkpoint validates the end-to-end pipeline. For useful broader
coverage, combine teacher-generated examples with a bounded public GEC corpus
and independently reviewed evaluation data. The live CLIProxyAPI pilot is
documented in [End-to-end pilot](live-pilot.md). Credentials are provided through
environment variables and never written into the repository or run ledger.

## Teacher generation

`training/generate_llm.py` supports OpenAI-compatible chat-completions gateways
and Anthropic messages endpoints. It uses Python's standard HTTP client, so no
SDK is required. Configure the base URL including `/v1`, model name, and
`TEACHER_API_KEY`. Models whose gateway requires `max_completion_tokens` can
select `--max-token-field max_completion_tokens`.

Plan a million-pair run without contacting an endpoint:

```sh
.venv/bin/python training/generate_llm.py \
  --base-url http://localhost:8000/v1 \
  --model YOUR_MODEL \
  --pairs 1000000 --batch-size 20 --concurrency 4 \
  --output data/teacher/english-v1
```

The plan shows request count and maximum requested output tokens; input tokens
and retries add to actual usage. Begin with a small run to measure validity,
diversity, cost, and quality before increasing the requested pair count. Add
`--execute` to actually generate. Repeat the identical command to resume.
Changing model/prompt/seed/output budget requires a new output directory.
Concurrency can be adjusted on resume.

Use `--provider anthropic --base-url https://api.anthropic.com/v1` for Anthropic.
The complete system prompt lives in [teacher.txt](../training/prompts/teacher.txt).
Requests rotate across domains and error types with stable diversity nonces;
every fourth batch consists of clean examples with identical source/target.

SQLite tracks committed jobs, pair hashes, unique source text, sanitized token
usage, and fixed error labels. Only a bounded number of requests are in flight.
429/408 and transient 5xx/connection failures receive bounded retries and backoff.
Finished jobs are skipped on resume. JSONL is deterministically materialized
from the ledger after each invocation. An interrupted provider request may have
been billed even if its response was not committed: this does not promise
provider-level exactly-once billing.
An exclusive local SQLite run lock prevents two generators from consuming the
same directory concurrently; a crashed process releases the lock automatically.

Schema checks reject missing fields, truncated output, wrong batch counts,
incorrect clean/category structure, oversized text, and contact/secret patterns.
Duplicate/conflicting source candidates are rejected. **These checks do not
prove grammatical correctness, complete privacy filtering, diversity, or meaning
preservation.** Keep candidate data private until sampled and reviewed, and check
the provider's generated-output terms before selecting a publication license.
Generated raw data and ledgers are Git-ignored.

## Ready-made public data

| Source | Size/type | Use |
| --- | --- | --- |
| [C4_200M original project](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction) | Broad synthetic corrections built from C4; original repository provides corruption edits | Large training candidate; original edits are CC-BY-4.0, and source-corpus notices also matter |
| [martinsr/c4_200m](https://huggingface.co/datasets/martinsr/c4_200m) | 183,894,319 ready-made `input`/`output` pairs, about 25.9 GB parquet | Stream directly with the supplied importer; mirror declares CC-BY-4.0 and retains original attribution |
| [JFLEG](https://github.com/keisks/jfleg) | Human-written fluency corrections | Separately licensed CC-BY-NC-SA-4.0 evaluation candidate; not permissive commercial training data |

The importer pins the live-verified parquet mirror revision
`c46acd6e099d2bf712cd8d0496cb2ba135e786c8`. It does not execute a remote dataset
loading script and does not download all shards by default.

```sh
uv pip install --python .venv/bin/python -r training/requirements-data.txt
.venv/bin/python training/import_c4.py --limit 10000 --max-scanned 100000
# Add --execute to stream a bounded sample into data/imported/c4-sample.
```

The manifest records revision, attribution, declared license, scan/accept/reject
counts, hashes, and the first-valid-row sampling policy. A prefix sample can be
biased; use multiple reviewed shards or an explicit reservoir/randomized
selection design for representative large runs. Records remain weak candidates,
and the importer refuses to overwrite an existing corpus.

The importer uses synchronous 1024-row Parquet batches with `use_threads=False`,
disables native pre-buffering, and explicitly closes an early-stopped iterator.
Live testing of the high-level `datasets` scanner on Python 3.10/PyArrow 25
materialized the sample but crashed during interpreter finalization with
`PyGILState_Release`. This path avoids the native background-reader lifetime
problem; related reports appear in [Arrow issue 43497](https://github.com/apache/arrow/issues/43497).

## Prepare data for the tiny edit model

`training/prepare_pairs.py` aligns pairs to tags, rejects unrepresentable edits,
deduplicates sources, removes conflicting targets, and derives the vocabulary
from training records. It supports repeated `--input` arguments.

```sh
.venv/bin/python training/prepare_pairs.py \
  --input data/imported/c4-sample/candidates.jsonl \
  --input data/teacher/english-v1/candidates.jsonl \
  --input data/reviewed/english-evaluation.jsonl \
  --allow-weak-train --teacher-license CC0-1.0 \
  --max-labels 4096 --output data/prepared/english-v1
```

Use a teacher license only after verifying output rights; the command above is
an example, not a finding about any unspecified provider. Reviewed records must
carry `review_status: "human-reviewed"` and an allowed license. All unreviewed
pairs are train-only. If a reviewed clean group lands in dev/test, its unreviewed
variants are dropped from training. Without reviewed dev/test examples, the
script prepares train data but reports `evaluation_ready: false`; it does not
manufacture a natural evaluation set from teacher responses.
For private local experiments only, `--allow-unverified-teacher-terms` together
with `--allow-weak-train` retains the unverified license and sets
`publication_allowed: false`. It does not grant rights or make teacher output
reviewed. The exported experimental model retains that publication restriction.

Train/test vocabulary coverage and unsupported-example rejection counts are
reported, because scoring only the representable subset can inflate results.
The finite tag set loses arbitrary rewrites, large insertions, casing changes,
and some punctuation/spacing edits. Audit coverage before choosing a larger
edit vocabulary, morphology tags, or a seq2seq student.

A live verified prefix sample imported 1,000 valid pairs after scanning 1,013
rows. Of those, 378 pairs could be aligned by the current single-pass edit
representation (433 labels); 622 were rejected. This 37.8% prefix coverage is
an architectural finding, not an estimate for the full corpus. Expanding edit
representation or using iterative corrections is a required experiment for
broader training; simply adding millions of pairs will not remove this limit.

The current `train.py` loads encoded splits into memory and is intended for
bounded first-model experiments. Million-scale training should use memory-mapped
tokenized shards and a separate training schedule; the generator/import/preparation
paths already stream records and use disk-backed ledgers.
