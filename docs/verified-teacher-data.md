# Recipe-based machine-screened teacher data

`training/verified_teacher.py` generates fictional clean English through the
configured local CLIProxyAPI route and derives single-error variants in code.
Separate blind repair requests and a pair critic screen the labels. Accepted
examples are **machine-verified weak supervision**, not human-reviewed labels.
GLM generates and verifies the examples in fresh requests; its errors can be
correlated across those requests. A human audit and independent natural-text
evaluation remain necessary before claims about quality or model promotion.

The v1 client uses only `http://127.0.0.1:8317/v1/chat/completions` and the exact
model `glm-5.3`. It disables environment HTTP proxies, rejects redirects, and has
no direct provider endpoint or fallback model. Optional monitor attribution
retains only approved provider/auth-mode values. Account identifiers, keys,
monitor tokens, raw headers and provider error bodies are never saved.

## Recipes and family construction

The versioned registry is `training/recipes/teacher-v1.json`. Each recipe names
its construction, category, constraints, examples and deterministic mutator:

| Recipe | Single-error mutation |
| --- | --- |
| Subject agreement | Same-tense be/have or explicit habitual agreement; includes singular heads with plural modifiers |
| Article sound class | Flip an existing `a/an` using closed pronunciation classes |
| Missing countable article | Remove a verified indefinite article before a closed singular countable object |
| Numeric count noun | Remove a regular plural suffix after an explicit plural numeral |
| Direct question auxiliary | Flip initial `do/does`, preserving the lexical base form |
| Perfect participle | Replace an irregular participle with its simple-past form after `has/have/had` |
| Mass/count quantifier | Flip `much/many` with unambiguous closed noun sets |
| Governed preposition | Change the preposition in a closed lexical frame |
| Duplicate function word | Duplicate an approved function word; exclude grammatical `had had`/`that that` |
| General clean controls | Preserve valid informal writing, questions, names, ambiguous tense and subjunctives exactly |

Contexts remain varied; eligibility rules limit the mutation, rather than
declaring all generated prose grammatical. A clean sibling must pass exact
blind identity and the pair critic before its corrupted sibling can be admitted.
An ineligible construction is retained with `recipe_no_eligible_mutation`, while
its valid clean sibling can still be screened. False semantic verdicts and
different repairs go to quarantine; accepted alternative repairs are not guessed.

Every family receives a split before corruption. The surface family hash uses
NFC casefolded lexical tokens, equates straight/curly apostrophes, ignores
punctuation, and collapses numeral values. This groups those surface variants,
not arbitrary paraphrases or related syntactic templates. The stable split is
80% train, 10% dev and 10% test. All siblings share the same split. Synthetic
dev/test remain diagnostics and do not certify natural-text generalization.

## Running a bounded pilot

Supply credentials in environment variables using the existing secure local
configuration loader. Never put them in CLI arguments or dataset files.

```sh
python training/verified_teacher.py \
  --output data/teacher/verified-pilot-v1 \
  --families 80 --batch-size 4 --seed 46 \
  --api-key-env TEACHER_API_KEY \
  --attribution-token-env GAMMA_MONITOR_TOKEN --require-attribution \
  --exclude data/imported/jfleg-evaluation/dev.jsonl \
  --exclude data/imported/jfleg-evaluation/test.jsonl \
  --exclude data/imported/task-exclusions.jsonl \
  --max-requests 100 --max-total-tokens 2000000
```

This is a dry-run. Add `--execute` after inspecting the plan. Reuse the exact
command to resume, or add `--snapshot-only` to export checkpointed progress
without a model call. Exclusion inputs must exist; include every heldout source,
target and reference, including task-specific text which must remain local.
Comparison uses local NFC/casefolded keys. Exclusion texts are never supplied
to the teacher. Recipe examples and prompts are newly authored fictional text.

There are up to four successful requests per finished generation job:
generation, clean-source blind repair, corrupted-source blind repair, and pair
criticism. Clean-only jobs need fewer requests. Generation uses reasoning effort
`low`; repair and criticism use `high`. Requests run sequentially. Request and
token ceilings include persisted attempts across resumes. Retries are bounded;
429 responses honor a bounded `Retry-After`. Unknown billing retains a
conservative reservation of UTF-8 request bytes plus framing and output budget.
Observed successful/malformed response usage is retained separately from that
budget charge; timeout/interruption billing is marked unknown.

## Blinding, checkpoints and outputs

Repair inputs contain only source, register, fictional context and an opaque
ID. Clean and corrupted siblings use separate requests; their family, recipe,
class, mutation and intended target are withheld. A variant batch cannot contain
its intended target as another source. The critic intentionally sees source and
target and returns explicit booleans for grammaticality, necessity, minimality,
meaning and register. False booleans and malformed row verdicts are quarantined.
Informal initial lowercase and other unaffected characters must remain exact.

SQLite stores stage inputs by hash, sanitized attempt receipts, rows and job
progress. A completed assistant response is committed before parsing/advancement,
so a crash after that checkpoint does not repeat the request. A crash during an
uncheckpointed request cannot prove whether the provider billed it; its attempt
is retained as unknown and counts against the bounded retry/budget policy.
Changed recipes, prompts, settings, exclusions or dependency code hashes require
a new directory. A local exclusive lock prevents overlapping invocations.
Fatal route/attribution/redirect/authentication failures stop the entire run
nonzero and preserve its pending stage and receipts. Exhausted ordinary stage
failures are quarantined within the fixed attempt policy; they are not silently
retried forever on subsequent invocations.

Each finished job writes an atomic immutable `shards/job-*.jsonl` screening
receipt. This avoids rewriting the full corpus after each job. Final exports
stream from the ledger, enforce exact-source deduplication, and remove conflicting
labels. A later conflict can revoke admission while the original immutable
screening receipt remains available. Effective status in final exports records
that revocation.

- `candidates.jsonl`: admitted train families only, with clean siblings,
  recipe/version, context, provenance, mutation and screening evidence.
- `diagnostic/dev.jsonl` and `diagnostic/test.jsonl`: admitted synthetic heldout
  families, kept out of the weak training export.
- `quarantine.jsonl`: rejected and unsupported originals with fixed reasons.
  Invalid generation-stage batches remain available in sanitized SQLite response
  receipts when they cannot be safely joined to row IDs.
- `review-sample.jsonl`: up to five admitted clean and five admitted error rows
  per recipe. `review-rejected-sample.jsonl` samples quarantined counterparts.
  These files are audit queues, not a claim that someone reviewed them.
- `manifest.json`: input/code/prompt hashes, output/shard hashes, attempts, usage,
  attribution, accepted/quarantined counts, split counts and recipe coverage.

Mutation spans record both Python Unicode-codepoint offsets and browser UTF-16
offsets. The teacher supplies no authoritative edit offsets. Compatibility checks
separately require exact schema-2 reconstruction and a current decoded-source
roundtrip at oracle confidence `1.0`, with conservative whole-source abstention
for protected spans. Tokenizer context budget and empirical student accuracy are
separate checks. Verified raw pairs are retained even when current guards block
them; coverage limitations must not redefine dataset quality.

All outputs stay local and private, with `provider-terms-unverified` provenance
and publication disabled. User authorization covers the configured CLIProxy use;
it does not turn machine screening into human certification or a release gate.
Adding data should be followed by a controlled development-only student
comparison, clean-text false-positive checks and the existing precision/support
promotion gates. Keep fresh natural holdouts out of generation and selection.

Focused offline validation:

```sh
python -m pytest training/test_verified_teacher.py
```

## Preparing and comparing students

Prepare only the admitted training export. Preparation rejects revoked rows,
synthetic teacher dev/test families, missing screening evidence and attempts to
relabel machine screening as human review. It preserves recipe/family/run
provenance and verifier receipts while retaining the weak-supervision label.

```sh
python training/prepare_pairs.py \
  --input data/teacher/verified-pilot-v1/candidates.jsonl \
  --output data/prepared/verified-pilot-v1 --schema 2 \
  --allow-weak-train --allow-unverified-teacher-terms
python training/train.py \
  --data data/prepared/verified-pilot-v1 \
  --evaluation-dir data/imported/jfleg-evaluation \
  --output artifacts/verified-student-v1/export \
  --checkpoint artifacts/verified-student-v1/checkpoint \
  --development-only --keep-weight 1.0 --scorer errant --local-files-only
```

The pilot is a pipeline diagnostic; a useful training comparison needs a larger
frozen corpus and a matched control. Keep supported row count, label inventory,
base, context, schedule and seeds fixed when comparing data recipes. Compare
KEEP weights separately so their effect is not attributed to dataset quality.
The default KEEP weight remains `0.3`; `--keep-weight` records a finite positive
alternative in the training schedule.

`--development-only` defers all test inference, scoring and parity, including
PyTorch and both ONNX exports. Test inputs/references are read locally for
exclusion checks and population hashes only. Reports explicitly mark test
results as deferred/null and label export metrics as development results.
Provisional manifests disable model edits even when development calibration
qualifies. Freeze a candidate before the final test and actual browser checks;
development improvements and machine-screening acceptance are not release proof.

`python -m pytest training` covers the dataset pipeline and admission. The
optional training environment also exercises the trainer's full control flow
with local test doubles that reject any test inference in development-only
mode. Those two tests skip in the dependency-light CI environment.
