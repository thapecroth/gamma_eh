# End-to-end teacher pilot

Use a small live run to validate transport, resume, alignment, training, export,
and browser integration before spending time or quota on a massive corpus.
The default route for the verified local CLIProxyAPI catalog is
`http://127.0.0.1:8317/v1`, using the exact wire model `glm-5.3-flash`.
Set `TEACHER_API_KEY` securely in the process environment; do not paste it into
commands, commit it, or copy secret-bearing proxy configuration into this project.
No proxy configuration or production routing changes are required.

## Run sequentially

```sh
.venv/bin/python training/generate_llm.py \
  --base-url http://127.0.0.1:8317/v1 --model glm-5.3-flash \
  --pairs 160 --batch-size 20 --concurrency 2 --max-tokens 4096 \
  --retries 1 --timeout 120 --output data/teacher/glm-flash-live-pilot --execute
# Repeat the identical command to verify completed jobs are skipped.

.venv/bin/python training/prepare_pairs.py \
  --input data/teacher/glm-flash-live-pilot/candidates.jsonl \
  --allow-weak-train --allow-unverified-teacher-terms \
  --output data/prepared/glm-flash-weak

.venv/bin/python training/data.py
.venv/bin/python training/assemble_pilot.py \
  --weak data/prepared/glm-flash-weak --max-template-train-rows 5000 \
  --output data/prepared/glm-flash-pilot

.venv/bin/python training/train.py --data data/prepared/glm-flash-pilot \
  --checkpoint models/checkpoints/glm-flash-pilot \
  --output artifacts/glm-flash-pilot/model --epochs 8

npm run check
.venv/bin/python -m pytest training -q
GAMMA_BUILD_PROFILE=glm-flash-pilot \
  GAMMA_MODEL_DIR=artifacts/glm-flash-pilot/model npm run build
GAMMA_BUILD_PROFILE=glm-flash-pilot \
  GAMMA_MODEL_DIR=artifacts/glm-flash-pilot/model \
  GAMMA_TEST_WEBGPU=1 npm run test:browser
```

Preparation refuses to overwrite an existing output. Choose fresh directories
when inputs or settings change. Alternate weights require an explicit build
profile, which isolates both app outputs under `dist/<profile>/` and browser
evidence under `artifacts/<profile>/`. The normal `models/browser/` checkpoint
and `dist/web`, `dist/extension` stay unchanged by the experimental profile.

## What this proves

The bounded student trains on 5,000 original training templates plus representable teacher
examples. Teacher candidates are **train-only and unreviewed**, with unverified
output rights. The model manifest records `publication_allowed: false`; do not
redistribute its data or weights until terms are independently cleared.

Evaluation uses the unchanged original synthetic development/test sets, filtered
only for train-vocabulary coverage. Assembly checks normalized target and source
overlap and drops teacher rows that collide with held-out examples. It reports
coverage, hashes, added teacher counts, and dropped rows. This checks the pipeline,
not natural-language generalization or teacher grammatical correctness.

The browser test loads the newly exported model in the actual web worker and
MV3 offscreen worker. It checks WASM and opt-in WebGPU, neural-only correction,
accept-all, individual extension edits, sensitive-field exclusions, rich-DOM
preservation, pause/resume, and site reenable. Software WebGPU adapters are valid
functional tests, not evidence of physical-GPU latency. Chrome's native optional
permission dialog still requires a separate interactive check.

## Live result on 2026-09-29

The authenticated local catalog listed `glm-5.3-flash`, and four real teacher
requests produced 80 unique candidates (20 each agreement, articles,
prepositions, and clean). Completed jobs recorded 16,623 total tokens. This is
not total billable usage: interrupted requests may also have consumed quota.
The execution session was interrupted by a switch to a restricted sandbox.
Resuming preserved all completed jobs, but its four remaining jobs failed
because host-proxy connections were denied before HTTP (`PermissionError`,
errno 1). The manifest truthfully reports four done and four failed jobs.

Preparation accepted 79/80 examples as train-only weak supervision. A fresh
bounded student used those 79 pairs plus 5,000 original training rows, 79 labels,
and the independent original synthetic evaluation splits. Train-derived
vocabulary filtering rejected seven development and four test examples; 5,832
dev and 6,159 test examples remained. No teacher examples entered either split.

Eight epochs completed on CPU in 75.65 seconds (training/calibration, excluding
export). The model has 4,379,599 parameters, with a 4,465,727-byte INT8 export.
FP32 and INT8 executed over the complete retained test split; FP32 argmax parity
was 100%, INT8 99.9519%. These are synthetic pipeline checks, not natural-language
accuracy. All new data and weights remain Git-ignored and marked non-publishable.

Local checks passed: 18 JavaScript tests, 17 Python tests, the default production
builds, and the isolated pilot builds. The browser verifier checked exact model
hashes in both isolated outputs, then failed at `listen EPERM` while starting its
localhost fixture. It did **not** launch Chrome or prove browser inference for
the fresh student. The earlier shipped baseline's browser checks remain separate
evidence in [Initial validation](validation.md).

After restoring host-network and local-listener access, repeat the identical
generation command to finish the remaining jobs. If the corpus changes, prepare
and assemble into fresh directories, then retrain. To test the already exported
partial pilot immediately, only rerun:

```sh
GAMMA_BUILD_PROFILE=glm-flash-pilot \
  GAMMA_MODEL_DIR=artifacts/glm-flash-pilot/model \
  GAMMA_TEST_WEBGPU=1 npm run test:browser
```

The actual capped training data for this interrupted run lives in
`data/prepared/glm-flash-pilot-smoke`, and its detailed export report is
`artifacts/glm-flash-pilot/model/evaluation.json`. The uncapped assembly is also
retained; neither replaces the baseline.
