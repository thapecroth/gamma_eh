# Trusted inference analysis and bounded optimization

`npm run analyze:inference` produces repeatable browser timing, actual sampled CPU
profiles, offline flamegraphs, elapsed wall traces, quality parity, and a machine
promotion decision. It executes the real bundled engine and each checkout's own
model assets on local Chromium. It does not train, modify, publish, or merge code.

Run `npm run test:inference-analysis` to verify the infrastructure itself in real
Chromium with fictional inputs. CI verifies profiles, flamegraphs, elapsed traces,
cache counters, output parity and refusal to promote without the frozen corpus.
It also rejects invalid options and alternate-model variables, and verifies that
interrupting an active analysis invalidates its receipt and releases the lock.
CI preserves fictional aggregate reports and profiles; private corpus outputs,
exception logs and model bundles are excluded from uploaded evidence.

## Start with a trusted harness

Keep a clean, committed checkout of this infrastructure separate from the candidate
you optimize. Record its commit and `report.harness.sha256`; invoke its script by
absolute path. An optimizer may edit engine performance code in the candidate,
but must not change the trusted harness, workloads, thresholds, frozen corpus,
tokenizer, model assets, correction policy or dependency lock to win a comparison.
The gate checks exact model/policy assets, tokenizer/guard sources, dependency
lock and actual bundled dependency bytes across variants. Full source stamps and
every esbuild input are checked again after execution. This detects accidental
drift; it is not a sandbox for hostile arbitrary candidate JavaScript.

Use Node 24 and the existing project dependencies and Chromium. Import the pinned
JFLEG corpus once using `python3 training/import_jfleg.py --execute --output
data/imported/jfleg-evaluation`. The corpus is evaluation only and separately
licensed; retain it locally in ignored directories. A run without it remains a
diagnostic and cannot promote a candidate.
Clear `GAMMA_MODEL_DIR` and `GAMMA_BUILD_PROFILE`: this performance-only gate
requires each checkout's bundled `models/browser` assets.

```sh
# Run from the trusted checkout. Baseline/anchor checkouts are read only.
npm run analyze:inference -- \
  --candidate /absolute/candidate \
  --baseline /absolute/incumbent \
  --anchor /absolute/original-main \
  --quality-dir /absolute/jfleg-evaluation

# Run the backend you intend to improve. Software adapters are labeled explicitly.
npm run analyze:inference -- --candidate /absolute/candidate \
  --baseline /absolute/incumbent --anchor /absolute/original-main \
  --quality-dir /absolute/jfleg-evaluation --webgpu

# Before shipping the selected candidate, add the untouched test split.
npm run analyze:inference -- --candidate /absolute/candidate \
  --baseline /absolute/incumbent --anchor /absolute/original-main \
  --quality-dir /absolute/jfleg-evaluation --include-holdout
```

The default output is a unique private directory under the trusted checkout's
ignored `artifacts/inference-analysis/`. `--output artifacts/UNIQUE` accepts a
fresh directory only. `report.json` and `report.html` contain aggregate evidence,
hashes and fictional workloads. `*-quality.private.json` contains corpus outputs
and raw logits; never commit or publish it. Engine bundles/maps also remain local.
The harness uses the existing shared repository lock across worktrees, refuses
overlap, closes browser contexts on errors and execution deadlines, and releases
its lock. Unexpected failures produce an invalid receipt and nonzero exit.
Signals also close the browser and invalidate the run. A machine crash or forced
kill can leave the shared lock: inspect its owner and confirm that process stopped
before removing it. Browser exception details stay in `error.private.log`.

## Interpret timing and flamegraphs

Each of five default paired trials starts a fresh browser context, loads the
model, and warms the fixed paragraph. Variant order alternates to reduce ordering
bias. Each workload records 25 unprofiled observations, its median and p95, and
actual model/cache/tokenizer counters. The versioned fictional fixture has 24
distinct tokenizer windows. It covers:

- Cold model startup plus the initial 24-sentence paragraph, measured separately.
- An unchanged warm paragraph, with zero model calls when the cache is enabled.
- A new final-sentence edit on every iteration, changing exactly one token window.
- A fresh 24-window paragraph on every iteration, with no cached windows.
- Warm nearby raw inference lengths 8–16, bypassing sentence caching.

The warm objective weights unchanged/edit/fresh by 0.1/0.6/0.3. It uses a weighted
geometric ratio of paired trial medians. A deterministic paired bootstrap resamples
whole trials 10,000 times; promotion requires the 95% lower bound of latency
reduction to exceed 5%. A median regression above 15% in any workload, including
cold startup or nearby shapes, rejects the candidate. A median paired-trial p95
regression above 25% in any warm workload also rejects it. Both guard families
apply against the immutable anchor. These are initial engineering
thresholds, not universal product workload frequencies. Review a policy change
independently and freeze the new version before another optimization campaign.

CPU profiling occurs after timing, in separate contexts. Each warm capture runs
for at least 500 ms (750 ms by default), has at least 100 samples, and emits:

- A genuine CDP `.cpuprofile`, importable in Chrome DevTools Performance.
- An offline standalone `.svg` flamegraph and an embedded copy in the HTML report.
- A `.trace.json` elapsed timeline, importable in Chrome/Perfetto, and an HTML preview.

Cold startup gets its own capture before the first model call. CPU frame width
counts sampled renderer/main-thread CPU, including garbage collection. GC is also
reported separately; idle, program and unsampled wall time are excluded from width
and preserved in the raw profile.
Inputs and fixture validation are prepared before captures; profiling cycles
bounded precomputed edit/fresh plans that exceed the cache capacity. Captured
inputs and timed outputs receive separate exact parity checks. Sparse active CPU
samples are explicitly labeled as insufficient for reliable hotspot ranking.
Profiler function symbols and source-map file/line references are retained, with
native/WASM frames shown as reported by Chromium. It does not measure device GPU
execution or other process/worker CPU. A narrow CPU graph beside a long
`JaxSession.run` wall span can indicate async waiting. Runtime spans include await
time and must not be described as device CPU time. Profile overhead never decides
acceptance; only the earlier unprofiled trials do.
The raw profiles follow the [CDP Profiler protocol](https://chromedevtools.github.io/devtools-protocol/tot/Profiler/).

Node, Chromium, Playwright, esbuild/runtime versions, actual model/bundle/input
hashes, commits, dirty scopes, adapter/software evidence, host load, Linux PSI and
swap counters accompany every report. Host pressure or active swapping makes
promotion inconclusive. Load at half the logical CPU count, memory full PSI avg10
above 1%, or I/O full PSI above 5% triggers this guard. Rerun sequentially on an
idle host; do not parallelize builds, profiling, or evaluation. Timing uncertainty
also yields an inconclusive result. Increase trials/repetitions within bounded
CLI limits when justified; never waive quality or environment checks.
The adapter is captured from the engine's actual initialization request, including
effective request options, and must remain identical across comparison contexts.
`performance` retains the timing assessment for diagnostics; `decision` is the
only promotion result and includes environment and quality requirements.

Profile input plans cycle beyond the shipping cache capacity and validate actual
miss counts. A changed cache capacity can invalidate that profile workload: revise
and independently review the trusted workload before measuring that campaign.

## Quality and machine decisions

Both variants must use identical enabled FP32 weights, labels, vocabulary and
manifest. Bundled 26-case regression and workload outputs must match exactly;
all edit fields match except finite confidence differences below `1e-4`.
Direct runtime probes at sequence boundaries 3–64 require correct output shape,
finite logits, identical argmax and maximum absolute difference below `5e-4`.
Promotion also requires every one of the frozen 754 JFLEG development sentences
in model/combined mode with one/two passes: 3,016 paired output comparisons.
The importer revision, population and exact JSONL hash are pinned; a reduced or
replacement corpus cannot pass. `--include-holdout` additionally checks all 747
test sentences and records a final-confirmation selection. Do not repeatedly
search on the holdout. Parity demonstrates preserved observed behavior, not
universal accuracy; the model remains an experimental synthetic-training baseline.

| Exit | `decision.status` | Meaning |
| --- | --- | --- |
| 0 | `accepted` | Complete parity and statistically supported gain; eligible to review. |
| 1 | `invalid` | Bad assets/input, fallback, regression/quality mismatch, drift, browser failure, or malformed evidence. |
| 2 | `rejected` | Performance insufficient or a workload/anchor regressed. |
| 3 | `inconclusive` | Missing baseline/corpus, pressure, too few trials, or noisy speedup. |

An accepted result is not PR, merge, deployment, or general quality proof. Read
`decision.reason`, `parity`, `trials`, provenance and flamegraphs before acting.

## Codex `/goal` hill climb

Give a new agent the trusted harness checkout, immutable original-main anchor,
current incumbent, writable candidate checkout, corpus path, target backend,
iteration/time budget and permitted engine paths. Start with a diagnostic capture
and one measured hypothesis. For each bounded iteration:

1. Change only the candidate's permitted performance code. Preserve weights,
   correction policy, source offsets, stale-text checks and local-only inference.
2. Invoke the trusted harness against the incumbent **and the immutable anchor**.
   Trust only a fresh exact-source `accepted` receipt; invalid, rejected and
   inconclusive results do not advance the incumbent.
3. Inspect the CPU and elapsed graphs and per-workload counters. Record the
   hypothesis, patch, report path/hash and decision in a campaign log. Retain
   failed trials as evidence; do not select a lucky rerun or redefine workloads.
4. If accepted, commit the candidate and advance the incumbent to that exact
   source; keep the original-main anchor fixed throughout the campaign. Anchor
   guards prevent allowed per-step regressions from accumulating unnoticed.
5. Stop at the stated budget. Evaluate the chosen result once with holdout, run
   `npm run check` and `npm run test:browser` sequentially, review, document and
   open a Conventional Commit PR. Merge or deploy only under user authorization.

Re-run after every source/dependency change. A receipt belongs only to its recorded
tree and backend. This infrastructure provides a measurable acceptance boundary
for automated search; it deliberately does not launch an unrestricted self-editing
loop or silently merge the result.

Use this starter goal with paths and a concrete budget filled in:

```text
/goal Optimize WASM inference within [budget]. Follow docs/inference-analysis.md.
Trusted harness: [path and commit]; anchor: [original main]; incumbent: [path];
candidate: [writable path]; quality corpus: [path]. Keep the trusted harness and
anchor fixed. Measure one hypothesis per iteration, advance only on a fresh
accepted decision, and log every trial. Preserve model behavior. Finish with
holdout, app/browser checks, review, documentation and a PR.
```
