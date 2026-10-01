# Inference performance

The shared JAX JS engine avoids repeated inference on unchanged sentences and
reduces compilation when a draft changes token length. Model weights, edit
policies, sentence boundaries, and the 64-token baseline context remain the same.

## Measured bottlenecks

The 2026-10-01 Chromium profile of main `310f10a` showed that model execution
consumed about 95% of a warm 24-sentence check on WASM. Editing one sentence still
ran all 24 windows. JAX also specialized its compiled graph for every sequence
length: first visits to nearby lengths were much slower than repeated runs.

Two changes address those costs:

- Each loaded model keeps an LRU of validated raw logits keyed by the exact token
  ID sequence. Unchanged windows reuse inference across drafts and correction
  passes. `modelRuns` counts calls actually made to the model, excluding hits.
- The runtime pads inputs to the next multiple of eight with a zero attention
  mask for padding. Nearby lengths share compiled kernels; at most seven extra
  positions are computed. Returned logits and dimensions exclude padding.

The cache holds at most 512 entries and 4 MiB of Float32 data plus UTF-16 key
bytes, excluding JavaScript object overhead. Entries larger than the budget are
not retained. It is scoped to the loaded model and backend, lives only in worker
memory, and is never written to storage or sent over the network. Session
loading behavior remains unchanged.

Only predictions are reused. Every check recalculates current source words,
UTF-16 offsets, casing, confidence thresholds, full-source safety guards, rule
priority, and insertion snapshots. Protected windows and rule deletions are
checked before reuse. Failed, malformed, and nonfinite outputs are not cached.
Runtime arrays are copied into the cache so disposal or later buffer mutation
cannot corrupt retained predictions.

## Before and after

Measured with Chromium 149.0.7827.55, JAX JS 0.1.25, ONNX loader 0.1.2, and the
same public FP32 model SHA-256
`cb3280db7b33a7273ae75e87c8c5d432c63f9bdac89101d93dda72c776b38266`.
Both versions ran sequentially in separate browser contexts on the same host.
The paragraph contains 24 distinct sentence windows; nine checks per warm
condition measure unchanged text and changes to the final sentence.

| Condition | WASM before | WASM after | WebGPU before | WebGPU after |
| --- | ---: | ---: | ---: | ---: |
| Unchanged paragraph, median | 22.5 ms | 0.6 ms | 362.8 ms | 0.8 ms |
| One sentence changed, median | 22.9 ms | 1.8 ms | 353.3 ms | 14.5 ms |
| First visits to lengths 8–16 after paragraph warmup, median | 29.2 ms | 1.2 ms | 205.9 ms | 13.3 ms |
| Initial paragraph including model load | 237.6 ms | 236.6 ms | 984.6 ms | 1024.5 ms |

Unchanged checks made zero runtime calls. Editing one sentence made one call
instead of 24: approximately 13 times faster on WASM and 24 times faster on the
tested WebGPU adapter. Graph reuse improves visits to nearby lengths, rather
than eliminating the first compilation of a bucket. Initial model loading did
not improve; padding adds computation to new windows. These are single-host
measurements and exclude UI debounce and worker messaging.

The WebGPU adapter was **SwiftShader software**, with `isFallbackAdapter: true`.
These timings do not establish hardware GPU performance. Do not select a global
backend or claim hardware latency from this comparison.

Both versions produced the same edits and corrected text for all 26 grammar
smoke cases and the paragraph workload on each backend. Fifteen selected input
lengths spanning 3–64 tested padding boundaries; output shapes and edit argmax
matched, with maximum FP32 logit differences of 0.00000477 on WASM and 0.00000382
on WebGPU. Confidence differences stayed below 0.0001 in the compared edits.
This checks numerical and behavioral parity for the tested inputs, and is not
a measure of real-world grammar accuracy.

## Reproduction

Run heavy commands sequentially. Use a separate unchanged worktree for the
reference; install the pinned dependencies there or link the matching local
installation. The benchmark uses only locally served model assets and fictional
text, rejects external requests and model fallback, and records model/source
hashes, browser version, adapter evidence, timings, and parity results.

```sh
npm run benchmark:inference
GAMMA_BENCHMARK_BASELINE=/path/to/unchanged-worktree npm run benchmark:inference
GAMMA_TEST_WEBGPU=1 GAMMA_BENCHMARK_BASELINE=/path/to/unchanged-worktree npm run benchmark:inference
```

Results are written to `artifacts/inference-benchmark-wasm.json` and
`artifacts/inference-benchmark-webgpu.json`. `GAMMA_MODEL_DIR` and
`GAMMA_BUILD_PROFILE` follow the existing isolated-build rules. A disabled
candidate model cannot produce an inference benchmark.

Validation includes the standard app checks, focused cache failure/safety tests,
independent ONNX reference logits, and actual browser/extension worker checks.
See [JAX JS inference](jax-js-runtime.md) and
[browser verification](browser-testing.md) for those commands and limits.
