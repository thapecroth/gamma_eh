# JAX JS inference

The web editor and Chrome MV3 extension share a local JAX JS inference engine.
`@jax-js/jax` 0.1.25 compiles array operations to WASM or WebGPU;
`@jax-js/onnx` 0.1.2 loads the existing trained ONNX graph into JAX functions.
No retraining or weight conversion is required. Python training and export stay
in `training/`.

## Execution and fallback

`packages/engine/src/jax-runtime.ts` initializes WASM and optionally WebGPU. It
loads the FP32 `model.onnx` on the selected device and JIT-compiles the forward
pass for sequence shapes rounded up to multiples of eight. Padding is masked
out and removed from returned logits. A bounded per-model raw-logit cache reuses
unchanged windows while decoding edits against the current source. See
[inference performance](inference-performance.md) for bounds and measurements.
GPU startup checks actual graph execution, including
kernel compilation. Missing adapters, allocation failures, and startup kernel
failures fall back to the same FP32 model on WASM. Later inference errors surface
through the existing rules fallback and model warning.

Each session places its inputs and weights on its own backend. ONNX int64 token
indices become JAX int32 indices, which represent the entire WordPiece vocabulary
exactly. JAX operations consume array references; the wrapper cleans up any
remaining inputs/outputs on failure and retains the model and compiled functions
for repeated checks. Synchronous tracing restores the previous default device,
so forced WASM checks can coexist with GPU sessions.

The runtime and ONNX loader are bundled into the inference workers by Vite and
esbuild. JAX emits WASM kernels locally rather than fetching a separate runtime.
There are no runtime CDN imports or inference endpoints. The extension retains
its bundled-only CSP, including `wasm-unsafe-eval`, and uses an offscreen document
with a dedicated worker. Neither app requires cross-origin isolation; WASM runs
without SharedArrayBuffer on these origins.

## Model format and size

Both devices execute the reviewed FP32 graph: 17,552,569 bytes for the public
baseline, compared with the previous 4,463,879-byte INT8 CPU graph. The JAX loader
does not implement the quantized graph's `DynamicQuantizeLinear` and
`MatMulInteger` operators. `model_quantized.onnx` remains an export/evaluation
artifact and stays in the reviewed model bundle, but is no longer executed by
the apps. Removed ONNX Runtime JS/WASM assets offset part of the package size.

Tokenization, UTF-16 offsets, confidence threshold, rule priority, protected
spans, edit guards, and stale-text checks keep the existing behavior. Synthetic
classifier scores still describe the exported model and do not establish
real-world grammar accuracy or JAX performance.

## Validation

Run the standard checks and real browser paths sequentially:

```sh
npm run check
GAMMA_TEST_WEBGPU=1 npm run check:environment
GAMMA_TEST_WEBGPU=1 npm run test:browser
GAMMA_TEST_WEBGPU=1 npm run test:extension
```

`tests/fixtures/onnx-reference.json` contains independent FP32 logits recorded
with ONNX Runtime Web 1.30.0 WASM before the migration. It pins the model SHA-256
and token IDs at sequence lengths 8, 10, and 64. Unit checks require maximum
absolute logit error below 0.0001 on WASM. Browser checks compare repeated WASM
and WebGPU runs against that reference with a 0.0005 tolerance and exact argmax
agreement, and require all 26 grammar regression corrections on each backend.
The regression witness includes a model-origin correction beyond the rules.

Browser checks also inject missing GPU adapters, GPU allocation failures, and
kernel compilation failures to require actual WASM model fallback. Extension
checks exercise the worker pipeline, AI opt-in, popup controls, UTF-16 edits,
stale suggestions, protected fields, and rich-DOM preservation. External network
requests and uncaught browser exceptions fail validation. Headless WebGPU may
use SwiftShader software; adapter evidence must accompany any GPU claim.

## Packaging

Both outputs include `inference-runtime.json` identifying the pinned libraries
and FP32 asset. The release packager validates installed library versions and
built runtime metadata, rejects leftover external runtime directories, and
includes JAX MIT and Protocol Buffers Apache/BSD notices. Reviewed model hashes
and the existing publication restrictions continue to apply.

See [browser verification](browser-testing.md), [releases](releases.md), and the
[upstream JAX JS loader](https://github.com/ekzhang/jax-js/tree/main/packages/onnx).
