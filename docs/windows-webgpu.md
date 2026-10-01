# Windows WebGPU initialization

Chromium on Windows ignores `powerPreference` in `GPU.requestAdapter()` and
records a warning in the extension Errors page. The pinned JAX JS 0.1.25 backend
always supplies `high-performance`; the previous ONNX Runtime 1.30 native loader
also supplies the option. This warning concerns adapter selection, rather than a
failed correction or an account/model-service limit.

`scripts/webgpu-adapter-options.mjs` transforms the locally bundled adapter
request. Windows calls omit only `powerPreference`; feature-level and fallback
options stay intact. Other platforms keep the library's original preference.
Detection uses browser client hints when present, then the legacy platform and
user agent. Empty descriptors fall through to the next platform source.
The transform scans JAX package modules regardless of filename or directory.
It expects exactly one module with one supported request and fails the build
if the pinned library removes, splits, or changes its adapter initialization.
Builds without JAX remain supported.

The Vite plugin covers web worker builds and the esbuild plugin covers extension
workers. Neither modifies `navigator`, filters the console, fetches executable
code, or changes GPU device creation. Existing WASM fallback remains in place.

Older ORT builds can apply the same transform when copying the bundled native
loader. Their expected mixed CPU/GPU graph-placement messages can be reduced
with per-session `logSeverityLevel: 3`; errors and fatal messages stay enabled.
This logging setting suppresses all session warnings, rather than selectively
identifying individual graph-placement messages. Current JAX builds do not
contain ORT or its placement diagnostics.

## Verification and installation

The adapter tests check Windows and other platform descriptors, preservation of
other options, immutability, and failure on upstream drift. Browser verification
captures actual adapter requests while simulating Windows client hints in the
engine probe, the built web and Chrome/Firefox extension workers, and the Vite
development worker (including its dependency optimization path). Each scope
must execute real model corrections with both WebGPU and GPU-unavailable WASM
fallback, with no ignored-preference warnings or uncaught errors. CI runs these
checks through `npm run check` and `GAMMA_TEST_WEBGPU=1 npm run test:browser`.
An upstream runtime change or a build that bypasses the transform fails these
checks instead of silently reintroducing the warning. A Linux browser with
simulated Windows metadata verifies the option passed by the build; the Firefox
worker bundle is also executed in that Chromium harness. These checks do not
establish behavior on a physical Windows GPU or in the Firefox browser.

The release workflow extracts all three packaged apps before running browser
verification, so the same checks cover the workers users receive in ZIP files.

Rebuild and reload the unpacked extension from the rebuilt directory. Previously
recorded Chrome extension warnings stay in the Errors page until **Clear all**
is clicked. Check for newly recorded warnings after enabling Local AI and typing
a fresh sentence. Unpacked extensions do not update from source changes or ZIP
downloads automatically.

References: [Chromium issue](https://issues.chromium.org/issues/369219127),
[ONNX Runtime session logging](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#logSeverityLevel),
and [JAX JS inference](jax-js-runtime.md).
