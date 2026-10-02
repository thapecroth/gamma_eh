# Local AI defaults and compatibility

The web playground and Chrome/Firefox extensions enable Local AI automatically.
No popup activation or AI opt-in is required. The extension uses this default
when its `useAI` preference is absent or malformed; an explicitly saved `false`
remains off, including after an update. The web preference lasts for the current
page session and resets to enabled on reload.
Users can turn the experimental model off at any time.

Compatibility is determined by executing the bundled runtime and model, rather
than guessing from a browser name, CPU count, or reported memory:

1. Initialize local WebAssembly and try WebGPU. The GPU path runs a model probe
   before reporting an active GPU backend.
2. If WebGPU is unavailable or cannot initialize or execute the probe, use the
   same bundled model with local WASM CPU inference.
3. If the local model cannot load or run, return spelling and rule suggestions
   with a model-unavailable notice. The UI reports the actual rules backend.

A missing GPU does not disable Local AI when CPU inference works. Failures do
not overwrite the user's AI preference, and model asset failures can be retried.
Empty drafts require no model inference. All checks stay on the device;
extension assets are bundled and no remote inference service is used.

Default enablement does not change the model's experimental status. Synthetic
training and smoke checks do not establish real-world grammar accuracy. Users
still review suggestions, and source-offset and stale-draft safeguards remain.

## Verification

`npm run check` includes preference parsing tests that cover unset preferences,
malformed values, and saved opt-outs. After building, `npm run test:browser`
checks automatic model execution in both apps, a neural-only extension
correction before any AI preference is saved, an opt-out after navigation, real
CPU fallback without WebGPU, and spelling/rules fallback without WebAssembly.
`GAMMA_TEST_WEBGPU=1 npm run test:browser` additionally requires actual WebGPU
execution and exercises GPU allocation and kernel failures. Run these jobs
sequentially. `npm run test:extension` checks the popup's default-enabled state
and both AI toggle directions through the supervised extension driver.
