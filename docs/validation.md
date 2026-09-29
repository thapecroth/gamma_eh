# Initial validation

Verified locally on 2026-09-29, Node 24.18.0, Python 3.10.12, RTX 3070 training GPU.
The model card and committed evaluation JSON report exact model/dataset evidence.

- `npm run check`: strict TypeScript, 15 focused engine/extension tests, and production web/extension builds passed.
- `.venv/bin/python -m pytest training -q`: corpus reconstruction/split isolation, teacher resume/failure/retry/protocol checks, and bounded import/held-out isolation tests passed.
- `npm run test:browser`: actual WASM model executed in web and extension workers; accept-all, keyboard AI toggle, mobile layout, textarea edits, plain contenteditable edits, sensitive-field exclusions, and rich-DOM preservation passed.
- `GAMMA_TEST_WEBGPU=1 npm run test:browser`: actual WebGPU inference and a neural-only correction passed, including MV3 offscreen inference, pause/resume, and site reenable. Adapter was Google's **SwiftShader software adapter**; no physical-GPU performance claim is made.
- Browser runs made no external network requests and had no uncaught page exceptions.
- The natural 15-case smoke set matched 14/15 expected outputs. One subject/verb case was missed; this is not a quality benchmark.
- The extension fixture grants localhost in a temporary test-only manifest copy. The shipping manifest has only optional site permissions. Chrome's native permission dialog was not automated.
- The pinned C4 mirror streamed 1,000 valid pairs from 1,013 rows and exited normally after synchronous-reader repair. Preparation reports its limited edit alignment coverage and correctly leaves dev/test empty for unreviewed data.
- A million-pair teacher generation dry run planned 50,000 requests without calling any model endpoint.

Browser screenshots and detailed smoke results are produced under Git-ignored
`artifacts/`. See [model limitations](../models/MODEL_CARD.md) and
[larger dataset workflow](massive-dataset.md) before interpreting scores.
