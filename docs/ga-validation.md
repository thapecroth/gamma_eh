# GA preparation validation

Preparation verified on 2026-10-01 (America/Los_Angeles), using Chromium
149.0.7827.55 on the shared Linux host. The branch was isolated from the existing
dirty checkout and based on main commit
`6883aaa87a0fa7d1151a829875ff1d95d1872f86`. This report describes local engineering
evidence, not GA approval, a store installation, a production deployment or a
physical-device qualification. The authoritative [readiness record](ga-readiness.json)
remains pending, with quality blocked.

## Integration with default-enabled extension

Public main advanced to `71536e9578ddd038bb7716be690d9061c2b50e47`
while the first contribution's CI was running. The integration preserves
default-enabled Local AI in both apps, saved extension opt-outs, real CPU
fallback and local spelling/rules fallback without WebAssembly. Help, privacy,
GA scope and the bundled model card now describe those defaults. Source
attribution includes the new default/compatibility disclosure; changing that
document invalidates reviewed stable evidence. The agent screenshot selector
also follows the renamed `ai-toggle` scenario.

Fresh `npm run check` passed: 227 app tests, 29 release/readiness tests,
11 harness tests, lint, strict typecheck and all builds. The 12 native editing
checks passed. Rebuilt ZIP checksums verified; the extracted CPU/WebGPU
playground, real browser/model/extension harness and all 12 extension scenarios
passed. Fresh extension activation required no stored AI preference; the harness
verified a model-only correction with UTF-16 offsets and a saved opt-out after
reload. The incompatible-runtime test retained local rule/spelling corrections.
Seven automated accessibility states passed with zero violations. WebGPU used
software SwiftShader; physical-device and manual qualification remain pending.

| Latest rebuilt archive | SHA-256 |
| --- | --- |
| Chrome | `787756e155ac9c323115d70c014c2c226010ff1a8eb2f3a390f3628250ed14ae` |
| Firefox | `1f354435bbe49bc11177540c69904abbdd9c43c00b51ecf48584ae6ad20cefb5` |
| Web | `8e2eec26c64c1a5f27d806036abe0e81b4161e71066293af6055c9ebd8b1809c` |

The earlier snapshots below remain historical. This integration changes neither
the engine's correction policies nor model weights and adds no quality,
physical-device or optimization approval claim.

## First integration with public main

The GA changes were first integrated onto public main
`67d5475c6d53e635b3957a9ed4c090083c0e6907`. The newer underlined writing editor,
default-enabled web Local AI and trusted inference-analysis workflow are
preserved. At that snapshot, Chrome Local AI started off. Privacy, support, release
notes and web ZIP installation instructions now describe those defaults.

Fresh integration checks passed: 226 app tests, 29 release/readiness tests,
11 harness tests, 138 Python tests, 12 native editing checks and the canonical
inference-analysis verification. The latter verifies real CPU sampling,
output parity, rejection of incomplete promotion evidence and interruption/
lock cleanup. Its self-comparison is diagnostic and does not approve an
optimization or physical-device performance claim.

The rebuilt experimental ZIPs were checksum-verified and tested after
extraction. CPU and software-WebGPU playground, actual model/browser checks
and all 12 extension scenarios passed. Seven accessibility states had zero
automated violations, including both default AI and rules-only desktop states.
Underline placement, scrolling and resizing coverage is retained.

| Rebuilt archive | SHA-256 |
| --- | --- |
| Chrome | `5fa616e55d0df66abe69734ad0abaacb5bd9bbd4a5b9ca8264fb4eb44dd6db84` |
| Firefox | `c8443b90c697d93e8feb8b3553d4ded63c249d3cf043618294471f7f32b1a8b8` |
| Web | `1d19b2907eba53dc2880707a1fc1719be5992b6c2840d1a72a02d3106399e32f` |

The remaining sections record the initial preparation snapshot and its measured
quality/performance results. Correction policies and model weights were unchanged
by the integration; no new natural-writing or performance claim is inferred.

## Engineering validation

| Check | Result |
| --- | --- |
| `npm run check` | Passed: lint, strict typecheck, 213 app tests, 29 release/readiness tests, 11 harness tests, web/Chrome/Firefox builds. Lint reports three pre-existing unused-variable warnings. |
| Python training tests | 71 passed using the existing isolated Python environment; no training or teacher calls were started. |
| `npm run test:editing` | 12 passed: textarea/plain-editable native Undo/Redo; cancelled, stale-mutated, newly private, detached and focus-redirected edits fail closed. |
| Extracted release web, CPU and WebGPU | Passed draft recovery, accept/UTF-16/undo, clipboard fallback, offline inference, stale replies, inference failure/retry, temporary drafts, local requests and bundled documentation. |
| Accessibility | Zero automated violations in six states: desktop rules, empty draft, privacy, help, Local AI and 320px layout. Decorative contrast checks remain flagged for manual review; automated checks do not establish screen-reader usability. |
| Extracted Chrome/browser harness | Passed actual WASM/WebGPU inference, 26 regression examples, FP32 parity, injected GPU fallback, UTF-16 edits, sensitive-field exclusions, plain/rich-field handling, site pauses and the unchanged shipping manifest. |
| Extracted extension driver | All 12 mandatory popup/editing scenarios passed, including opt-in AI, stale suggestions, protected payment/private/opt-out fields, UTF-16 and inline interaction. No browser errors or network violations. |
| Pages-path web variant | CPU and software-WebGPU playground checks passed under `/gamma_eh/`, including local policy/help pages and asset paths. |
| Dependencies | `npm audit` reports zero vulnerabilities after upgrading esbuild to 0.28.1. The added accessibility scanner and Markdown parser are development dependencies; no executable dependency is fetched from a CDN. |

The edit-integrity regression initially reproduced a defect: direct field value
and DOM mutations did not preserve native Undo. The extension now uses the
browser's undoable `insertText` command, verifies the text/eligibility/focus
immediately before it, and refuses unsupported commands. This relies on the
deprecated `execCommand` API because it preserves the native edit history;
manual browser/site compatibility remains part of the launch gate. Site-specific
editing handlers and rich editors remain outside the supported plain-field claim.

The web app serves script/runtime/model assets locally, with a production
content policy and no-referrer metadata. The extracted-package harnesses observed
no external requests or draft uploads. This is bounded browser evidence, not a
blanket claim about browser vendors, hosting logs or every website.

## Tested experimental package bytes

`npm run release:package -- --tag v0.1.1` built local diagnostic ZIPs. Their
checksums were verified, then they were extracted into the isolated
`release-smoke` build profile and tested without replacing their contents.
These are working-branch packages, not newly published versions.

| Archive | SHA-256 |
| --- | --- |
| Chrome | `37b49006efea4afc93ff49ff7d9338ec2c23fab87f1ca89b749cbad9d3f4ed40` |
| Firefox | `1f2bdc4e447d8c3e9474b53ea508e36400ed1e65924615a2681a0ea88fa51d42` |
| Web | `7e4f4e747d8cb8937964327b83dbb8d8256694420e5bc91c04006de537bea69e` |

Firefox was packaged and its bundled worker exercised in Chromium fixtures;
this is not a Firefox installation or signing qualification. Chrome was loaded
unpacked, so native store permission prompts, automatic updates and consumer
installation still require actual store evidence.

## Independent natural-writing results

The complete existing frozen JFLEG populations were evaluated with the actual
shipping browser engine: development 754 sentences, test 747. Rules, neural-only
and combined modes ran separately, with one pass, zero inference failures and no
blocked/external requests. No model weights or engine correction policies changed.
The isolated scorer used ERRANT 3.0.2, spaCy 3.8.7, English model 3.8.0 and
RapidFuzz 3.14.5. Its best-sentence-reference token-edit procedure is custom;
these are not official JFLEG GLEU scores or general grammar-accuracy percentages.

| Population/mode | Predicted edits | Edit precision | Edit F0.5 | Accepted clean sources changed |
| --- | ---: | ---: | ---: | ---: |
| Development rules | 349 | 70.49% | 41.43% | 38/216 (17.59%) |
| Development combined | 350 | 70.57% | 41.50% | 38/216 (17.59%) |
| Test rules | 363 | 71.90% | 43.56% | 2/182 (1.10%) |
| Test combined | 363 | 71.90% | 43.56% | 2/182 (1.10%) |
| Development neural-only | 11 | 90.91% | 3.45% | 1/216 (0.46%) |
| Test neural-only | 8 | 87.50% | 2.31% | 0/182 (0%) |

The proposed core quality gate requires at least 95% edit precision, at most 2%
clean-source changes on both populations, and at least 25 predicted edits.
Rules fail precision on both splits and clean preservation on development.
Neural-only evidence is below the minimum edit support and remains experimental.
Human review of meaning preservation has not been signed off.

Spelling edits account for 87/103 development false positives and 85/102 test
false positives. The next quality work should improve conservative spelling
selection using development evidence and fresh held-out qualification; changing
the launch threshold or suppressing failures from the evaluation does not resolve
this blocker. Existing test results are recorded, not used to select new weights.

Population SHA-256:

- Development: `b5581c7aa7dfcbacdf39b41256e3b01e13e85ace621baa766ac2b3868723f3a5`.
- Test: `22504390dc7921e1a4e85d77847b89953965ce385d22b428efec764e2abe6760`.
- Bundled model manifest: `8f2880d15e8289d26aecaaae22d35d94f6f808ef9fb5933239c53da83124c7bf`.

Raw corpora and per-sentence predictions stay in ignored local artifacts. Only
aggregate results are recorded here; the evaluation corpus is not used for
training or bundled in release assets.

## Local performance diagnostics

`npm run benchmark:inference` measured the existing 26-example regression fixture
and repeat/edit workloads on CPU and SwiftShader WebGPU. These are small local
diagnostics, with inference caches, not production p95 latency or physical GPU
measurements.

| Backend | Cold analysis | Unchanged warm median | Edited warm median |
| --- | ---: | ---: | ---: |
| CPU/WASM | 231.6 ms | 0.6 ms | 1.8 ms |
| Software WebGPU/SwiftShader | 952.6 ms | 0.8 ms | 15.3 ms |

No Windows/macOS hardware matrix, physical GPU, peak memory, pilot or approved
performance budget is inferred from these numbers.

## Reproduce and retain evidence

Run heavy jobs sequentially and acquire the shared harness lock before browser
evaluation. Browser runs use fictional text and temporary profiles. The release
workflow now tests CPU/WebGPU web flows, native editing and extension scenarios
against extracted ZIPs; CI retains sanitized browser/release evidence for 90 days.

```sh
npm run check
python -m pytest training -q
npm run test:editing
npm run release:package -- --tag vX.Y.Z
# Extract the ZIPs into the default build directories or an isolated profile.
GAMMA_TEST_WEBGPU=0 npm run test:playground
GAMMA_TEST_WEBGPU=1 npm run test:playground
GAMMA_TEST_WEBGPU=1 npm run test:browser
GAMMA_TEST_WEBGPU=1 npm run test:extension
node scripts/evaluate-browser.mjs --input PATH_TO_JFLEG/dev.jsonl --output artifacts/ga-quality-dev.json --modes rules,model,combined --passes 1
python training/evaluate.py --predictions artifacts/ga-quality-dev.json --evaluation-dir PATH_TO_JFLEG --split dev --scorer errant --output artifacts/ga-quality-dev-errant.json
# Repeat the complete frozen test split; install requirements-scoring.txt in isolation.
```

Stable publication requires reviewed receipts naming the actual release tag,
source/model digest and archive hashes. These local reports intentionally do not
populate passed GA receipts. Store approval, production deployment/recovery,
physical devices, manual accessibility and a representative pilot remain open
in [the checklist](ga-readiness.md).
