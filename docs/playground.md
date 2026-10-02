# Web playground

The web app is the Gamma EH playground: a standalone writing demo that needs no
Chrome extension, account, API key, or inference server. Visitors open the page
and type or paste a draft. Rules and the bundled spelling dictionary run by
default. Local AI is opt-in through its switch or the **Try local AI** example.

## Run locally

With Node 24.11+ (24.x) and dependencies installed:

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite+. The development server serves the
included model from this checkout; no extension build or installation is needed.

## Use the playground

- Choose an email, everyday writing, spelling, or local AI example.
- Edit the draft and review live suggestions. Each card identifies a rule,
  dictionary, or model suggestion.
- Accept or dismiss a change, or accept all current suggestions.
- **Undo correction** restores the draft before the last accepted change or
  group of changes. Editing, clearing, loading an example, or resetting discards
  this one-step undo so it cannot overwrite a newer draft.
- **Reset example** restores the selected example, including dismissed
  suggestions. **Clear** gives you a blank page. **Copy text** copies the draft;
  if clipboard access fails, the page selects the text for manual copying.

Switching examples keeps the current AI preference, except **Try local AI**
explicitly enables it. The page reports the actual WebGPU, WASM CPU, or rules
backend and elapsed check time. If the model fails to load, rule and spelling
suggestions still work and the page explains the fallback.

The bundled model is an experimental synthetic-template baseline. Its example
corrections are demonstrations, not evidence of real-world grammar accuracy.

## Privacy and assets

Checks execute in a browser worker. Typed text is never sent to a server,
analytics, or a remote model. Drafts and the one-step undo stay in memory and
are discarded on reload or closing the tab. Copying explicitly writes the text
to the visitor's clipboard. Fonts and interface assets need no external service.

The initial page and optional model weights are downloaded from the same origin.
After those assets load, fresh checks work even if the connection drops. This
does not claim offline navigation or installation: there is no service worker
or offline app cache.

Every draft or mode change immediately invalidates pending results. The page
also checks the source snapshot before applying corrections; the shared engine
preserves JavaScript UTF-16 offsets, including text containing emoji.

## Build and host

```sh
npm run build
npx vp preview --config apps/web/vite.config.ts --host 127.0.0.1
```

The standard build creates the standalone site in `dist/web`, including its
bundled worker/runtime and `models/` assets. Serve that entire directory on a
static host at the origin root. Serve JavaScript as JavaScript and WASM as
`application/wasm`. Use HTTPS for a public deployment; localhost also supports
the clipboard and WebGPU. The additional extension artifacts are separate;
visitors to the web playground do not use them.

Opening `index.html` through `file://` is unsupported because workers and model
fetches require HTTP(S). No public deployment is created by these build commands.

## Verify without an extension

After building and installing Chromium (`npx playwright install chromium`):

```sh
npm run test:playground
GAMMA_TEST_WEBGPU=1 npm run test:playground
```

Run the jobs sequentially. The first disables WebGPU to prove actual CPU model
inference; the second requires actual WebGPU model inference. Both launch a
fresh browser with extensions disabled and exercise examples, dismiss/reset,
single/bulk acceptance, undo, UTF-16 offsets, clipboard success/failure, clear,
keyboard AI controls, checks while offline, stale replies after mode changes,
model-load failure and retry, draft reload, and narrow/mobile layouts. Every
external HTTP request and upload fails the
test. The test writes `artifacts/playground-smoke.json`, separate CPU/WebGPU
receipts, and desktop/mobile screenshots. The hosted browser lane runs this
alongside the existing full
extension suite; a local pass is separate from hosted CI or deployment proof.
