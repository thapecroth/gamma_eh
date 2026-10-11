# Web playground

**[Open the public playground](https://thapecroth.github.io/gamma_eh/)** — no
installation required. See [GitHub Pages deployment](github-pages.md) for the
hosting workflow and live verification.

The web app is the Gamma EH playground: a standalone writing demo that needs no
Chrome extension, account, API key, or inference server. Visitors open the page
and type or paste a draft. Local AI starts enabled alongside rules and the bundled
spelling dictionary. Its switch turns the experimental model off or back on;
**Try local AI** also enables it. If the model cannot run, spelling and rule
suggestions continue automatically. See [local AI defaults](local-ai-defaults.md).

The footer opens bundled `privacy.html` and `help.html` pages in a new tab, so
reading them preserves the draft. The build renders these pages from
`docs/privacy.md` and `docs/support.md` using a build-only Markdown dependency;
it ships no document parser or external runtime. Both pages share the app's
local content policy, work under the configured base path, and are included in
release ZIPs. Other project reference links open the public repository.

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
- Red wavy underlines locate suggested corrections directly in your draft.
  Accepting or dismissing a suggestion removes its mark. Typing clears old marks
  immediately while the checker refreshes.
- Accept or dismiss a change, or accept all current suggestions.
- **Undo last change** restores the draft before the last accepted correction,
  Clear, Reset example, or example selection. New typing invalidates that
  recovery snapshot. Reloading or closing the tab loses the draft. It does not
  provide a persistent document history. Accept all records the draft before
  that entire group of changes.
- **Reset example** restores the selected example, including dismissed
  suggestions. **Clear** gives you a blank page. **Copy text** copies the draft;
  if clipboard access fails, the page selects the text for manual copying.

Switching examples keeps the current AI preference, except **Try local AI**
explicitly enables it. The page reports the actual WebGPU, WASM CPU, or rules
backend and elapsed check time. If the model fails to load, rule and spelling
suggestions still work and the page explains the fallback.

Rules and spelling run in their own lightweight worker and appear while AI is
loading. The AI runtime is imported only by the separate AI worker; the editor
and rules worker do not initialize JAX. First use downloads about 18 MB of model
weights, with received bytes shown during the download. Switching AI off
immediately terminates its pending work, independently of the 450 ms typing
debounce, so a stalled download cannot block local rules or a quick off/on retry.

Download progress refreshes a 30-second inactivity deadline, so slow transfers
can take longer than 30 seconds as long as bytes keep arriving. A stalled download
leaves rules available and displays a retry notice. Once the weights arrive, an
AI worker that stalls for 30 seconds is terminated and retried once in a fresh
CPU worker, using the latest draft. A second 30-second timeout leaves rules
available. These are recovery deadlines, not speed guarantees. The timer lives
on the page so it also handles blocked worker compilation. Typing cannot extend
a stalled operation's deadline, and terminated-worker replies are ignored.
Successful CPU recovery remains selected for that tab until AI is switched off.
Model failures suspend automatic retries until the visitor switches AI off/on.

The bundled model is an experimental synthetic-template baseline. Its example
corrections are demonstrations, not evidence of real-world grammar accuracy.

## Privacy and assets

Checks execute in browser workers. Typed text is never sent to a server,
analytics, or a remote model. Drafts and the one-step undo stay in memory and
are discarded on reload or closing the tab. Copying explicitly writes the text
to the visitor's clipboard. Fonts and interface assets need no external service.

The initial page and bundled model weights are downloaded from the same origin.
After those assets load, fresh checks work even if the connection drops. This
does not claim offline navigation or installation: there is no service worker
or offline app cache.

Every draft or mode change immediately invalidates pending results. The page
also checks the source snapshot before applying corrections; the shared engine
preserves JavaScript UTF-16 offsets, including text containing emoji.

The editor keeps a native textarea for typing, selection, and clipboard behavior.
A transparent, non-interactive mirror draws underlines from the current
suggestions' UTF-16 ranges without changing draft text. Both layers share font,
padding, and wrapping styles; scrolling and resizing synchronize the mirror to
the textarea's content area, including its scrollbar width. Insertions with no
source characters remain available in their suggestion cards.

## Build and host

```sh
npm run build
npx vp preview --config apps/web/vite.config.ts --host 127.0.0.1
```

The standard build creates the standalone site in `dist/web`, including its
bundled worker/runtime and `models/` assets. Serve that entire directory on a
static host at the origin root, or set `GAMMA_WEB_BASE` for a repository subpath
as described in [GitHub Pages deployment](github-pages.md). Serve JavaScript as JavaScript and WASM as
`application/wasm`. Use HTTPS for a public deployment; localhost also supports
the clipboard and WebGPU. The additional extension artifacts are separate;
visitors to the web playground do not use them.

Opening `index.html` through `file://` is unsupported because workers and model
fetches require HTTP(S). Local build commands do not publish the site; checked
main-branch pushes publish it through GitHub Actions.

## Verify without an extension

After building and installing Chromium (`npx playwright install chromium`):

```sh
npm run test:playground
GAMMA_TEST_WEBGPU=1 npm run test:playground
npm run test:web-startup
# Install WebKit first: npx playwright install webkit
GAMMA_STARTUP_BROWSER=webkit npm run test:web-startup
```

Run the jobs sequentially. Startup tests hold a model download open while editing
and accepting rules, cancel/retry AI, run actual model corrections offline, and
hang GPU adapter initialization (Chromium) or hold the preferred worker request
(WebKit) to prove recovery using actual CPU inference.
WebKit on Linux provides engine coverage; it does not replace a physical Mac test.
The first playground command disables WebGPU to prove actual CPU model
inference; the second requires actual WebGPU model inference. Both launch a
fresh browser with extensions disabled and exercise examples, dismiss/reset,
single/bulk acceptance, undo, UTF-16 offsets, clipboard success/failure, clear,
default-enabled AI, red underline positions and wrapping, textarea scrolling and
resizing, keyboard AI controls, checks while offline, stale replies after mode changes,
model-load failure and retry, draft reload, and narrow/mobile layouts. Every
external HTTP request and upload fails the
test. The test writes `artifacts/playground-smoke.json`, separate CPU/WebGPU
receipts, and desktop/mobile screenshots. The hosted browser lane runs this
alongside the existing full
extension suite; a local pass is separate from hosted CI or deployment proof.
