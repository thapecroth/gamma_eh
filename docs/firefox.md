# Firefox extension

Run `npm ci` and `npm run build` with Node 24. The Firefox bundle is in
`dist/firefox/` (or `dist/<profile>/firefox/` for an experimental model).
It includes the model, JAX JS, and all executable code locally.

## Temporary installation

1. Use Firefox 140 or newer. Open `about:debugging#/runtime/this-firefox`.
2. Choose **Load Temporary Add-on**, then select `dist/firefox/manifest.json`.
3. Open or reload an ordinary HTTP/HTTPS tab. Suggestions start automatically.
4. Focus a textarea or supported plain-text contenteditable. Accept or dismiss
   suggestions, or use the popup to pause checking globally or on the current site.
5. Experimental Local AI is on by default, with local CPU or rules fallback
   when needed. The popup can disable it; saved opt-outs are respected.

Temporary add-ons are removed when Firefox restarts. Permanent installation in
standard Firefox requires Mozilla signing; this project does not yet publish a
signed Firefox add-on. Internal browser pages and Mozilla-protected sites cannot
be checked. The same private-field exclusions and stale-text guards apply as in Chrome.

## Architecture

Firefox Manifest V3 uses a nonpersistent background event page. That page starts
the bundled inference worker directly. Chrome retains its service worker and
an offscreen document. Both use the same worker request broker and checker.
The Firefox build maps `chrome` API references to Firefox's promise-based
`browser` namespace. The manifest is derived from the Chrome source, removes
`offscreen`, sets a stable Gecko ID, and declares no data collection.

Worker requests have bounded timeouts and validate sender identity, top-level
frame, host permission, and global/per-site pause settings before inference. Text is never
sent to a server. Synthetic evaluation scores are not real-world accuracy.

See [Mozilla background documentation](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json/background).

## Verification

Firefox 156.0.1 was tested with geckodriver 0.37.1 in an isolated headless
profile. A temporary copy of the built add-on was granted only localhost access,
matching the Chrome fixture strategy; the shipping manifest was unchanged.
A real content-script message reached the Firefox background event page and
bundled worker, producing rule suggestions and JAX WASM inference. Requests
from an extension-page tab without a website origin were denied. This verifies
local worker execution; Firefox permission dialogs, inline editing UI, and WebGPU
remain separate coverage gaps.
