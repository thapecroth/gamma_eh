# Firefox extension

Run `npm ci` and `npm run build` with Node 24. The Firefox bundle is in
`dist/firefox/` (or `dist/<profile>/firefox/` for an experimental model).
It includes the model, JAX JS, and all executable code locally.

## Temporary installation

1. Use Firefox 140 or newer. Open `about:debugging#/runtime/this-firefox`.
2. Choose **Load Temporary Add-on**, then select `dist/firefox/manifest.json`.
3. Open the popup, choose **Enable on this site**, allow access, and reload the tab.
4. Focus a textarea or supported plain-text contenteditable. Accept or dismiss
   suggestions, or use the popup to pause checking globally or remove access for the current site.
5. Local AI is experimental and off by default; enable it in the popup to try it.

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
frame, host permission, and registered site access before inference. Text is never
sent to a server. Synthetic evaluation scores are not real-world accuracy.

See [Mozilla background documentation](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json/background).
