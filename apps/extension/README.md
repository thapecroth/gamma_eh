# Chrome extension

Run `npm install` and `npm run build` at the repository root. The build bundles
the runtime, model, and extension scripts into `dist/extension/`.

1. Open `chrome://extensions` in a recent Chrome version with offscreen-document
   support (Chrome 116 or newer).
2. Enable **Developer mode**, choose **Load unpacked**, and select
   `dist/extension/`.
3. Open a website, select the Gamma EH toolbar button, and choose **Enable on
   this site**. Chrome requests access to that site only.
4. Focus an ordinary textarea or plain-text contenteditable field. Suggestions
   appear in a small panel. Choose **Accept** or **Dismiss**; close the panel to
   pause that field.

The popup can pause suggestions everywhere, disable experimental local AI, or
remove a site's permission. Password inputs, payment/authentication fields,
fields marked `spellcheck="false"`, `data-private`, `data-sensitive`, or
`data-gamma-ignore`, and unsupported rich editors are excluded. Iframes are not
checked. The extension limits each checked field to 6,000 UTF-16 code units.

Typed text is sent only between the extension's content script and its bundled
inference worker. It is not stored or transmitted to a server. Permissions and
checker preferences are stored locally. A tiny model trained on synthetic data
is an experimental baseline, not evidence of real-world grammar accuracy.

The extension checks the original text and offsets before accepting a
suggestion. Some sites intercept editing events; an accepted edit may not be
compatible with every site. Rich editors are intentionally declined rather
than changing their DOM or application state unsafely.
