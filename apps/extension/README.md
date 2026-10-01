# Chrome extension

## Download a built package

When available, download `gamma-eh-chrome-vX.Y.Z.zip` from the
[Releases page](https://github.com/thapecroth/gamma_eh/releases), then extract it
into a permanent folder. Use the folder containing `manifest.json` in the steps
below. No Node.js or build step is needed. Do not download the automatic source
ZIP for installation. Packages contain the runtime, model, and licenses.

To build from source instead, run `npm ci` and `npm run build` at the repository
root, then use `dist/extension/` as the installation folder.

## Install

1. Open `chrome://extensions` in a recent Chrome version with offscreen-document
   support (Chrome 116 or newer).
2. Enable **Developer mode**, choose **Load unpacked**, and select
   the extracted release folder (or `dist/extension/` for a source build).
3. Open a website, select the Gamma EH toolbar button, and choose **Enable on
   this site**. Chrome requests access to that site only.
4. Focus an ordinary textarea or plain-text contenteditable field. Suggestions
   appear in a small panel. Choose **Accept** or **Dismiss**; close the panel to
   pause that field.

Keep the extracted folder on disk. To update, remove the old unpacked extension,
extract the new release, and load its folder. Re-enable the desired sites; this
resets preferences. Unpacked installs do not auto-update. This is developer-mode
installation, not a Chrome Web Store or one-click CRX installer. See the
[release runbook](../../docs/releases.md) for checksums and publication details.

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


## Firefox

Build output is `dist/firefox/`. See [Firefox installation](../../docs/firefox.md).
