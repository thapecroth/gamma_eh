# Using Gamma EH and getting help

Try the [web editor](https://thapecroth.github.io/gamma_eh/) without installing
anything. Type or paste a draft, review suggestions, and accept or dismiss each
change. Experimental Local AI starts enabled in the web editor and extension;
switch it off in the editor or popup to use only rules. Saved extension opt-outs
are preserved. If AI cannot run, local spelling and rule suggestions remain available.
Keep a copy of important writing before closing or reloading the tab. Web drafts
are temporary and are not saved by Gamma EH.

For Chrome, use the verified store link when one is published in
[store distribution](chrome-web-store.md). Until then, the
[release ZIP](releases.md) requires Developer mode: extract it into a permanent
folder, open `chrome://extensions`, choose **Load unpacked**, then reload your
website tab. Checking starts automatically in supported fields. Use the popup
to pause a site or all suggestions and to enable or disable Local AI. Chrome's
extension settings control site access. Unpacked installations do not update
automatically; see the [update instructions](releases.md).

## Supported writing

The web editor accepts drafts up to 20,000 UTF-16 code units. In the extension,
use ordinary textareas and plain-text editable fields up to 6,000 UTF-16 code
units. Rich document editors, embedded frames, shadow-root editors, password,
payment, explicitly private and other sensitive fields are excluded. Some
websites intercept editing events and may be incompatible. The checker offers
selected spelling and grammar suggestions; it does not provide general
rewriting, style advice or guaranteed correctness. Review edits before accepting
them. See [the model card](../models/MODEL_CARD.md) for measured quality and
experimental AI limitations.

## Troubleshooting

- **No suggestions:** reload the website after installing or updating. Check
  that writing suggestions are enabled, the site is not paused, Chrome grants
  site access, and the field is a supported plain-text editor. Try the fictional
  sentence `She have a freind.` in the web editor to check the rules path.
- **Local AI is unavailable or slow:** leave it off and continue with rules.
  WebGPU availability depends on your browser and device; the runtime can fall
  back to local CPU inference. See [Windows WebGPU guidance](windows-webgpu.md)
  for that platform. Do not download runtime code from third-party CDNs.
- **A suggestion is wrong:** dismiss it. Pause the field, site or global checker
  if suggestions interfere with writing. In the web editor, **Undo last change**
  also recovers Clear, Reset, and example selection until you type again. In
  supported Chrome website fields, use the editor's native Undo/Redo for an
  accepted correction; sites with custom editing handlers still need compatibility
  verification. Reloading the web editor loses its unsaved draft.
- **The extension changed after an update:** refresh existing website tabs.
  Store-managed updates and unpacked updates have different lifecycles. Keep an
  older ZIP when manually updating an unpacked installation.
- **Two sets of suggestions appear:** disable or remove the unpacked extension
  when switching to a store-managed installation, and check for other writing
  extensions running on the same page.

## Report a problem safely

Open a [GitHub issue](https://github.com/thapecroth/gamma_eh/issues) with the
version, browser/OS, web editor or extension, rules or Local AI, field type,
steps to reproduce and expected result. Replace any real draft with a short
fictional example. Include a sanitized screenshot only when useful.

Do not post private writing, personal information, credentials, browser profiles
or screenshots of real messages. Public issue reports are visible to others.
The [privacy policy](privacy.md) explains local processing and your controls.
For a possible privacy or draft-safety problem, pause checking while you collect
safe reproduction steps. The project has no published response-time commitment.
