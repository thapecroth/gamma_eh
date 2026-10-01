# Inline extension suggestions

The Chrome extension marks detected errors with red wavy underlines in the
focused textarea or supported plain-text contenteditable. Hovering or clicking
a marked word opens one correction beside it. Accept changes only that edit;
Dismiss removes its mark without changing the text. A compact count button
opens the first suggestion and provides previous/next navigation. Alt+F8 opens
the card and focuses Accept; Escape closes it and returns focus to the editor.
The card also offers Pause this field. Checking, empty and unsupported-editor
messages live behind the small button rather than a persistent corner panel.

## Measurement and privacy

`apps/extension/src/highlights.ts` measures textarea offsets in a temporary
typography-matched mirror inside the extension's closed shadow root. Plain
contenteditable uses native Range rectangles and the same UTF-16 boundaries
as edit application, including line breaks. Neither method inserts markup
into the page's editor, replaces its contents, or changes its selection.
The mirror is removed immediately after measurement. Underlines do not receive
pointer events, so clicking text still positions the caret normally.

The extension repositions marks and the card on scrolling, field and ancestor
resizing, layout movement, viewport resizing and font loading. It watches the
active field's bounding box without repeating text measurement while stationary.
The mirror copies computed wrapping styles, including unwrapped textareas.
Offscreen lines are clipped against the
field, viewport and scrolling ancestors. Wrapped ranges can have several marks.
Typing and IME composition immediately clear old marks; snapshots are checked
again before showing or accepting a correction. Text still stays local, and
the bundled experimental model retains its limitations. An underline means
the checker found a suggestion, not that every error has been detected.

## Verification and limits

`npm run check` verifies lint, types, app tests and packaging. After a build,
`npm run test:extension` runs the existing MV3 correction/privacy scenarios plus
real pointer and keyboard interaction, independent word-position measurements,
UTF-16 offsets, IME invalidation, textarea scrolling, plain-editor BR boundaries,
stale programmatic edits, CSS wrapping, horizontal and document scrolling,
translated ancestors, clipped fields, narrow viewport card placement and the
`halo` greeting example through hover and pointer acceptance. It saves fictional
fixture screenshots and traces under `artifacts/agents/extension-*/`.

The suite uses the shipping script on a local fixture; it does not automate a
signed-in GitHub page or Chrome's native site-permission dialog. Rich editors,
iframes and shadow editors remain unsupported. Rotated/skewed editors and
nonstandard vertical writing layouts are outside the measured layout scope.
