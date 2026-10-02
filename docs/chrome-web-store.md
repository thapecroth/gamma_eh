# Chrome Web Store distribution

Gamma EH is prepared for an **unlisted** Chrome Web Store item: anyone with
the link can install it, without appearing in store search. Unlisted items
receive the same review and automatic updates as public items. This document
is preparation, not evidence of publication. Record the actual store URL here
after the first approval.

## First submission

1. [Register a publisher](https://developer.chrome.com/docs/webstore/register)
   with the Google account that should own the extension. Google requires a
   one-time registration payment and two-step verification. Complete the
   dashboard's contact and identity requirements.
2. Merge store preparation, then create a versioned release following the
   [release runbook](releases.md). Upload the Chrome ZIP from the verified
   Release workflow: `gamma-eh-chrome-vX.Y.Z.zip`. It includes the root manifest,
   local model/runtime assets, icons, and licenses. Never upload a source ZIP,
   private model experiment, or test-only site-grant manifest.
3. In the [Developer Dashboard](https://chrome.google.com/webstore/devconsole),
   create a new item, upload the ZIP, and fill Store listing, Privacy,
   Distribution, and test instructions using the material below. Choose
   **Unlisted**. This extension is currently experimental.
4. After merging, use the publicly accessible [privacy policy](privacy.md) URL:
   `https://github.com/thapecroth/gamma_eh/blob/main/docs/privacy.md`.
   Confirm it opens without signing in before entering it in the dashboard.
5. Submit for review. Confirm Google's approval and installation from the real
   store URL before claiming store distribution or automatic updates. Add the
   link here and to the extension installation README.

## Listing material

**Name:** Gamma EH — Local writing assistant

**Short description:** An open-source English writing assistant. Your text stays on your device.

**Category:** Productivity (choose the closest writing/productivity category
available in the dashboard). **Language:** English.

**Detailed description:**

> A little clarity, right where you write.
>
> Gamma EH offers English spelling and selected grammar suggestions in ordinary
> website text boxes and plain-text editors. Checking starts automatically on
> ordinary websites. Focus a supported field to see possible issues underlined. Hover or click
> an underlined word to review a correction. Accept it, dismiss it, or pause
> checking for that field. Alt+F8 opens suggestions; Escape closes them.
>
> Your writing stays on your device. Checking uses bundled local rules, a
> spelling dictionary, and an optional tiny AI model. No account, tracking, or
> cloud writing service is required. Local AI is experimental and uses a
> synthetic training baseline; its suggestions are not evidence of general
> grammar accuracy. Review corrections before applying them.
>
> Use the toolbar popup to pause the current site, pause suggestions globally,
> or turn Local AI on or off. Browser extension settings control host access.
> Store installations receive updates through Chrome after
> approved releases.
>
> Supported: ordinary textareas and plain-text contenteditable fields, up to
> 6,000 UTF-16 code units. Password and payment inputs, explicitly private
> fields, unsupported rich editors, and embedded frames are excluded. Some
> websites may intercept editing events and are not compatible.
>
> Open source: https://github.com/thapecroth/gamma_eh

**Homepage:** `https://github.com/thapecroth/gamma_eh`

**Support:** `https://github.com/thapecroth/gamma_eh/issues`

Required images:

- Icon: `apps/extension/icons/icon-128.png` (128 × 128, transparent padding).
- Small tile: `docs/assets/webstore/promo-440x280.png` (440 × 280).
- Screenshots: `docs/assets/webstore/suggestions-1280x800.png` and
  `correction-1280x800.png` (1280 × 800). These show the actual built extension
  on an identified demonstration page with fictional writing.

Run `npm run webstore:artwork` to regenerate icons/tile. PNGs are committed so
normal builds do not require a browser for icons. After `npm run build`, run
`npm run webstore:screenshots` to capture real UI and assert the demonstrated
corrections. Like the extension test driver, the screenshot fixture uses the
unchanged shipping manifest and a fictional localhost page. It
blocks external requests and removes its temporary browser profile afterward.
Screenshots do not prove the native permission dialog or store installation.
Both asset commands use the shared harness lock; wait for an active run to
finish before starting them.

## Privacy fields

**Single purpose:** Provide local English writing suggestions in supported
editable fields on ordinary websites, with global and per-site pause controls.

| Permission | Justification |
| --- | --- |
| `activeTab` | Read the current tab's address when the user opens the popup, to display its site and pause state. |
| `scripting` | Remove obsolete per-site script registrations when upgrading from older versions. Checking now uses bundled static content scripts. |
| `storage` | Save the writing-suggestion and Local AI preferences and paused-site list locally. Draft text and suggestions are not persisted. |
| `offscreen` | Run the bundled local inference worker in an extension-owned document without sending text to a service. |
| HTTP/HTTPS hosts | Automatically check supported fields on ordinary websites. The popup can pause globally or by site; browser settings control host access. |

**Remote code:** Select **No**. Runtime code, model weights, and dictionary data
are bundled. Model reads use extension-local URLs; CSP limits connections to
the extension's own origin.

**User data:** Disclose local handling, even though nothing is sent to the
developer. Relevant categories are **Website content** (editable-field text),
**Personal communications** (if a field contains a draft message), and **Web
history** (the current site address for access controls, without creating a
history log). Explain that processing stays on the device for the single
purpose. Do not claim the extension handles no user data. Use the
[privacy policy](privacy.md) and certify Limited Use statements while they
match shipped behavior. Answer any additional dashboard questions against
the implementation; offline processing is not exempt from disclosure.

## Reviewer instructions

No login or subscription is required. Install in Chrome 116 or later. On an
ordinary HTTPS page containing a textarea or plain-text editor, reload the tab
and focus the field. Checking and **Local AI** start automatically. Also verify
the Local AI off/on control and that a saved opt-out survives reloading.

Enter `She have a freind.` and focus the field. Open the suggestion badge or
hover an underlined word. Accept both suggestions; the result is
`She has a friend.`. Enter `😀 A freind.` and accept the spelling change; the
emoji must remain intact. Edit a sentence before accepting an old suggestion;
the extension should recheck instead of applying stale offsets.

Test **Writing suggestions** off/on, **Remove site permission**, and enabling
the site again. Password/payment inputs and fields with `spellcheck="false"`
or `data-private` should have no checking UI. Default-enabled Local AI runs a bundled
experimental model via WebGPU when available, with local CPU and rule fallbacks.

## Automating later releases

Create and publish the first unlisted item in the dashboard. Then configure
the official [Chrome Web Store API v2](https://developer.chrome.com/docs/webstore/using-api):

1. Enable the API in a Google Cloud project, create an OAuth client, and obtain
   a refresh token for the owning publisher account with the
   `https://www.googleapis.com/auth/chromewebstore` scope. Follow Google's
   linked setup instructions. Keep credentials out of source and chat.
2. Create a GitHub environment named `chrome-web-store`. Add variables
   `CWS_PUBLISHER_ID` (Publisher → Settings) and `CWS_EXTENSION_ID` (32-character
   item ID). Add secrets `CWS_CLIENT_ID`, `CWS_CLIENT_SECRET`, and
   `CWS_REFRESH_TOKEN`.
3. Set repository variable `CWS_ENABLED` to `true` after the first approved
   unlisted publication. Before that, the store job is skipped. Normal ZIP
   releases still run. Optionally restrict the environment to version tags.

The Release workflow's store job runs after app/data/browser checks and GitHub
release publication. It reuses the exact tested Chrome ZIP. The submitter
checks SHA-256, root manifest/version, item identity, and store state; uploads
through v2; waits for processing; and submits with review enabled. Store jobs
are serialized, warnings block submission, a different pending review is
preserved, and versions already pending or published are recognized. OAuth
credentials and Google response bodies are not printed. PR checks have no
store credentials.

`PENDING_REVIEW` means submitted, not published. Google may approve or reject
later. Confirm the dashboard and a store-installed browser's version before
claiming users updated. Chrome checks periodically and normally installs when
the extension is idle; updates are not immediate. Existing tabs may need a
refresh afterward.

After a failure, inspect the dashboard before retrying, then use **Re-run failed
jobs** on the Release run to retry only the failed store job. This avoids
replacing published GitHub assets. For package fixes, increase the package,
lockfile, and manifest versions and make a new gated release. Disable
`CWS_ENABLED` to stop future submissions. Chrome rejects lower versions; roll
back code through a higher-version release containing the revert.

## Moving unpacked installations

After approval, install from the store link and disable or remove the unpacked
copy so two checkers do not run. The store item has its own identity; choose
settings and site pauses once. Later updates come through Chrome.
An `update_url` does not turn an unpacked installation into a store installation.

Sources: [preparation](https://developer.chrome.com/docs/webstore/prepare),
[images](https://developer.chrome.com/docs/webstore/images),
[visibility](https://developer.chrome.com/docs/webstore/cws-dashboard-distribution),
[local data disclosure](https://developer.chrome.com/docs/webstore/program-policies/user-data-faq),
and [update lifecycle](https://developer.chrome.com/docs/extensions/develop/concepts/extensions-update-lifecycle).
