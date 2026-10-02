# Gamma EH privacy policy

Effective date: October 1, 2026.

Gamma EH provides English writing suggestions in its web editor and inside
supported website text fields through its extension. Checking runs on your device.
It has no account system, analytics,
advertising, or developer-operated service that receives your writing.

## Web editor

The web editor checks the draft you type or paste in that tab. Local rules and
the bundled dictionary run on your device. Local AI is experimental and starts
enabled in the web editor; you can switch it off anytime. When enabled, it loads
model files from the site's own origin
and runs inference on your device. Drafts are never included in those asset
requests or sent to an inference service.

Drafts, suggestions, and the one-step undo snapshot remain in tab and worker
memory. The web editor does not save them in browser storage, cookies, a
database, or an account. Reloading or closing the tab loses the draft. Copy
text writes the current draft to your system clipboard only when you choose
that action; your operating system and other applications control clipboard
retention and access. Gamma EH does not read your clipboard automatically.

The public site is hosted on GitHub Pages. Loading it sends ordinary requests
for HTML, JavaScript, styles, icons, and optional model files to the host. The
host can process connection information such as your IP address and browser
headers under [GitHub's privacy statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).
Those requests do not contain your draft. Gamma EH adds no analytics or draft
logging. Host-operated access logs are separate from the local checker; this
policy does not promise that the hosting provider retains no connection data.

## Extension information processed on your device

When you focus a supported textarea or plain-text
editor, the extension reads that field's text, including existing text, to find
possible corrections. This may include a draft message or personal information
you choose to write. Text and suggestions pass between extension components
and remain temporarily in memory during checking and review. Gamma EH does
not save drafts or suggestions to persistent storage, send them to a server,
or use them to train a model.

Experimental Local AI starts enabled unless you have saved an opt-out. You can
switch it off in the popup. Its bundled model runs locally, with local CPU or
rule fallback when a runtime is unavailable.

The extension reads the current tab's address to display the site in its popup
and manage access. It does not create a browsing-history record or monitor
unrelated page content. Checking starts automatically in the top frame on
ordinary HTTP/HTTPS sites with browser-granted access, unless you pause it.
Password inputs, payment/authentication input fields, fields marked private or
sensitive, and unsupported rich editors are excluded. Gamma EH cannot recognize
every kind of sensitive information in an ordinary text field.

## Extension information retained locally

Gamma EH saves preferences in local extension storage: whether writing
suggestions and experimental Local AI are enabled, plus the list of paused
sites. Chrome manages host-access grants. These choices
persist until you change them or remove the extension. Drafts and suggestions
are temporary in-memory data; closing the page ends that page's checking.
Processing components may retain active or queued text while work completes.

## Sharing and updates

Gamma EH does not sell, transfer, or disclose your writing, site choices, or
settings to its developers or third parties. It does not use information for
advertising, profiling, credit decisions, or unrelated purposes. Model weights,
dictionary data, and executable runtime code are included in the package.

Chrome downloads extension packages and updates from the Chrome Web Store.
Those browser-managed requests do not include your writing from Gamma EH.
Google and the websites you visit have their own privacy practices; this policy
describes Gamma EH's web editor and extension. Visiting project links or sending a support
report is a separate action; anything you include in a report is visible to
its recipient. Do not post private writing in public issues.

Gamma EH's use of information is limited to its writing-assistance features
and complies with the Chrome Web Store User Data Policy, including its
Limited Use requirements.

## Your controls and contact

Use the popup to pause writing suggestions globally or on the current site,
or switch Local AI off. Use Chrome's extension settings to manage
site access or uninstall Gamma EH. Uninstalling removes its local extension
settings. You can review each suggested correction before accepting it.

For privacy questions, use the project's
[GitHub issues](https://github.com/thapecroth/gamma_eh/issues). Describe the
problem without including sensitive writing or personal information.

Changes to this policy will be published here with an updated effective date.
