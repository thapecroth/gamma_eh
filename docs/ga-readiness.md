# General availability readiness

**Status: pending; quality is blocked.** The project maintainer owns launch
approval and all unassigned external checks. This document establishes a release
process, not a claim that the product is ready. The authoritative versioned
record is [ga-readiness.json](ga-readiness.json), described by
[its schema](ga-readiness.schema.json). No passed receipts are supplied.

## Execution checklist

The following preparation is implemented and locally verified; it does not
replace final launch signoff. See [the validation report](ga-validation.md) for
commands, measurements, tested package hashes and coverage limits.

- [x] Define a proposed Chrome/web launch scope and measurable quality gates.
- [x] Evaluate the complete independent natural development and test populations
  with actual browser predictions and pinned ERRANT scoring.
- [x] Preserve native extension Undo/Redo and reject stale, private, detached,
  cancelled or focus-redirected edits.
- [x] Make web Clear, Reset, example selection and corrections recoverable until
  newer typing invalidates the recovery snapshot.
- [x] Bundle privacy/help pages, explain temporary drafts and hosting metadata,
  and supply safe onboarding, troubleshooting and issue-report guidance.
- [x] Add a production content policy and correction-specific accessible labels;
  verify keyboard navigation, responsive layouts and automated accessibility.
- [x] Verify local inference, CPU/software-WebGPU fallback, bundled model hashes,
  licenses and unchanged production extension permissions in extracted ZIPs.
- [x] Measure local inference startup and repeat/edit workloads without claiming
  physical-device performance.
- [x] Keep ordinary releases experimental; require complete, reviewed,
  source/model/archive-bound evidence for immutable stable promotion.
- [x] Run edit-safety and web checks in CI; retain browser and release evidence
  for 90 days; keep private-mirror Pages publication opt-in.
- [x] Prepare incident, higher-version hotfix, pilot and web-recovery runbooks.
- [ ] Meet approved natural-writing quality targets and independently review
  meaning preservation. Current precision and development clean-text results fail.
- [ ] Complete the declared physical Windows/macOS/Linux device matrix, including
  performance budgets and manual IME/caret/long-draft compatibility.
- [ ] Obtain a store publisher/item, approval and actual store install/update/
  uninstall evidence; validate the final production web assets and recovery.
- [ ] Complete screen-reader, zoom and remaining manual accessibility review.
- [ ] Complete a consenting representative user pilot against predeclared targets.
- [ ] Assign named launch/incident reviewers, configure the `stable-release`
  environment protections and sign exact-release receipts before promotion.

The proposed first GA scope is the Chrome extension and HTTPS web editor on
Windows, macOS, and Linux. Rules and spelling are the proposed GA core; Local AI
stays explicitly experimental and user-controllable. It starts enabled in the
web editor; Chrome users enable it in the extension popup. Firefox ZIPs remain experimental.
The web editor supports drafts up to 20,000 UTF-16 code units. The extension
supports ordinary textareas and plain-text contenteditable editors up to
6,000 UTF-16 code units. Rich document editors, iframes, shadow roots, and
sensitive/private fields are outside this launch scope. The maintainer must
approve the scope and quality policy before collecting final launch evidence.

## Current gate ownership

| Gate | Owner | Status and required evidence |
| --- | --- | --- |
| Natural-writing quality | Project maintainer | **Blocked:** current independent rules precision is 71.90% on test; development changes 38/216 accepted clean sources (17.59%). These miss proposed 95% precision and 2% clean-change targets. |
| Draft safety | Project maintainer | Pending release-candidate stale-text, UTF-16, caret, undo/redo, IME, accept-all, sensitive-field, fast-typing and long-text verification. |
| Physical devices/performance | Project maintainer | Pending Windows/macOS/Linux CPU and physical-WebGPU measurements against approved startup/latency/memory budgets. SwiftShader does not qualify. |
| Consumer distribution | Project maintainer | Pending actual store approval, clean store install, upgrade/settings/site-access/uninstall, and production web asset/recovery checks. The public playground exists; its URL alone does not prove these checks. |
| Privacy | Project maintainer | Pending exact-package network/storage/logging, permission, bundled-code and disclosure audit. The existing [privacy policy](privacy.md) is preparation. |
| Accessibility | Project maintainer | Pending keyboard, screen reader, contrast, zoom, responsive layout and status-announcement verification. Automated checks alone do not establish screen-reader usability. |
| User pilot | Project maintainer | Pending a completed representative pilot with predeclared acceptance criteria and review. |
| Support/recovery | Project maintainer | Runbook below prepared; named incident ownership and hotfix/recovery rehearsal remain pending. |

The model card and [controlled tuning](model-tuning.md) describe current results.
The minimum 25 predicted edits is an evidence floor, not statistical
certification. Synthetic scores are never natural-writing GA evidence. Both
rules and combined modes need measured development/test results and a human
meaning-preservation review. Experimental AI may remain below GA precision
targets only with honest reviewed limitations; marking AI `ga` applies the
quality targets to combined mode too. No meaning-changing failures are accepted
in either mode's reviewed acceptance evidence.

## Collect reviewed evidence

1. Choose an existing experimental version with immutable, tested ZIPs. Run
   `npm run release:readiness -- identity refs/tags/vX.Y.Z` to obtain the source
   commit/digest and model-manifest digest. Copy the archive hashes from its
   `SHA256SUMS.txt`. Retest those exact archives for all manual checks.
2. Record the proposed scope, approved quality/performance/pilot targets, named
   owner and final reviewer. Freeze natural development/test populations before
   tuning. Keep evaluation separate from training and weak teacher labels, and
   retain population hashes, licenses, methodology and aggregate metrics.
3. Create compact sanitized JSON receipts under `docs/ga-evidence/`. Each names
   `schema_version: 1`, `gate`, `tag`, `source_sha256`,
   `model_manifest_sha256`, `assets` (chrome/firefox/web SHA-256 hashes),
   `performed_at` (UTC `YYYY-MM-DDTHH:MM:SSZ`), `reviewed_by`, and `checks`.
   Every required check contains `id`, `passed: true`, and specific `notes`
   identifying the method and outcome. Required IDs are exported as
   `readinessChecks` in [the validator](../scripts/ga-readiness.mjs); device
   checks are `<os>-cpu` and `<os>-webgpu` for every claimed platform.
4. Quality receipts additionally name `population`, `methodology`,
   `experimental_ai_limitations`, and `metrics.development`/`metrics.test`, each
   with `rules`/`combined`. Metrics contain sentence and clean-sentence counts,
   predicted-edit count, edit precision and clean-changed rate as fractions,
   and a reviewed `meaning_changing_errors` count. Platform receipts contain
   `devices` with OS, CPU, GPU, browser, `webgpu_adapter: "physical"`,
   `cold_start_ms`, `warm_p50_ms`, `warm_p95_ms`, `peak_memory_mb`, and reviewed
   `performance_budgets`/`performance_budgets_passed`. Distribution receipts
   include verified `store_url` and `web_url`. Pilot receipts include participant
   count, `acceptance_criteria` and `acceptance_criteria_met`.
5. Hash each receipt and update the matching gate's owner, path and SHA-256.
   Set status `ready` only after every gate passes, then record `approved_by`
   and `approved_at`. Evidence expires within 90 days. Review the evidence PR
   before dispatching stable promotion. Never commit drafts, personal data,
   credentials, private corpora, browser profiles or confidential raw reports.

Source attribution includes shipping apps, engine, model, scripts, dependencies,
licenses and user-facing disclosure/scope documents. Version metadata alone is
normalized, so an automatic tagged snapshot can share its parent's reviewed
source digest. The GA record, schema and receipts are excluded from that digest
so recording evidence does not invalidate it. Evidence may name an ancestor
source commit only when its complete normalized shipping digest matches the tag.
Changing runtime, model, dependency or disclosure inputs requires fresh evidence.

## Stable promotion and release verification

Normal main pushes, version tags, and default manual dispatches create
experimental prereleases. Packaging with `GAMMA_RELEASE_CHANNEL=stable` is
refused. Stable publication requires an explicit **Release** workflow dispatch
with an existing tag and `channel: stable`. It downloads that prerelease's
unchanged archives, verifies checksums and every readiness receipt, retains the
approval, then changes only release status/title/notes. It never rebuilds or
replaces reviewed archives. Pending, missing, stale, failed, software-GPU,
wrong-version, changed-model and changed-archive evidence fails closed.
Candidate snapshot verification executes the trusted verifier from main, so an
unverified tag cannot replace its own provenance check.

Configure the GitHub `stable-release` environment with maintainer required
reviewers and main-only deployment rules before enabling stable promotion.
Repository administrators own those external settings. The committed gate still
fails closed without complete evidence. Store submission/Google approval and
GitHub prerelease promotion are separate actions; promotion does not publish an
item in Chrome's store. CI browser and release evidence retain for 90 days;
archive important sanitized signoffs before that retention expires.

Local validation is sequential: `npm run check`, Python training tests,
browser capability preflight, then the web/extension CPU and WebGPU harnesses
against extracted release ZIPs. The [release runbook](releases.md) describes
exact-package validation; [browser testing](browser-testing.md) describes the
coverage limits. Hosted success is separate from physical-device, store,
screen-reader, and user-pilot proof.

## Pilot and support runbook

Recruit consenting people who represent the declared everyday writing use
cases and supported devices. Use fictional drafts for recorded demonstrations.
Before the pilot, document success targets for installation, suggestion
usefulness, incorrect suggestions, repeat use and draft safety. Record
aggregate findings and review actionable failures; collect no writing by
default. Participants can report a voluntarily sanitized example. The
maintainer approves acceptance criteria and signs the completed pilot receipt.

Users can report problems through [GitHub issues](https://github.com/thapecroth/gamma_eh/issues).
Ask for extension/web version, browser/OS, CPU/WebGPU backend, supported field
type, reproduce steps and a fictional example. Tell users to exclude private
writing, personal data, screenshots of real conversations and browser profiles.
Store installation, [unpacked updates](releases.md), controls and model limits
are documented separately. An issue URL is an available contact channel; it
does not promise response times or an assigned on-call schedule.

For draft corruption or a privacy incident, the maintainer owns triage: advise
pausing checking/Local AI, reproduce on fictional text, preserve sanitized
version and backend evidence, fix and revalidate the affected path. Disable
`CWS_ENABLED` to stop automated store submissions. Never overwrite release ZIPs.
Store rollback requires a higher-version release containing the reverted code;
the web editor can redeploy the last verified asset set following the
[Pages recovery guide](github-pages.md). Rehearse a package hotfix and web
recovery before signing operations readiness. Keep critical defects open until
verified fixed, and preserve settings and draft safety through updates.
