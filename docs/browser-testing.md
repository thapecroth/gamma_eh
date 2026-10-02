# Browser verification and environment prerequisites

Browser checks must execute the actual bundled model in the web worker and
Chrome MV3 offscreen worker. A build, unit-test pass, or available WebGPU adapter
alone does not prove this path.

## Preflight

```sh
GAMMA_TEST_WEBGPU=1 npm run check:environment
```

The preflight independently checks a localhost HTTP roundtrip and actual
Chromium startup. With `GAMMA_TEST_WEBGPU=1` it also requests an adapter from a
localhost page. It collects all failures instead of stopping at the first one,
and writes sanitized, scoped evidence to `artifacts/environment.json`.
No sandbox, firewall, proxy, or host configuration is changed by the preflight.
Local server cleanup explicitly closes its own speculative browser connections
before waiting for shutdown. Navigation and adapter acquisition have bounded
timeouts; an unavailable or stalled adapter fails readiness rather than passing
via CPU fallback.

To check the full live-pipeline prerequisites, securely set `TEACHER_API_KEY` and:

```sh
TEACHER_BASE_URL=http://127.0.0.1:8317/v1 \
  TEACHER_MODEL=glm-5.3-flash GAMMA_REQUIRE_TEACHER=1 \
  GAMMA_TEST_WEBGPU=1 npm run check:environment
```

This contacts the authenticated model catalog only, not chat completions. Keys,
HTTP/provider bodies, headers, and raw exception messages are not persisted.
Browser-only readiness is explicitly different from live-pipeline readiness.

## Hosted browser lane

The `browser-e2e` job in `.github/workflows/check.yml` runs after app and data
checks, on a fresh GitHub-hosted runner. It installs the lockfile-matched
Chromium and operating-system dependencies, checks capabilities, builds the
shipping app, and runs `npm run test:browser` with WebGPU required. Runs preserve
the environment report, correction results, and fictional-fixture screenshots
for seven days, including on failure. PR checks run once per change; main pushes
are also checked.

Chromium uses a persistent profile and the supported `chromium` channel, following
[Playwright's extension guidance](https://playwright.dev/docs/chrome-extensions).
Every grammar smoke case must now match its expected correction. A miss is a
failing assertion, not a successful run with a lower accuracy counter.

The previously missed simple singular-subject/habitual-verb case is covered by
a limited rule fallback. It only recognizes a small set of simple noun subjects,
unambiguous present-tense verbs, and explicit habitual time cues. Questions,
compound/plural subjects, subjunctives, ambiguous past forms such as `read`, and
protected code have negative tests. This does not improve the neural benchmark
or establish general grammar quality.

## Scope and remaining gates

The additional `npm run test:extension` suite exercises automatic site activation
and settings, stale edits, emoji offsets, dismissing and protected fields. It
runs in hosted browser and release jobs and saves traces and screenshots under
`artifacts/agents/extension-*/`. For Luna-driven scenarios and the Codex repair
loop, see [agentic development](agentic-development.md).

Hosted CI only uses the public Apache-2.0 baseline and original fictional
fixtures. It does not upload the private, output-terms-unverified GLM corpus or
student weights, store teacher credentials, or connect back to the development
machine. A hosted app pass cannot prove the unfinished live teacher pipeline.

The extension fixtures load an unchanged copy of the shipping manifest with
automatic HTTP/HTTPS content scripts. Checking starts without popup activation
or dynamic per-site registration, with Local AI enabled by default. The fixtures
verify model corrections before any AI setting is saved and saved opt-outs after
navigation. A separate shared-engine browser check disables WebAssembly and
verifies spelling/rules fallback. Both localhost and `gamma-http.test` point to the same local server; the
latter is an ordinary insecure HTTP origin. Session identifiers use
`crypto.getRandomValues` so startup also works where `crypto.randomUUID` is
unavailable. The HTTP fixture must be insecure and have no `randomUUID`.

The test exercises the real popup's per-site pause and resume buttons, verifies
that pauses affect every same-site tab and persist across navigation, and checks
that another site's suggestions remain enabled. Resuming restarts checking in
the already-focused field. Global pause, protected fields, plain contenteditable
corrections, rich-DOM preservation, and zero external requests remain checked.

Chrome's native installation and site-access controls require separate interactive
verification. Existing website tabs must be reloaded after an extension update.
Rich editors and nested/shadow frames are still unsupported, not claimed as tested
correction targets. Software WebGPU is functional proof, not physical-GPU
performance evidence. See [Live pilot](live-pilot.md) for the private-student run.
