# Frontend toolchain

Gamma EH uses project-local Vite+ 1.0.0 with Node.js 24.11 or newer in the 24.x
line. Contributors and CI use the same npm lockfile; no global `vp` installation,
runtime manager, or machine-level configuration changes are required.

```sh
npm ci
npm run dev       # Vite+ web editor development server
npm run lint      # Vite+ Oxlint, scoped to frontend code and scripts
npm test          # Vite+ bundled Vitest
npm run check    # Sequential lint, TypeScript, tests, and both production builds
```

## Dependency identity

`vite-plus` and the `vite` alias to `@voidzero-dev/vite-plus-core` are both pinned
to 1.0.0. npm overrides route all Vite consumers, including the React plugin,
through that alias and pin Vitest to Vite+'s bundled 5.0.1. This avoids mixed
Vite instances or test-runner internals. Upgrade these entries together.
Configuration imports `defineConfig` from `vite-plus`; test assertions import
from `vite-plus/test`. `scripts/build.mjs` intentionally retains its programmatic
`build` import from `vite`, which resolves to the aliased core. Vite+ does not
guarantee all upstream programmatic exports on its root entry.

This is a scoped manual migration: the existing tests use basic Node assertions
without mocks, fake timers, browser-mode matchers, or compatibility overrides.
TypeScript's existing `tsc --noEmit` gate is retained. Repository-wide formatting,
Git hooks, agent files, and package-manager migration are not part of this change.

## Production packaging

Always use `npm run build`. Its orchestrator calls the Vite+ core to build the
web editor, then uses the existing esbuild packager for Chrome's ESM background,
offscreen, and inference workers and IIFE content/popup scripts. It copies model
weights, manifests, tokenizer files, and ONNX runtime assets into both outputs.
Nothing is fetched from a CDN at inference time.

Bare `vp build` only performs a Vite application build. It does not perform the
extension packaging or model/runtime asset-copy steps and is not the release
command for this repository. Isolated `GAMMA_BUILD_PROFILE` and `GAMMA_MODEL_DIR`
builds still use the same orchestrator and cannot overwrite the shipping model.

## Verification

`npm run check` validates frontend code and builds; Python dataset tests and
actual Chromium/WASM/WebGPU E2E remain separate gates. See
[Browser verification](browser-testing.md) for the hosted runner and local
capability checks. Toolchain changes do not grant OS networking/browser permissions
or establish grammar quality.

References: [Project-local CLI](https://viteplus.dev/guide/local-cli),
[Migration rules](https://viteplus.dev/guide/migrate-rules), and
[Vitest 5 compatibility](https://viteplus.dev/guide/vitest-v5).
