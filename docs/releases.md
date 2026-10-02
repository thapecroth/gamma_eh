# Releases and installation

The Chrome ZIP also serves as the Chrome Web Store upload package. See
[store distribution](chrome-web-store.md) for unlisted submission material
and the optional automated store job. Preparation does not establish a live
listing; record the actual link after Google's approval.

The [GitHub Releases page](https://github.com/thapecroth/gamma_eh/releases) is
the distribution point for built packages. Release automation is defined in
[release.yml](../.github/workflows/release.yml). A workflow or local ZIP is not
itself a published release: publication requires a successful tagged run.

## Downloads

| Asset | Use |
| --- | --- |
| `gamma-eh-chrome-vX.Y.Z.zip` | Extract, then load the root folder in Chrome Developer mode. |
| `gamma-eh-web-vX.Y.Z.zip` | Extract and serve the root folder on localhost or HTTPS. |
| `gamma-eh-firefox-vX.Y.Z.zip` | Load temporarily in Firefox using `about:debugging`; permanent installs need signing. |
| `SHA256SUMS.txt` | Verify downloaded ZIPs. |

All three ZIPs include the public baseline model, bundled JAX JS runtime metadata, licenses,
model card, and `INSTALL.md`. No build tools or inference server are needed for
the extension. GitHub's automatic source archives are not installable packages.

Use the generic Releases page, not `releases/latest/download`: GitHub excludes
prereleases from the latest stable release shortcut.

### Chrome installation and updates

1. Extract the Chrome ZIP into a permanent directory.
2. Open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**,
   and select the folder containing `manifest.json`.
3. Open or reload a regular site. Checking starts automatically on supported fields.
4. Local AI is on by default. WebGPU falls back to local CPU inference, then
   spelling and rules if the model cannot run. A saved AI opt-out is respected.

Keep the extracted directory. Chrome loads unpacked assets from it. For updates,
replace its contents with the newer extracted package, click **Reload** on the
extension card, then reload existing website tabs. Keeping the same installation
preserves its settings and permissions; removing it resets them. Keep the old
ZIP for a rollback using the same replace/reload steps. Unpacked installs do
not update automatically.

This is **not** a signed CRX or Chrome Web Store installation. Ordinary one-click
installation requires a separate Web Store submission and review. See Chrome's
[unpacked installation guide](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world)
and [distribution restrictions](https://developer.chrome.com/docs/extensions/how-to/distribute/install-extensions).

### Check a download

Download the checksum file and the desired ZIP into the same directory. On Linux:

```sh
sha256sum gamma-eh-chrome-v0.1.0.zip
```

Compare the result with the corresponding entry in `SHA256SUMS.txt`.
On macOS use `shasum -a 256`; on Windows PowerShell use `Get-FileHash -Algorithm SHA256`.
With all three ZIPs downloaded, `sha256sum --check SHA256SUMS.txt` checks them together.
Checksums detect a mismatch; they are not an independent signing or provenance system.

## Build packages locally

Requires Node 24.11+ (24.x), npm, `zip`, and `unzip`.

```sh
npm ci
npm run test:release
npm run release:package
```

The packager always runs a fresh default build, then writes
`artifacts/releases/vX.Y.Z/`. It refuses to replace an existing version directory.
Keep or move an earlier local package directory before a repeated packaging run;
never overwrite published release assets.

Local packaging does not run browser checks or publish anything. To verify
locally, first run the normal checks and browser preflight, then extract the
packages and run the documented browser harness against their contents. The
hosted release workflow performs this exact-package verification automatically.

## Publish an experimental version

1. Open a PR, review the docs/model card, and merge it to `main`. The main-push
   workflow automatically chooses a new version and creates a version-only
   tagged snapshot as described below. Main's version need not be bumped for
   every automatic release.
2. For a manually chosen version instead, update the root package, lockfile root
   metadata and extension manifest together. Use a nonzero `X.Y.Z`, with each
   component at most 65535; GitHub prerelease status is metadata, not a `-beta`
   suffix in Chrome's version. Merge the reviewed metadata, then tag the commit:

   ```sh
   git tag -a v0.1.0 -m 'Gamma EH v0.1.0 experimental baseline'
   git push origin v0.1.0
   ```

3. Wait for the **Release** workflow. It checks tag/version agreement and that
   the tag is on main or is its reviewed version-only child snapshot. App checks, packaging tests,
   dataset tests, and capability preflight run sequentially. It builds once,
   unpacks the release ZIPs, and exercises actual WASM/WebGPU web and extension
   inference before the publication job can run.
4. Confirm the new prerelease has all three versioned ZIPs and `SHA256SUMS.txt`.
   Download the Chrome asset and perform a clean unpacked install. The native
   installation and site-access controls still need this manual check; the
   automated fixture loads the unchanged shipping manifest and verifies automatic
   activation and per-site pause/resume.

The build job has read-only repository permissions and no provider credentials.
The prepare job gets `contents: write` to push its version tag; publishing and
explicit stable-promotion jobs get it for release metadata. `gh release create --verify-tag`
requires the existing tag; it does not create one implicitly. Existing releases
are not clobbered. Manual workflow dispatch supports retrying an existing tag
after a transient failure before publication. If already published, use a new
version rather than replacing downloads. Dispatch defaults to `experimental`.
An explicit `channel: stable` promotes an existing prerelease only after the
[GA readiness gate](ga-readiness.md) passes for its exact archives. Main pushes
never automatically create stable releases.

## Safety boundaries and evidence

- Alternate model directories and experimental build profiles are refused.
- Only the reviewed Apache-2.0 model with original CC0 template provenance is
  currently publishable. `publication_allowed: false` is rejected in either
  model or corpus metadata. New data licenses require a separate rights review
  and an intentional change to the release gate.
- Every exported model hash is checked against its manifest; both compiled apps
  must contain exactly those model files. A test-only permission manifest is
  refused. Hidden files, symlinks, and unexpected model files fail closed.
- Runtime notices cover pinned JAX JS and Protocol Buffers packages; updating
  the runtime requires updating notices, version pins, and built metadata together.
- No private teacher corpus, isolated student checkpoint, API key, or provider
  configuration belongs in a release. The workflow uploads only public packages
  and browser-test evidence, not arbitrary contents of `artifacts/`.
- The public model is experimental. Hosted WebGPU uses SwiftShader software;
  successful checks do not establish physical-GPU speed or grammar accuracy.

Browser and release evidence is retained as a workflow artifact for 90 days. The public
ZIPs remain attached to the release. See [browser verification](browser-testing.md)
for what the automated checks do and do not establish.

## Automatic releases on main

Every push to `main` queues the Release workflow. It chooses the next patch
version above the source version and existing version tags. The workflow creates
an immutable tagged child commit containing only synchronized version updates in
`package.json`, `package-lock.json`, and the extension manifest; it does not write
to main. Retries reuse the snapshot for the same source commit.

The release pipeline validates that the snapshot's parent belongs to main and
that its only changes are version metadata. It runs the existing app, training,
Chrome and web browser gates, builds all three ZIPs, checks their contents and
checksums, then publishes an experimental GitHub prerelease. Failed gates leave
the tag available for retry but publish no release. Manually dispatched releases
and manually pushed version tags remain supported. Release runs queue sequentially
so version allocation and publication cannot overlap. No extra PAT is needed:
the workflow's repository token can push tags, and its tag pushes do not recursively
trigger another release workflow.

See [Firefox installation](firefox.md). Firefox packaging and manifest checks are
covered. Firefox 156.0.1 also passed a live rules/WASM worker smoke check with a
localhost fixture grant; full Firefox UI and permission-dialog coverage remain a follow-up.
