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
| `SHA256SUMS.txt` | Verify both downloaded ZIPs. |

Both ZIPs include the public baseline model, bundled JAX JS runtime metadata, licenses,
model card, and `INSTALL.md`. No build tools or inference server are needed for
the extension. GitHub's automatic source archives are not installable packages.

Use the generic Releases page, not `releases/latest/download`: GitHub excludes
prereleases from the latest stable release shortcut.

### Chrome installation and updates

1. Extract the Chrome ZIP into a permanent directory.
2. Open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**,
   and select the folder containing `manifest.json`.
3. Open a regular site, click Gamma EH, and enable checking for that site.
4. Local AI is off by default; enable it in the popup to try the model.

Keep the extracted directory. Chrome loads unpacked assets from it. For updates,
remove the old unpacked extension and load the newer extracted directory, then
grant site permissions again. This resets extension preferences. Keeping the
old ZIP allows a rollback using the same remove/load steps. Unpacked installs
do not update automatically.

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
With both ZIPs downloaded, `sha256sum --check SHA256SUMS.txt` checks both together.
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

## Publish a version

1. Update the root `package.json` version and `apps/extension/manifest.json`
   together; update package-lock metadata with npm. Use a nonzero `X.Y.Z` with
   each component between 0 and 65535. Prerelease status is GitHub metadata,
   not a `-beta` suffix in Chrome's version.
2. Open a PR, review the docs/model card, and merge it to `main`.
3. From the merged commit, create and push the matching version tag:

   ```sh
   git tag -a v0.1.0 -m 'Gamma EH v0.1.0 experimental baseline'
   git push origin v0.1.0
   ```

4. Wait for the **Release** workflow. It checks tag/version agreement and that
   the tagged commit is reachable from `main`. App checks, packaging tests,
   dataset tests, and capability preflight run sequentially. It builds once,
   unpacks the release ZIPs, and exercises actual WASM/WebGPU web and extension
   inference before the publication job can run.
5. Confirm the new prerelease has both versioned ZIPs and `SHA256SUMS.txt`.
   Download the Chrome asset and perform a clean unpacked install. The native
   optional-site-permissions dialog still needs this manual check: the automated
   extension fixture uses a test-only localhost grant.

The build job has read-only repository permissions and no provider credentials.
Only the publishing job gets `contents: write`. `gh release create --verify-tag`
requires the existing tag; it does not create one implicitly. Existing releases
are not clobbered. Manual workflow dispatch supports retrying an existing tag
after a transient failure before publication. If already published, use a new
version rather than replacing downloads.

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

Browser evidence is retained as a workflow artifact for seven days. The public
ZIPs remain attached to the release. See [browser verification](browser-testing.md)
for what the automated checks do and do not establish.
