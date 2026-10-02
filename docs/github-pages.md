# Public playground on GitHub Pages

The public demo is at **https://thapecroth.github.io/gamma_eh/**. Anyone can open
it and try Gamma EH without installing the Chrome extension, creating an account,
or supplying an API key. GitHub serves static files; checks run on the visitor's
device. Local AI starts enabled, falls back when unavailable, and remains an
experimental synthetic baseline.

## Deployment

The repository's Pages publishing source is **GitHub Actions**. The `Check`
workflow publishes only pushes to `main`, after its app, data, and browser gates
pass. Pull requests exercise root and repository-path builds on CPU and WebGPU,
but cannot deploy.

Verify that the account's user site has no conflicting custom domain before
publishing. GitHub [inherits that domain for project sites](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages), which redirects the
`github.io` URL; removing a project-level domain alone does not clear inheritance.

`pages-build` obtains the site's base path from `actions/configure-pages`, builds
an isolated `GAMMA_BUILD_PROFILE=pages` variant, and uploads only `dist/pages/web`.
`pages-deploy` deploys that artifact to the `github-pages` environment. Only the
deployment job receives Pages write and OIDC permissions. Deployment runs are
serialized. Normal builds and release ZIPs keep their origin-root paths and
outputs; release packaging rejects the isolated Pages build profile and non-root
web paths.

Runtime URLs use Vite's `BASE_URL`: scripts, worker chunks, the icon, and model
files resolve under `/gamma_eh/`. `GAMMA_WEB_BASE` accepts only a same-origin
directory path with a trailing slash. It rejects remote URLs, traversal,
queries, and fragments so inference assets cannot be redirected to a CDN.

## Reproduce and verify

With Node 24.x and dependencies installed, run sequentially:

```sh
GAMMA_BUILD_PROFILE=pages GAMMA_WEB_BASE=/gamma_eh/ npm run build
GAMMA_BUILD_PROFILE=pages GAMMA_WEB_BASE=/gamma_eh/ npm run test:playground
GAMMA_BUILD_PROFILE=pages GAMMA_WEB_BASE=/gamma_eh/ GAMMA_TEST_WEBGPU=1 npm run test:playground
```

The browser suite navigates the real subpath and fails if any request escapes
that prefix, uploads text, or uses a different origin. It requires an actual
model-origin correction, checks CPU/WebGPU execution, and exercises the same
editing, failure/retry, offline-after-load, clipboard, and privacy paths as the
local playground. Evidence is under `artifacts/pages/`.

After deployment, run against the public site:

```sh
GAMMA_BUILD_PROFILE=pages GAMMA_WEB_BASE=/gamma_eh/ \
  GAMMA_PLAYGROUND_URL=https://thapecroth.github.io/gamma_eh/ npm run test:playground
```

Set `GAMMA_TEST_WEBGPU=1` for the WebGPU variant. A hosted deployment success and
HTTP 200 are distinct from this live browser verification. The initial page and
weights need a connection; subsequent checks work locally after they load.

## Updates and rollback

Merge a reviewed, passing PR into `main` to update the site. A failed check or
build leaves the existing deployed version in place. For rollback, revert the
specific change through a PR and let the same checks and deployment workflow
publish it. Re-run a successful main workflow to redeploy the same revision.

See [GitHub's workflow deployment guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [Vite's repository-path hosting guide](https://vite.dev/guide/static-deploy.html).
