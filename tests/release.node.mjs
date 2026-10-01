import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFile, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import { packageBuiltRelease, publicBuildPaths, releaseVersion } from '../scripts/package-release.mjs';
import { runtimeManifest, runtimeVersions } from '../scripts/inference-runtime.mjs';
import { verifiedArchive } from '../scripts/webstore-submit.mjs';

const run = promisify(execFile);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

test('vendored JAX licenses match the installed runtime', async () => {
  const notice = await readFile(new URL('../licenses/jax-js/LICENSE', import.meta.url));
  for (const name of ['jax', 'onnx']) {
    assert(notice.equals(await readFile(new URL(`../node_modules/@jax-js/${name}/LICENSE`, import.meta.url))));
  }
});

async function put(root, filename, content) {
  await mkdir(dirname(join(root, filename)), {recursive: true});
  await writeFile(join(root, filename), typeof content === 'string' ? content : JSON.stringify(content));
}

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'gamma-release-test-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  await put(root, 'package.json', {version: '0.1.0', dependencies: runtimeVersions});
  const {icons, action} = JSON.parse(await readFile(new URL('../apps/extension/manifest.json', import.meta.url), 'utf8'));
  const extension = {manifest_version: 3, name: 'Fictional fixture', version: '0.1.0', icons, action};
  await put(root, 'apps/extension/manifest.json', extension);
  for (const [name, version] of Object.entries(runtimeVersions)) await put(root, `node_modules/${name}/package.json`, {version});
  const model = {model_license: 'Apache-2.0', files: {}};
  for (const name of ['model.onnx', 'model_quantized.onnx', 'vocab.txt', 'labels.json', 'config.json']) {
    const bytes = Buffer.from(`Fictional ${name}; not an executable model`);
    model.files[name] = {bytes: bytes.length, sha256: hash(bytes)};
    await put(root, `models/browser/${name}`, bytes.toString());
  }
  await put(root, 'models/browser/manifest.json', model);
  await put(root, 'models/browser/dataset-manifest.json', {license: 'CC0-1.0', origin: 'original-template-v1'});
  for (const name of ['LICENSE', 'NOTICE', 'licenses/jax-js/LICENSE', 'licenses/jax-js/README.md',
    'licenses/spelling/README.md', 'licenses/spelling/SymSpell-LICENSE', 'licenses/spelling/SCOWL-Copyright',
    'licenses/onnx/LICENSE', 'licenses/onnx/README.md',
    'licenses/protobuf/LICENSE', 'licenses/protobuf/BSD-3-Clause.txt', 'licenses/protobuf/README.md', 'node_modules/react/LICENSE', 'node_modules/react-dom/LICENSE', 'node_modules/scheduler/LICENSE']) {
    await put(root, name, `Fictional notice: ${name}\n`);
  }
  await put(root, 'models/MODEL_CARD.md', 'Fictional card: [data](browser/dataset-manifest.json) [docs](../docs/massive-dataset.md)\n');
  for (const kind of ['extension', 'web']) {
    for (const name of [...Object.keys(model.files), 'manifest.json', 'dataset-manifest.json']) {
      await mkdir(join(root, 'dist', kind, 'models'), {recursive: true});
      await copyFile(join(root, 'models/browser', name), join(root, 'dist', kind, 'models', name));
    }
    await put(root, `dist/${kind}/inference-runtime.json`, runtimeManifest);
  }
  await put(root, 'dist/web/index.html', '<!doctype html><title>Fictional editor</title>');
  await put(root, 'dist/extension/manifest.json', extension);
  await mkdir(join(root, 'dist/extension/icons'), {recursive: true});
  for (const filename of Object.values(icons)) await copyFile(new URL(`../apps/extension/${filename}`, import.meta.url), join(root, 'dist/extension', filename));
  for (const name of ['background.mjs', 'content.js', 'popup.js', 'popup.html', 'popup.css', 'offscreen.html', 'offscreen.mjs', 'inference-worker.mjs']) {
    await put(root, `dist/extension/${name}`, `Fictional asset: ${name}`);
  }
  return {root, model};
}

test('store upload uses the packaged version and refuses a modified archive', async t => {
  const {root} = await fixture(t);
  const directory = await packageBuiltRelease(root);
  assert((await verifiedArchive(directory, '0.1.0')).length > 0);
  await assert.rejects(verifiedArchive(directory, '../escape'), /Chrome-compatible/u);
  await writeFile(join(directory, 'gamma-eh-chrome-v0.1.0.zip'), 'modified archive');
  await assert.rejects(verifiedArchive(directory, '0.1.0'), /checksum/u);
});

test('release packaging refuses a missing or incorrectly sized store icon', async t => {
  const {root} = await fixture(t);
  const filename = join(root, 'dist/extension/icons/icon-128.png');
  const icon = await readFile(filename);
  icon.writeUInt32BE(64, 16);
  await writeFile(filename, icon);
  await assert.rejects(packageBuiltRelease(root), /PNG dimensions/u);
  await rm(filename);
  await assert.rejects(packageBuiltRelease(root), /ENOENT/u);
});

test('validates Chrome-compatible versions and exact matching release tags', () => {
  assert.equal(releaseVersion('0.1.0', '0.1.0', 'v0.1.0'), '0.1.0');
  assert.equal(releaseVersion('65535.0.1', '65535.0.1'), '65535.0.1');
  for (const version of ['0.0.0', '01.2.3', '1.2', '1.2.3-beta', '1.2.65536', '../escape', '', undefined]) {
    assert.throws(() => releaseVersion(version, version), /Chrome-compatible/u);
  }
  assert.throws(() => releaseVersion('0.1.0', '0.2.0'), /must match/u);
  assert.throws(() => releaseVersion('0.1.0', '0.1.0', 'v0.1.1'), /must match/u);
});

test('rejects private model directories and experimental profiles', () => {
  assert.equal(publicBuildPaths('/project').modelDirectory, '/project/models/browser');
  assert.throws(() => publicBuildPaths('/project', {GAMMA_BUILD_PROFILE: 'pilot'}), /default public/u);
  assert.throws(() => publicBuildPaths('/project', {GAMMA_MODEL_DIR: 'artifacts/private/model'}), /isolated/u);
});

test('packages root-level manifests, local inference assets, instructions, and licenses with valid checksums', async t => {
  const {root} = await fixture(t);
  // Private artifacts outside the shipping build must not be picked up.
  await put(root, 'artifacts/private/model/secret.onnx', 'must not ship');
  await put(root, 'dist/pilot/extension/private.txt', 'must not ship');
  const output = await packageBuiltRelease(root, {tag: 'v0.1.0'});
  const sums = await readFile(join(output, 'SHA256SUMS.txt'), 'utf8');
  for (const kind of ['chrome', 'web']) {
    const name = `gamma-eh-${kind}-v0.1.0.zip`;
    assert(sums.includes(`${hash(await readFile(join(output, name)))}  ${name}\n`));
    const {stdout: listing} = await run('unzip', ['-Z1', join(output, name)]);
    const names = new Set(listing.trim().split('\n'));
    for (const required of ['INSTALL.md', 'LICENSE', 'NOTICE', 'MODEL_CARD.md', 'models/model.onnx',
      'models/model_quantized.onnx', 'inference-runtime.json',
      'licenses/jax-js/LICENSE', 'licenses/onnx/LICENSE', 'licenses/protobuf/LICENSE', 'licenses/protobuf/BSD-3-Clause.txt',
      'licenses/spelling/README.md', 'licenses/spelling/SymSpell-LICENSE', 'licenses/spelling/SCOWL-Copyright']) assert(names.has(required), required);
    assert(!listing.includes('private') && !listing.includes('secret') && !listing.includes('pilot'));
    const {stdout: instructions} = await run('unzip', ['-p', join(output, name), 'INSTALL.md']);
    assert.match(instructions, /Experimental synthetic-template baseline/u);
    assert.match(instructions, kind === 'chrome' ? /Load unpacked/u : /secure origin/u);
    const {stdout: card} = await run('unzip', ['-p', join(output, name), 'MODEL_CARD.md']);
    assert(card.includes('](models/dataset-manifest.json)'));
    assert(card.includes('](https://github.com/thapecroth/gamma_eh/blob/v0.1.0/docs/massive-dataset.md)'));
    if (kind === 'chrome') {
      assert(names.has('manifest.json'));
      assert(!names.has('extension/manifest.json'));
      const {stdout: manifest} = await run('unzip', ['-p', join(output, name), 'manifest.json']);
      assert.equal(JSON.parse(manifest).version, '0.1.0');
    } else assert(names.has('licenses/react/LICENSE'));
    const extracted = join(root, 'unpacked', kind);
    await mkdir(extracted, {recursive: true});
    await run('unzip', ['-q', join(output, name), '-d', extracted]);
    assert((await readFile(join(extracted, 'models/model.onnx'))).equals(await readFile(join(root, 'models/browser/model.onnx'))));
  }
  const before = await readFile(join(output, 'SHA256SUMS.txt'));
  await assert.rejects(packageBuiltRelease(root), {code: 'EEXIST'});
  assert((await readFile(join(output, 'SHA256SUMS.txt'))).equals(before), 'Existing release assets are immutable');
});

test('rejects unpublished provider-derived weights before creating release assets', async t => {
  const {root, model} = await fixture(t);
  await put(root, 'models/browser/manifest.json', {...model, publication_allowed: false});
  await assert.rejects(packageBuiltRelease(root), /reviewed Apache-2.0/u);
});

test('treats nonboolean publication flags as unreviewed', async t => {
  const {root, model} = await fixture(t);
  await put(root, 'models/browser/manifest.json', {...model, publication_allowed: 'provider-terms-unverified'});
  await assert.rejects(packageBuiltRelease(root), /reviewed Apache-2.0/u);
});

test('rejects unreviewed corpus provenance', async t => {
  const {root} = await fixture(t);
  await put(root, 'models/browser/dataset-manifest.json', {license: 'provider-terms-unverified', origin: 'teacher'});
  await assert.rejects(packageBuiltRelease(root), /baseline provenance/u);
});

test('rejects source model tampering', async t => {
  const {root} = await fixture(t);
  await put(root, 'models/browser/model.onnx', 'tampered weights');
  await assert.rejects(packageBuiltRelease(root), /Model hash mismatch/u);
});

test('rejects stale or swapped built weights and removes only the incomplete output', async t => {
  const {root} = await fixture(t);
  await put(root, 'dist/extension/models/model.onnx', 'stale weights');
  await assert.rejects(packageBuiltRelease(root), /differs from the reviewed source/u);
  await assert.rejects(readFile(join(root, 'artifacts/releases/v0.1.0/SHA256SUMS.txt')), {code: 'ENOENT'});
  assert.match(await readFile(join(root, 'LICENSE'), 'utf8'), /Fictional notice/u);
});

test('rejects symlinks in built assets', async t => {
  const {root} = await fixture(t);
  await symlink(join(root, 'LICENSE'), join(root, 'dist/extension/secret.txt'));
  await assert.rejects(packageBuiltRelease(root), /Symlink release asset/u);
});

test('rejects a symlink output parent without modifying its target', async t => {
  const {root} = await fixture(t);
  await put(root, 'external-marker/preserve.txt', 'keep this');
  await symlink(join(root, 'external-marker'), join(root, 'artifacts'));
  await assert.rejects(packageBuiltRelease(root), /output parent must be an ordinary directory/u);
  assert.equal(await readFile(join(root, 'external-marker/preserve.txt'), 'utf8'), 'keep this');
});

test('rejects hidden files instead of accidentally releasing credentials', async t => {
  const {root} = await fixture(t);
  await put(root, 'dist/extension/.env', 'FICTIONAL_TOKEN=not-a-secret');
  await assert.rejects(packageBuiltRelease(root), /Hidden release asset/u);
});

test('rejects extra model files', async t => {
  const {root} = await fixture(t);
  await put(root, 'models/browser/unreviewed.json', {private: true});
  await assert.rejects(packageBuiltRelease(root), /Unreviewed model asset/u);
});

test('rejects a missing bundled runtime manifest', async t => {
  const {root} = await fixture(t);
  await rm(join(root, 'dist/extension/inference-runtime.json'));
  await assert.rejects(packageBuiltRelease(root), /Incomplete chrome build/u);
});

test('rejects a test-only extension permission manifest', async t => {
  const {root} = await fixture(t);
  await put(root, 'dist/extension/manifest.json', {manifest_version: 3, version: '0.1.0', host_permissions: ['http://localhost/*']});
  await assert.rejects(packageBuiltRelease(root), /manifest differs/u);
});

test('requires matching runtime notices before packaging', async t => {
  const {root} = await fixture(t);
  await put(root, 'node_modules/@jax-js/jax/package.json', {version: '0.1.26'});
  await assert.rejects(packageBuiltRelease(root), /upstream notices/u);
});

test('rejects stale runtime metadata and leftover ONNX Runtime files', async t => {
  const {root} = await fixture(t);
  await put(root, 'dist/extension/inference-runtime.json', {...runtimeManifest, engine: 'onnxruntime'});
  await assert.rejects(packageBuiltRelease(root), /runtime manifest differs/u);
  await put(root, 'dist/extension/inference-runtime.json', runtimeManifest);
  await put(root, 'dist/extension/runtime/obsolete.wasm', 'must not ship');
  await assert.rejects(packageBuiltRelease(root), /Obsolete external runtime/u);
});
