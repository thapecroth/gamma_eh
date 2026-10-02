import { execFile } from 'node:child_process';
import { constants } from 'node:fs';
import { copyFile, lstat, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { firefoxManifest } from './firefox-manifest.mjs';
import { buildPaths } from './paths.mjs';
import { webBase } from './web-base.mjs';
import { runtimeManifest, runtimeVersions } from './inference-runtime.mjs';

const run = promisify(execFile);
const extensionFiles = ['manifest.json', 'background.mjs', 'content.js', 'popup.js', 'popup.html', 'popup.css', 'offscreen.html', 'offscreen.mjs', 'inference-worker.mjs'];
const json = async filename => JSON.parse(await readFile(filename, 'utf8'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const mayPublish = flag => flag === undefined || flag === true;

export async function validateExtensionIcons(directory, manifest) {
  for (const [size, filename] of Object.entries(manifest.icons ?? {})) {
    if (![16, 32, 48, 128].includes(Number(size)) || filename !== `icons/icon-${size}.png`) throw new Error('Unexpected extension icon path or size.');
    const bytes = await readFile(join(directory, filename));
    if (bytes.length < 24 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
        || bytes.toString('ascii', 12, 16) !== 'IHDR' || bytes.readUInt32BE(16) !== Number(size) || bytes.readUInt32BE(20) !== Number(size)) {
      throw new Error(`Invalid PNG dimensions for ${filename}.`);
    }
  }
  for (const size of [16, 32, 48, 128]) if (!manifest.icons?.[size]) throw new Error(`Missing ${size}px extension icon.`);
  for (const size of [16, 32]) if (manifest.action?.default_icon?.[size] !== manifest.icons[size]) throw new Error('Toolbar icons must match bundled extension icons.');
}

export function releaseVersion(packageVersion, extensionVersion, tag = `v${packageVersion}`) {
  if (typeof packageVersion !== 'string' || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.test(packageVersion)
      || packageVersion === '0.0.0' || packageVersion.split('.').some(part => Number(part) > 65535)) {
    throw new Error('Release version must be a nonzero Chrome-compatible X.Y.Z version (components 0–65535).');
  }
  if (extensionVersion !== packageVersion || tag !== `v${packageVersion}`) {
    throw new Error('Release tag, package.json version, and extension manifest version must match.');
  }
  return packageVersion;
}

export function publicBuildPaths(root, environment = {}) {
  const paths = buildPaths(root, environment);
  if (webBase(environment) !== '/') {
    throw new Error('Release archives must use the root web path; repository paths belong in the Pages build.');
  }
  if (environment.GAMMA_BUILD_PROFILE || paths.modelDirectory !== join(root, 'models/browser')) {
    throw new Error('Releases must use the default public model, never an experimental build profile.');
  }
  return paths;
}

// Never follow a symlink, include a hidden file, or copy a training checkpoint.
async function filesBelow(directory, prefix = '') {
  const info = await lstat(directory);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`Expected an ordinary directory: ${directory}`);
  const files = [];
  for (const name of (await readdir(directory)).sort()) {
    if (name.startsWith('.')) throw new Error(`Hidden release asset refused: ${name}`);
    const filename = join(directory, name);
    const entry = await lstat(filename);
    if (entry.isSymbolicLink()) throw new Error(`Symlink release asset refused: ${filename}`);
    const relative = prefix ? `${prefix}/${name}` : name;
    if (entry.isDirectory()) files.push(...await filesBelow(filename, relative));
    else if (entry.isFile()) files.push(relative);
    else throw new Error(`Non-file release asset refused: ${filename}`);
  }
  return files;
}

async function publicModel(root) {
  const directory = join(root, 'models/browser');
  const files = await filesBelow(directory);
  const manifest = await json(join(directory, 'manifest.json'));
  const dataset = await json(join(directory, 'dataset-manifest.json'));
  if (manifest.model_license !== 'Apache-2.0' || !mayPublish(manifest.publication_allowed)
      || !mayPublish(dataset.publication_allowed) || dataset.license !== 'CC0-1.0'
      || dataset.origin !== 'original-template-v1') {
    throw new Error('Public release requires the reviewed Apache-2.0 model and original CC0 baseline provenance.');
  }
  const entries = Object.entries(manifest.files ?? {});
  for (const required of ['model.onnx', 'model_quantized.onnx', 'vocab.txt', 'labels.json', 'config.json']) {
    if (!entries.some(([name]) => name === required)) throw new Error(`Model manifest is missing ${required}`);
  }
  const allowed = new Set(['manifest.json', 'dataset-manifest.json', 'evaluation.json']);
  for (const [filename, evidence] of entries) {
    if (!evidence || !/^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_.-]+$/u.test(filename) || !/^[a-f0-9]{64}$/u.test(evidence.sha256)
        || !Number.isSafeInteger(evidence.bytes) || evidence.bytes <= 0) throw new Error('Invalid model asset manifest.');
    allowed.add(filename);
    const bytes = await readFile(join(directory, filename));
    if (bytes.length !== evidence.bytes || hash(bytes) !== evidence.sha256) throw new Error(`Model hash mismatch: ${filename}`);
  }
  for (const filename of files) if (!allowed.has(filename)) throw new Error(`Unreviewed model asset refused: ${filename}`);
  return {directory, files};
}

async function metadata(root, environment, tag) {
  const paths = publicBuildPaths(root, environment);
  const pkg = await json(join(root, 'package.json'));
  const extension = await json(join(root, 'apps/extension/manifest.json'));
  const version = releaseVersion(pkg.version, extension.version, tag);
  for (const [name, version] of Object.entries(runtimeVersions)) {
    if ((pkg.dependencies?.[name] ?? pkg.overrides?.[name]) !== version
        || (await json(join(root, 'node_modules', name, 'package.json'))).version !== version) {
      throw new Error(`JAX runtime version for ${name} must match the bundled upstream notices.`);
    }
  }
  return {...paths, version, extension, model: await publicModel(root)};
}

async function copy(source, destination) {
  if (!(await lstat(source)).isFile()) throw new Error(`Expected an ordinary release file: ${source}`);
  await mkdir(dirname(destination), {recursive: true});
  await copyFile(source, destination, constants.COPYFILE_EXCL);
}

async function releaseDirectory(root, version) {
  // Check each parent before descending, so cleanup cannot escape via a symlink.
  for (const directory of [join(root, 'artifacts'), join(root, 'artifacts/releases')]) {
    await mkdir(directory, {recursive: true});
    const info = await lstat(directory);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('Release output parent must be an ordinary directory.');
  }
  return join(root, 'artifacts/releases', `v${version}`);
}

async function licenses(root, stage, web, version) {
  for (const filename of ['LICENSE', 'NOTICE']) await copy(join(root, filename), join(stage, filename));
  const card = (await readFile(join(root, 'models/MODEL_CARD.md'), 'utf8'))
    .replaceAll('](browser/', '](models/')
    .replaceAll('](../docs/', `](https://github.com/thapecroth/gamma_eh/blob/v${version}/docs/`);
  await writeFile(join(stage, 'MODEL_CARD.md'), card, {flag: 'wx'});
  for (const filename of ['jax-js/LICENSE', 'jax-js/README.md', 'onnx/LICENSE', 'onnx/README.md',
    'protobuf/LICENSE', 'protobuf/BSD-3-Clause.txt', 'protobuf/README.md']) {
    await copy(join(root, 'licenses', filename), join(stage, 'licenses', filename));
  }
  for (const filename of ['README.md', 'SymSpell-LICENSE', 'SCOWL-Copyright']) {
    await copy(join(root, 'licenses/spelling', filename), join(stage, 'licenses/spelling', filename));
  }
  for (const name of web ? ['react', 'react-dom', 'scheduler'] : []) {
    await copy(join(root, 'node_modules', name, 'LICENSE'), join(stage, 'licenses', name, 'LICENSE'));
  }
}

function installInstructions(version, web, firefox = false) {
  if (firefox) return `# Gamma EH ${version} — Firefox extension\n\nUse Firefox 140 or newer. Extract this ZIP, open about:debugging#/runtime/this-firefox,\nchoose Load Temporary Add-on and select manifest.json. Open or reload a normal website. Suggestions start automatically.\nTemporary installs disappear on restart. Permanent installation requires Mozilla signing.\nLocal AI is opt-in; use the popup to pause checking globally or on the current site.\n`;

  return web ? `# Gamma EH ${version} — web editor\n\nServe this extracted directory with a static HTTP server on localhost or HTTPS.\nDo not open index.html as file://; workers and WebGPU need a secure origin.\nFor example, with Python 3: python -m http.server 8080 --bind 127.0.0.1\nThen open http://localhost:8080. Enable Local AI to try the model.\n`
    : `# Gamma EH ${version} — Chrome extension\n\n1. Extract the ZIP into a permanent folder. No build tools are required.\n2. Open chrome://extensions in Chrome 116 or newer.\n3. Enable Developer mode, click Load unpacked, and select this folder\n   (the one containing manifest.json). Keep the folder after installation.\n4. Open or reload a normal website. Suggestions start automatically.\n5. Use a textarea or plain-text contenteditable. Local AI is opt-in.\n\nTo update an unpacked install, replace the contents of the same folder,\nclick Reload on its card in chrome://extensions, and refresh website tabs.\nRemoving the installation resets settings and site permissions.\nUnpacked installs do not auto-update. Once a Chrome Web Store listing is\navailable, switch to its store install once to receive automatic updates.\nSee https://github.com/thapecroth/gamma_eh/blob/main/docs/chrome-web-store.md\nfor publication status and instructions. This ZIP is also the store upload\npackage; downloading a ZIP does not install a store-managed extension.\n`;
}

// Separate from the CLI's mandatory fresh build so archive safety is testable
// with tiny fixtures. Publishing uses only the CLI, never this function alone.
export async function packageBuiltRelease(root, {environment = {}, tag} = {}) {
  root = resolve(root);
  const meta = await metadata(root, environment, tag);
  const output = await releaseDirectory(root, meta.version);
  const temporary = await mkdtemp(join(tmpdir(), 'gamma-release-'));
  let reserved = false;
  try {
    // Reserve the version without overwriting any existing assets.
    await mkdir(output);
    reserved = true;
    const sums = [];
    for (const [kind, directory] of [['chrome', meta.extensionOutput], ['firefox', join(dirname(meta.extensionOutput), 'firefox')], ['web', meta.webOutput]]) {
      const files = await filesBelow(directory);
      const required = kind === 'chrome' ? extensionFiles : kind === 'firefox' ? ['manifest.json', 'background.js', 'content.js', 'popup.js', 'popup.html', 'popup.css', 'inference-worker.mjs'] : ['index.html'];
      for (const filename of [...required, 'inference-runtime.json']) {
        if (!files.includes(filename)) throw new Error(`Incomplete ${kind} build: missing ${filename}`);
      }
      if (JSON.stringify(await json(join(directory, 'inference-runtime.json'))) !== JSON.stringify(runtimeManifest)) {
        throw new Error('Built JAX runtime manifest differs from the pinned runtime.');
      }
      if (files.some(name => name.startsWith('runtime/'))) throw new Error('Obsolete external runtime assets must not ship with JAX.');
      if (kind !== 'web') {
        const built = await json(join(directory, 'manifest.json'));
        if (JSON.stringify(built) !== JSON.stringify(kind === 'firefox' ? firefoxManifest(meta.extension) : meta.extension)) throw new Error('Built extension manifest differs from its source.');
        await validateExtensionIcons(directory, built);
      }
      const modelFiles = files.filter(name => name.startsWith('models/')).map(name => name.slice(7));
      if (JSON.stringify(modelFiles.sort()) !== JSON.stringify([...meta.model.files].sort())) throw new Error('Built model contains missing or unreviewed assets.');
      for (const filename of meta.model.files) {
        if (!(await readFile(join(directory, 'models', filename))).equals(await readFile(join(meta.model.directory, filename)))) {
          throw new Error(`Built model differs from the reviewed source: ${filename}`);
        }
      }
      const stage = join(temporary, kind);
      await mkdir(stage);
      for (const filename of files) await copy(join(directory, filename), join(stage, filename));
      await licenses(root, stage, kind === 'web', meta.version);
      await writeFile(join(stage, 'INSTALL.md'), installInstructions(meta.version, kind === 'web', kind === 'firefox')
        + '\nExperimental synthetic-template baseline; not general grammar accuracy.\nText stays local. WebGPU is preferred with WASM fallback.\nSee MODEL_CARD.md for limitations and licenses/ for upstream notices.\n', {flag: 'wx'});
      const asset = `gamma-eh-${kind}-v${meta.version}.zip`;
      await run('zip', ['-q', '-r', '-X', join(output, asset), '.'], {cwd: stage});
      await run('unzip', ['-tq', join(output, asset)]);
      sums.push(`${hash(await readFile(join(output, asset)))}  ${asset}`);
    }
    await writeFile(join(output, 'SHA256SUMS.txt'), sums.join('\n') + '\n', {flag: 'wx'});
    await writeFile(join(output, 'RELEASE_NOTES.md'), `# Gamma EH v${meta.version} (experimental)\n\n`
      + 'Download gamma-eh-chrome-v' + meta.version + '.zip, extract it, and load the folder\n'
      + 'in chrome://extensions → Developer mode → Load unpacked. No Node.js or build step.\n'
      + 'Keep the folder on disk, grant checking per site, and opt into Local AI if desired.\n\n'
      + 'The Firefox ZIP loads temporarily via about:debugging; permanent installs require Mozilla signing.\n\n'
      + 'The web ZIP is a static editor bundle; serve it on localhost or HTTPS.\n'
      + 'SHA256SUMS.txt contains checksums for all three ZIPs. Licenses and instructions are inside.\n\n'
      + 'This is an experimental model trained on original CC0 synthetic templates.\n'
      + 'It is not yet a general Grammarly replacement. This release is gated on actual\n'
      + 'WASM/WebGPU web-editor and extension checks. Hosted WebGPU uses SwiftShader\n'
      + 'software; those checks are not physical-GPU benchmarks or real-world accuracy.\n\n'
      + 'ZIP downloads are developer-mode installations and do not auto-update.\n'
      + 'For Chrome Web Store publication status and automatic-update setup, see\n'
      + 'https://github.com/thapecroth/gamma_eh/blob/main/docs/chrome-web-store.md\n', {flag: 'wx'});
    return output;
  } catch (error) {
    if (reserved) await rm(output, {recursive: true, force: true}); // Only our newly reserved, incomplete version.
    throw error;
  } finally {
    await rm(temporary, {recursive: true, force: true});
  }
}

async function main() {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const environment = process.env;
  const tag = environment.GAMMA_RELEASE_TAG;
  await metadata(root, environment, tag); // Fail closed before building private assets.
  await import('./build.mjs');
  console.log(`Release packages: ${await packageBuiltRelease(root, {environment, tag})}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
