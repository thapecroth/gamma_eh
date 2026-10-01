import { access, copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as buildScripts } from 'esbuild';
import { build as buildWeb } from 'vite';
import { firefoxManifest } from './firefox-manifest.mjs';
import { buildPaths } from './paths.mjs';
import { runtimeManifest } from './inference-runtime.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const {webOutput, extensionOutput, modelDirectory} = buildPaths(root);
const firefoxOutput = join(dirname(extensionOutput), 'firefox');
const requiredModels = ['model.onnx', 'model_quantized.onnx', 'vocab.txt', 'labels.json', 'manifest.json', 'config.json'];

for (const filename of requiredModels) {
  try { await access(join(modelDirectory, filename)); }
  catch { throw new Error(`Missing local model asset models/browser/${filename}. Run the documented dataset/training/export pipeline before building.`); }
}

await buildWeb({ configFile: join(root, 'apps/web/vite.config.ts'), build: {outDir: webOutput, emptyOutDir: true} });
await rm(extensionOutput, { recursive: true, force: true });
await mkdir(extensionOutput, { recursive: true });

const common = { define: {GAMMA_FIREFOX: 'false'}, bundle: true, minify: true, platform: 'browser', target: 'chrome116', alias: { '@gamma/engine': join(root, 'packages/engine/src/index.ts') }, outdir: extensionOutput, logLevel: 'info' };
await buildScripts({ ...common, format: 'esm', entryPoints: { background: join(root, 'apps/extension/src/background.ts'), offscreen: join(root, 'apps/extension/src/offscreen.ts'), 'inference-worker': join(root, 'apps/extension/src/inference-worker.ts') }, outExtension: { '.js': '.mjs' } });
await buildScripts({ ...common, format: 'iife', entryPoints: { content: join(root, 'apps/extension/src/content.ts'), popup: join(root, 'apps/extension/src/popup.ts') } });

async function copy(source, destination) {
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}

for (const filename of ['manifest.json', 'popup.html', 'popup.css', 'offscreen.html']) await copy(join(root, 'apps/extension', filename), join(extensionOutput, filename));
await rm(firefoxOutput, {recursive: true, force: true});
await mkdir(firefoxOutput, {recursive: true});
const firefox = {...common, outdir: firefoxOutput, target: 'firefox140', define: {chrome: 'browser', GAMMA_FIREFOX: 'true'}};
await buildScripts({...firefox, format: 'esm', entryPoints: {'inference-worker': join(root, 'apps/extension/src/inference-worker.ts')}, outExtension: {'.js': '.mjs'}});
await buildScripts({...firefox, format: 'iife', entryPoints: {background: join(root, 'apps/extension/src/background.ts'), content: join(root, 'apps/extension/src/content.ts'), popup: join(root, 'apps/extension/src/popup.ts')}});
for (const filename of ['popup.html', 'popup.css']) await copy(join(root, 'apps/extension', filename), join(firefoxOutput, filename));
await writeFile(join(firefoxOutput, 'manifest.json'), JSON.stringify(firefoxManifest(JSON.parse(await readFile(join(root, 'apps/extension/manifest.json'), 'utf8'))), null, 2) + '\n');
const models = await readdir(modelDirectory);
for (const output of [webOutput, extensionOutput, firefoxOutput]) {
  for (const filename of models) if ((await stat(join(modelDirectory, filename))).isFile()) await copy(join(modelDirectory, filename), join(output, 'models', filename));
  await writeFile(join(output, 'inference-runtime.json'), JSON.stringify(runtimeManifest, null, 2) + '\n');
}

console.log('Built web editor and Chrome/Firefox extensions with bundled local model/runtime assets.');
