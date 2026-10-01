import { access, copyFile, mkdir, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as buildScripts } from 'esbuild';
import { build as buildWeb } from 'vite';
import { buildPaths } from './paths.mjs';
import { runtimeManifest } from './inference-runtime.mjs';
import { webGPUAdapterEsbuildPlugin } from './webgpu-adapter-options.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const {webOutput, extensionOutput, modelDirectory} = buildPaths(root);
const requiredModels = ['model.onnx', 'model_quantized.onnx', 'vocab.txt', 'labels.json', 'manifest.json', 'config.json'];

for (const filename of requiredModels) {
  try { await access(join(modelDirectory, filename)); }
  catch { throw new Error(`Missing local model asset models/browser/${filename}. Run the documented dataset/training/export pipeline before building.`); }
}

await buildWeb({ configFile: join(root, 'apps/web/vite.config.ts'), build: {outDir: webOutput, emptyOutDir: true} });
await rm(extensionOutput, { recursive: true, force: true });
await mkdir(extensionOutput, { recursive: true });

const common = { plugins: [webGPUAdapterEsbuildPlugin()], bundle: true, minify: true, platform: 'browser', target: 'chrome116', alias: { '@gamma/engine': join(root, 'packages/engine/src/index.ts') }, outdir: extensionOutput, logLevel: 'info' };
await buildScripts({ ...common, format: 'esm', entryPoints: { background: join(root, 'apps/extension/src/background.ts'), offscreen: join(root, 'apps/extension/src/offscreen.ts'), 'inference-worker': join(root, 'apps/extension/src/inference-worker.ts') }, outExtension: { '.js': '.mjs' } });
await buildScripts({ ...common, format: 'iife', entryPoints: { content: join(root, 'apps/extension/src/content.ts'), popup: join(root, 'apps/extension/src/popup.ts') } });

async function copy(source, destination) {
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}

for (const filename of ['manifest.json', 'popup.html', 'popup.css', 'offscreen.html']) await copy(join(root, 'apps/extension', filename), join(extensionOutput, filename));
for (const size of [16, 32, 48, 128]) await copy(join(root, 'apps/extension/icons', `icon-${size}.png`), join(extensionOutput, 'icons', `icon-${size}.png`));
const models = await readdir(modelDirectory);
for (const output of [webOutput, extensionOutput]) {
  for (const filename of models) if ((await stat(join(modelDirectory, filename))).isFile()) await copy(join(modelDirectory, filename), join(output, 'models', filename));
  await writeFile(join(output, 'inference-runtime.json'), JSON.stringify(runtimeManifest, null, 2) + '\n');
}

console.log('Built web editor and Chrome extension with bundled local model/runtime assets.');
