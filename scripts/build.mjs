import { access, copyFile, mkdir, readdir, rm, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as buildScripts } from 'esbuild';
import { build as buildWeb } from 'vite';

const root = fileURLToPath(new URL('../', import.meta.url));
const webOutput = join(root, 'dist/web');
const extensionOutput = join(root, 'dist/extension');
const modelDirectory = join(root, 'models/browser');
const runtimeDirectory = join(root, 'node_modules/onnxruntime-web/dist');
const requiredModels = ['model.onnx', 'model_quantized.onnx', 'vocab.txt', 'labels.json', 'manifest.json', 'config.json'];
// ORT1.30's native WebGPU runtime uses asyncify. jsep belongs to the older
// JavaScript execution provider and is not the graph runtime imported by us.
const requiredRuntime = ['ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.asyncify.mjs'];

for (const filename of requiredModels) {
  try { await access(join(modelDirectory, filename)); }
  catch { throw new Error(`Missing local model asset models/browser/${filename}. Run the documented dataset/training/export pipeline before building.`); }
}
for (const filename of requiredRuntime) {
  try { await access(join(runtimeDirectory, filename)); }
  catch { throw new Error(`Missing ONNX runtime asset ${filename}. Install dependencies before building.`); }
}

await buildWeb({ configFile: join(root, 'apps/web/vite.config.ts') });
await rm(extensionOutput, { recursive: true, force: true });
await mkdir(extensionOutput, { recursive: true });

const common = { bundle: true, minify: true, platform: 'browser', target: 'chrome116', alias: { '@gamma/engine': join(root, 'packages/engine/src/index.ts') }, outdir: extensionOutput, logLevel: 'info' };
await buildScripts({ ...common, format: 'esm', entryPoints: { background: join(root, 'apps/extension/src/background.ts'), offscreen: join(root, 'apps/extension/src/offscreen.ts'), 'inference-worker': join(root, 'apps/extension/src/inference-worker.ts') }, outExtension: { '.js': '.mjs' } });
await buildScripts({ ...common, format: 'iife', entryPoints: { content: join(root, 'apps/extension/src/content.ts'), popup: join(root, 'apps/extension/src/popup.ts') } });

async function copy(source, destination) {
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}

for (const filename of ['manifest.json', 'popup.html', 'popup.css', 'offscreen.html']) await copy(join(root, 'apps/extension', filename), join(extensionOutput, filename));
const models = await readdir(modelDirectory);
const optionalRuntime = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];
for (const output of [webOutput, extensionOutput]) {
  for (const filename of models) if ((await stat(join(modelDirectory, filename))).isFile()) await copy(join(modelDirectory, filename), join(output, 'models', filename));
  for (const filename of requiredRuntime) await copy(join(runtimeDirectory, filename), join(output, 'runtime', filename));
  for (const filename of optionalRuntime) {
    try { await access(join(runtimeDirectory, filename)); }
    catch { continue; }
    await copy(join(runtimeDirectory, filename), join(output, 'runtime', filename));
  }
}

console.log('Built web editor and Chrome extension with bundled local model/runtime assets.');
