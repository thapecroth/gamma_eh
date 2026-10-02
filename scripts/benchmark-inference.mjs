import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createServer} from 'node:http';
import {mkdtemp, mkdir, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import {chromium} from '@playwright/test';
import {browserArguments, findChromium, withLocalHttp} from './browser-environment.mjs';
import {buildPaths} from './paths.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const {modelDirectory, artifactDir} = buildPaths(root);
const preferWebGPU = process.env.GAMMA_TEST_WEBGPU === '1';
const roots = process.env.GAMMA_BENCHMARK_BASELINE
  ? {baseline: resolve(process.env.GAMMA_BENCHMARK_BASELINE), optimized: root} : {current: root};
const assets = new Map([['/', Buffer.from('<!doctype html><title>Local inference benchmark</title>')]]);
const temporary = await mkdtemp(join(tmpdir(), 'gamma-inference-benchmark-'));
const evidence = {schema: 1, measuredAt: new Date().toISOString(), requestedBackend: preferWebGPU ? 'webgpu' : 'wasm', profiles: {}};
let browser;

try {
  for (const name of ['manifest.json', 'labels.json', 'vocab.txt', 'model.onnx']) {
    assets.set('/models/' + name, await readFile(join(modelDirectory, name)));
  }
  const manifest = JSON.parse(assets.get('/models/manifest.json').toString());
  evidence.modelSha256 = createHash('sha256').update(assets.get('/models/model.onnx')).digest('hex');
  assert.equal(evidence.modelSha256, manifest.files['model.onnx'].sha256);
  assert.notEqual(manifest.disableModelEdits, true, 'Benchmark requires an enabled model');
  for (const [profile, sourceRoot] of Object.entries(roots)) {
    const outfile = join(temporary, profile + '.mjs');
    await build({stdin: {contents: `export * from './packages/engine/src/index.ts'; export {JaxSession} from './packages/engine/src/jax-runtime.ts';`, resolveDir: sourceRoot},
      outfile, bundle: true, format: 'esm', platform: 'browser', target: 'chrome116'});
    assets.set('/' + profile + '.mjs', await readFile(outfile));
    const sourceHash = createHash('sha256');
    for (const name of (await readdir(join(sourceRoot, 'packages/engine/src'))).sort()) {
      sourceHash.update(name + '\0');
      sourceHash.update(await readFile(join(sourceRoot, 'packages/engine/src', name)));
    }
    evidence.profiles[profile] = {engineSourceSha256: sourceHash.digest('hex')};
  }
  await withLocalHttp(async origin => {
    browser = await chromium.launch({executablePath: await findChromium(), headless: true, args: browserArguments()});
    evidence.chromiumVersion = browser.version();
    const regression = JSON.parse(await readFile(join(root, 'data/regression.json'), 'utf8'));
    for (const profile of Object.keys(roots)) {
      const context = await browser.newContext();
      const external = [];
      const errors = [];
      await context.route('**/*', route => {
        if (!route.request().url().startsWith(origin)) { external.push(route.request().url()); return route.abort(); }
        return route.continue();
      });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(origin);
      const results = await page.evaluate(async ({origin, profile, preferWebGPU, cases}) => {
        const engine = await import(origin + profile + '.mjs');
        const options = {modelBaseUrl: origin + 'models/', preferWebGPU};
        const runs = [];
        const originalRun = engine.JaxSession.prototype.run;
        engine.JaxSession.prototype.run = async function(ids) {
          const start = performance.now();
          const output = await originalRun.call(this, ids);
          runs.push({tokens: ids.length, elapsedMs: performance.now() - start});
          return output;
        };
        const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
        async function check(text) {
          runs.length = 0;
          const result = await engine.analyzeText(text, options);
          if (result.modelError || result.backend !== (preferWebGPU ? 'webgpu' : 'wasm')) throw new Error(result.modelError ?? `Unexpected backend ${result.backend}`);
          return {elapsedMs: result.elapsedMs, inferenceMs: runs.reduce((sum, row) => sum + row.elapsedMs, 0),
            modelRuns: result.modelRuns, runtimeCalls: runs.length, backend: result.backend, suggestions: result.suggestions,
            corrected: engine.applySuggestions(text, result.suggestions)};
        }
        // Distinct token windows avoid measuring a repeated-sentence best case.
        const paragraph = Array.from({length: 24}, (_, i) => `The students has ${i + 1} notebooks in the classroom.`).join(' ');
        const cold = await check(paragraph);
        const unchanged = [];
        const edited = [];
        for (let i = 0; i < 9; i++) {
          unchanged.push(await check(paragraph));
          edited.push(await check(paragraph.replace('24 notebooks', `${i + 30} notebooks`)));
        }
        const bytes = new Uint8Array(await (await fetch(origin + 'models/model.onnx')).arrayBuffer());
        const session = await engine.JaxSession.create(bytes, preferWebGPU);
        const firstShapes = [];
        const warmShapes = [];
        const logits = [];
        try {
          for (const measurements of [firstShapes, warmShapes]) {
            for (let length = 8; length <= 16; length++) {
              const start = performance.now();
              await session.run([101, ...Array(length - 2).fill(1996), 102]);
              measurements.push(performance.now() - start);
            }
          }
          for (const length of [3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64]) {
            const output = await session.run([101, ...Array(length - 2).fill(1996), 102]);
            if (output.dims[1] !== length) throw new Error('Padding leaked into output shape');
            logits.push({length, dims: output.dims, values: Array.from(output.data)});
          }
        } finally { session.dispose(); }
        const rows = [];
        for (const item of cases) {
          const result = await check(item.source);
          if (result.corrected !== item.target) throw new Error(`Regression failed: ${item.source}`);
          rows.push({source: item.source, corrected: result.corrected, suggestions: result.suggestions});
        }
        let adapter = null;
        if (preferWebGPU) {
          const gpu = await navigator.gpu.requestAdapter();
          if (gpu) adapter = {vendor: gpu.info.vendor, architecture: gpu.info.architecture, description: gpu.info.description, isFallbackAdapter: gpu.info.isFallbackAdapter};
        }
        const summarize = rows => ({medianMs: median(rows.map(row => row.elapsedMs)),
          medianInferenceMs: median(rows.map(row => row.inferenceMs)),
          runtimeCalls: rows.map(row => row.runtimeCalls)});
        return {cold: {elapsedMs: cold.elapsedMs, inferenceMs: cold.inferenceMs, runtimeCalls: cold.runtimeCalls},
          unchanged: summarize(unchanged), edited: summarize(edited),
          firstShapesMedianMs: median(firstShapes), warmShapesMedianMs: median(warmShapes), firstShapes, warmShapes,
          regressionMatches: rows.length, regression: rows, adapter, logits,
          workload: [cold, ...unchanged, ...edited].map(({corrected, suggestions}) => ({corrected, suggestions}))};
      }, {origin, profile, preferWebGPU, cases: regression.cases});
      assert.deepEqual(external, [], 'Benchmark must remain local');
      assert.deepEqual(errors, [], 'Browser errors fail the benchmark');
      Object.assign(evidence.profiles[profile], results);
      console.log(JSON.stringify({profile, ...results, workload: undefined, regression: undefined, logits: undefined}, null, 2));
      await context.close();
    }
  }, () => createServer((request, response) => {
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    const bytes = assets.get(pathname);
    if (!bytes) { response.writeHead(404).end(); return; }
    response.writeHead(200, {'Content-Type': pathname.endsWith('.mjs') ? 'text/javascript' : pathname.endsWith('.json') ? 'application/json' : pathname === '/' ? 'text/html' : 'application/octet-stream'});
    response.end(bytes);
  }));
  if (evidence.profiles.baseline) {
    const {baseline, optimized} = evidence.profiles;
    let maximumLogitDifference = 0;
    for (let i = 0; i < baseline.logits.length; i++) {
      const before = baseline.logits[i];
      const after = optimized.logits[i];
      assert.deepEqual(before.dims, after.dims, 'Logit dimensions must exclude padding');
      assert.equal(before.values.length, after.values.length);
      const labels = before.dims[2];
      for (let start = 0; start < before.values.length; start += labels) {
        let bestBefore = 0;
        let bestAfter = 0;
        for (let j = 0; j < labels; j++) {
          maximumLogitDifference = Math.max(maximumLogitDifference, Math.abs(before.values[start + j] - after.values[start + j]));
          if (before.values[start + j] > before.values[start + bestBefore]) bestBefore = j;
          if (after.values[start + j] > after.values[start + bestAfter]) bestAfter = j;
        }
        assert.equal(bestBefore, bestAfter, 'Padding must preserve edit argmax');
      }
    }
    assert(maximumLogitDifference < 0.0005, 'Padding must preserve FP32 logits');
    evidence.maximumLogitDifference = maximumLogitDifference;
    for (const field of ['regression', 'workload']) {
      assert.equal(baseline[field].length, optimized[field].length);
      for (let i = 0; i < baseline[field].length; i++) {
        const before = baseline[field][i];
        const after = optimized[field][i];
        assert.equal(before.corrected, after.corrected, `${field} correction parity`);
        assert.equal(before.suggestions.length, after.suggestions.length, `${field} suggestion count parity`);
        for (let j = 0; j < before.suggestions.length; j++) {
          const {confidence: beforeConfidence, ...beforeEdit} = before.suggestions[j];
          const {confidence: afterConfidence, ...afterEdit} = after.suggestions[j];
          assert.deepEqual(beforeEdit, afterEdit, `${field} edit parity`);
          assert(Math.abs(beforeConfidence - afterConfidence) < 0.0001, `${field} confidence parity`);
        }
      }
    }
    evidence.outputParity = true;
  }
  await mkdir(artifactDir, {recursive: true});
  await writeFile(join(artifactDir, `inference-benchmark-${evidence.requestedBackend}.json`), JSON.stringify(evidence, null, 2) + '\n');
} finally {
  if (browser) await browser.close();
  await rm(temporary, {recursive: true, force: true});
}
