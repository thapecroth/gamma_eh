// Actual JAX/WASM proposals, with an auditable threshold sweep on fixed outputs.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createServer} from 'node:http';
import {mkdir, mkdtemp, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {build} from 'esbuild';
import {chromium} from '@playwright/test';
import {browserArguments, findChromium, withLocalHttp} from './browser-environment.mjs';
import {runtimeManifest} from './inference-runtime.mjs';

const {values} = parseArgs({options: {input: {type: 'string'}, model: {type: 'string'},
  output: {type: 'string'}, 'engine-root': {type: 'string'}, 'verify-all': {type: 'boolean', default: false},
  thresholds: {type: 'string', default: '0.6,0.8,0.9,0.95,0.98,0.995'}}});
if (!values.input || !values.model || !values.output) throw new Error('--input, --model, and --output are required');
const root = fileURLToPath(new URL('../', import.meta.url));
const engineRoot = resolve(values['engine-root'] ?? root);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
let thresholds = [];
const report = {schema: 1, passed: false, runtime: runtimeManifest, backend: 'wasm',
  evaluated_at: new Date().toISOString(), network: [], page_errors: [], thresholds,
  public_api_verification: values['verify-all'] ? 'Every sentence at grid endpoints' : 'First sentence of every batch at grid endpoints'};
let temporary, browser;
try {
  const parts = values.thresholds.split(',');
  thresholds = parts.map(Number);
  assert(parts.every(part => part.trim()) && new Set(thresholds).size === thresholds.length &&
    thresholds.every(t => Number.isFinite(t) && t >= 0 && t <= 1), 'Invalid confidence threshold grid');
  report.thresholds = thresholds;
  const input = await readFile(resolve(values.input));
  const corpus = values.input.endsWith('.jsonl') ? input.toString().trim().split('\n').map(line => JSON.parse(line)) : JSON.parse(input).cases;
  assert(Array.isArray(corpus) && corpus.length > 0, 'Empty evaluation population');
  assert(new Set(corpus.map(item => item.id)).size === corpus.length, 'Duplicate evaluation IDs');
  assert(corpus.every(item => typeof item.id === 'string' && typeof item.source === 'string' && item.source.trim() && item.source.length <= 20_000), 'Invalid evaluation row');
  report.input_sha256 = hash(input);
  report.sentences = corpus.length;
  const model = resolve(values.model);
  const manifestBytes = await readFile(join(model, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  report.model = {name: manifest.name, manifest_sha256: hash(manifestBytes),
    model_sha256: manifest.files['model.onnx'].sha256, policy: manifest};
  // Explicit diagnostic override; never changes files or claims release qualification.
  const servedManifest = {...manifest, disableModelEdits: false};
  report.diagnostic_policy_override = {disableModelEdits: false, confidenceThreshold: 0};
  const assets = new Map([['/models/manifest.json', Buffer.from(JSON.stringify(servedManifest))]]);
  for (const name of ['labels.json', 'vocab.txt', 'model.onnx']) {
    const bytes = await readFile(join(model, name));
    assert.equal(hash(bytes), manifest.files[name].sha256, `Asset hash mismatch: ${name}`);
    assets.set('/models/' + name, bytes);
  }
  const engineHash = createHash('sha256');
  for (const name of (await readdir(join(engineRoot, 'packages/engine/src'))).sort()) {
    engineHash.update(name + '\0'); engineHash.update(await readFile(join(engineRoot, 'packages/engine/src', name)));
  }
  report.engine_source_sha256 = engineHash.digest('hex');
  temporary = await mkdtemp(join(tmpdir(), 'gamma-tuning-'));
  await build({stdin: {contents: "export * from './packages/engine/src/index.ts'; export {analyzeModel} from './packages/engine/src/model.ts'; export {overlaps} from './packages/engine/src/edits.ts';",
    resolveDir: engineRoot, sourcefile: 'tuning-engine.ts'}, outfile: join(temporary, 'engine.mjs'), bundle: true,
    format: 'esm', platform: 'browser', target: 'chrome116'});
  assets.set('/engine.mjs', await readFile(join(temporary, 'engine.mjs')));
  assets.set('/', Buffer.from('<!doctype html><html lang="en"><title>Local model evaluation</title></html>'));
  report.rows = [];
  await withLocalHttp(async origin => {
    browser = await chromium.launch({executablePath: await findChromium(), headless: true, args: browserArguments()});
    report.chromium_version = browser.version();
    const context = await browser.newContext();
    await context.route('**/*', route => {
      if (!route.request().url().startsWith(origin)) { report.network.push(route.request().url()); return route.abort(); }
      return route.continue();
    });
    const page = await context.newPage();
    page.on('pageerror', error => report.page_errors.push(error.message));
    await page.goto(origin);
    for (let start = 0; start < corpus.length; start += 16) {
      const batch = corpus.slice(start, start + 16);
      const outputs = await page.evaluate(async ({origin, batch, thresholds, verifyAll}) => {
        const engine = await import(origin + 'engine.mjs');
        const options = {modelBaseUrl: origin + 'models/', preferWebGPU: false, confidenceThreshold: 0};
        const results = [];
        for (const item of batch) {
          const rules = engine.analyzeRules(item.source);
          const model = await engine.analyzeModel(item.source, options);
          const combined = rules.some(edit => !edit.replacement) ? await engine.analyzeModel(item.source, options, rules) : model;
          if (model.backend !== 'wasm' || combined.backend !== 'wasm') throw new Error('Required WASM backend unavailable');
          const predictions = thresholds.map(threshold => {
            const neural = model.suggestions.filter(edit => edit.confidence >= threshold);
            const full = [...rules];
            for (const edit of combined.suggestions.filter(edit => edit.confidence >= threshold)) {
              if (!full.some(previous => engine.overlaps(previous, edit))) full.push(edit);
            }
            full.sort((a, b) => a.start - b.start);
            return {threshold, model_only: engine.applySuggestions(item.source, neural),
              full_engine: engine.applySuggestions(item.source, full)};
          });
          // Verify threshold filtering and merge against the public API at both ends.
          if (verifyAll || results.length === 0) for (const threshold of new Set([thresholds[0], thresholds.at(-1)])) {
            const actual = await engine.analyzeText(item.source, {...options, confidenceThreshold: threshold});
            if (actual.backend !== 'wasm' || actual.modelError) throw new Error('Public API inference failed');
            if (engine.applySuggestions(item.source, actual.suggestions) !== predictions.find(p => p.threshold === threshold).full_engine) throw new Error('Threshold sweep differs from public API');
          }
          results.push({id: item.id, source: item.source, rules: engine.applySuggestions(item.source, rules),
            model_suggestions: model.suggestions, combined_suggestions: combined.suggestions, predictions});
        }
        return results;
      }, {origin, batch, thresholds, verifyAll: values['verify-all']});
      assert.equal(outputs.length, batch.length);
      outputs.forEach((item, index) => { assert.equal(item.id, batch[index].id); assert.equal(item.source, batch[index].source); });
      report.rows.push(...outputs);
      console.log(`${manifest.name}: ${Math.min(start + 16, corpus.length)}/${corpus.length}`);
    }
    assert.equal(report.rows.length, corpus.length);
    assert.equal(report.network.length, 0, 'Nonlocal inference request');
    assert.equal(report.page_errors.length, 0, 'Browser error');
    report.passed = true;
  }, () => createServer((request, response) => {
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    const bytes = assets.get(pathname);
    if (!bytes) { response.writeHead(404).end(); return; }
    response.writeHead(200, {'Content-Type': pathname.endsWith('.mjs') ? 'text/javascript' : pathname.endsWith('.json') ? 'application/json' : pathname === '/' ? 'text/html' : 'application/octet-stream'}).end(bytes);
  }));
} catch (error) {
  report.failure = error instanceof Error ? error.message : String(error); throw error;
} finally {
  try { await mkdir(dirname(resolve(values.output)), {recursive: true}); await writeFile(resolve(values.output), JSON.stringify(report) + '\n'); }
  finally { try { if (browser) await browser.close(); } finally { if (temporary) await rm(temporary, {recursive: true, force: true}); } }
}
