import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdtemp, readFile, readdir, rm, writeFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { chromium } from '@playwright/test';
import { browserArguments, findChromium, withLocalHttp } from './browser-environment.mjs';
import { validateCorpus, summarizeRows } from './challenge-metrics.mjs';
import { runtimeManifest } from './inference-runtime.mjs';
import { buildPaths } from './paths.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const {modelDirectory, artifactDir} = buildPaths(root);
const corpusPath = resolve(root, process.env.GAMMA_CHALLENGE_FILE ?? 'data/generated/harder-v1/challenge.json');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
let temporary;
let browser;
const network = [];
const pageErrors = [];
const evidence = {schema: 1, evaluated_at: new Date().toISOString(), runtime: runtimeManifest,
  evaluation_backend: 'wasm', infrastructure_passed: false, network, page_errors: pageErrors};
const resultPath = join(artifactDir, 'synthetic-challenge.json');

try {
  const corpusBytes = await readFile(corpusPath);
  const corpus = JSON.parse(corpusBytes);
  validateCorpus(corpus);
  const manifestBytes = await readFile(join(modelDirectory, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  const labels = JSON.parse(await readFile(join(modelDirectory, 'labels.json'), 'utf8'));
  const assets = new Map();
  for (const filename of ['manifest.json', 'labels.json', 'vocab.txt', 'model.onnx']) {
    const bytes = await readFile(join(modelDirectory, filename));
    if (filename !== 'manifest.json') assert.equal(hash(bytes), manifest.files[filename].sha256, `Model asset hash mismatch: ${filename}`);
    assets.set('/models/' + filename, bytes);
  }
  const engineHash = createHash('sha256');
  for (const name of (await readdir(join(root, 'packages/engine/src'))).sort()) {
    engineHash.update(name + '\0');
    engineHash.update(await readFile(join(root, 'packages/engine/src', name)));
  }
  Object.assign(evidence, {scope: corpus.scope, corpus_sha256: hash(corpusBytes),
    engine_source_sha256: engineHash.digest('hex'), model_name: manifest.name,
    model_sha256: manifest.files['model.onnx'].sha256, model_manifest_sha256: hash(manifestBytes),
    confidence_threshold: manifest.confidenceThreshold});
  temporary = await mkdtemp(join(tmpdir(), 'gamma-synthetic-'));
  await build({stdin: {contents: `export * from './packages/engine/src/index.ts'; export {WordPieceTokenizer} from './packages/engine/src/tokenizer.ts';`,
    resolveDir: root, sourcefile: 'challenge-engine.ts'}, outfile: join(temporary, 'engine.mjs'),
    bundle: true, format: 'esm', platform: 'browser', target: 'chrome116'});
  assets.set('/engine.mjs', await readFile(join(temporary, 'engine.mjs')));
  assets.set('/', Buffer.from('<!doctype html><html lang="en"><title>Synthetic challenge</title><body>Local evaluation</body></html>'));
  await withLocalHttp(async origin => {
    browser = await chromium.launch({executablePath: await findChromium(), headless: true, args: browserArguments()});
    evidence.chromium_version = browser.version();
    const context = await browser.newContext();
    await context.route('**/*', route => {
      if (!route.request().url().startsWith(origin)) {
        network.push(route.request().url());
        return route.abort();
      }
      return route.continue();
    });
    const page = await context.newPage();
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.goto(origin);
    const rows = {rules: [], full_engine: []};
    for (let start = 0; start < corpus.cases.length; start += 16) {
      const batch = corpus.cases.slice(start, start + 16);
      const checked = await page.evaluate(async ({origin, cases}) => {
        const engine = await import(origin + 'engine.mjs');
        const vocabulary = await (await fetch(origin + 'models/vocab.txt')).text();
        const manifest = await (await fetch(origin + 'models/manifest.json')).json();
        const tokenizer = new engine.WordPieceTokenizer(vocabulary);
        const outputs = [];
        for (const item of cases) {
          const rules = engine.analyzeRules(item.source);
          const full = await engine.analyzeText(item.source, {modelBaseUrl: origin + 'models/', preferWebGPU: false});
          if (full.backend !== 'wasm' || full.modelError) throw new Error(`Model inference failed for ${item.id}: ${full.modelError ?? full.backend}`);
          // applySuggestions validates UTF-16 spans, original text, and overlap.
          outputs.push({id: item.id, chunks: tokenizer.chunks(item.source, manifest.maxSequenceLength).length,
            rules: {actual: engine.applySuggestions(item.source, rules), suggestions: rules},
            full_engine: {actual: engine.applySuggestions(item.source, full.suggestions), suggestions: full.suggestions,
              backend: full.backend, elapsedMs: full.elapsedMs}});
        }
        return outputs;
      }, {origin, cases: batch});
      assert.equal(checked.length, batch.length);
      for (let index = 0; index < checked.length; index++) {
        assert.equal(checked[index].id, batch[index].id);
        for (const mode of ['rules', 'full_engine']) rows[mode].push({...batch[index], ...checked[index][mode], chunks: checked[index].chunks});
      }
      console.log(`Checked ${Math.min(start + 16, corpus.cases.length)}/${corpus.cases.length} synthetic cases.`);
    }
    assert.equal(network.length, 0, 'Inference requested a nonlocal resource');
    assert.equal(pageErrors.length, 0, 'Browser execution error');
    assert(rows.full_engine.filter(row => row.family === 'window-boundaries').every(row => row.chunks > 1),
      'Every window-boundary case must cross actual model windows');
    const errors = corpus.cases.filter(row => !row.clean);
    evidence.model_label_coverage = {erroneous_cases: errors.length,
      representable: errors.filter(row => row.requiredTags.every(tag => labels.includes(tag))).length,
      unsupported_cases_included: errors.filter(row => row.requiredTags.some(tag => !labels.includes(tag))).length};
    evidence.scores = Object.fromEntries(Object.entries(rows).map(([mode, results]) => [mode, summarizeRows(results)]));
    evidence.rows = rows;
    evidence.infrastructure_passed = true;
  }, () => createServer((request, response) => {
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    const bytes = assets.get(pathname);
    if (!bytes) { response.writeHead(404).end(); return; }
    const contentType = pathname.endsWith('.mjs') ? 'text/javascript' : pathname.endsWith('.json') ? 'application/json' : pathname === '/' ? 'text/html' : 'application/octet-stream';
    response.writeHead(200, {'Content-Type': contentType});
    response.end(bytes);
  }));
  console.log(JSON.stringify({infrastructure_passed: true,
    scores: Object.fromEntries(Object.entries(evidence.scores).map(([mode, {by_family: _families, ...score}]) => [mode, score])),
    model_label_coverage: evidence.model_label_coverage}, null, 2));
} catch (error) {
  evidence.failure = error instanceof Error ? error.message : String(error);
  throw error;
} finally {
  try {
    await mkdir(artifactDir, {recursive: true});
    await writeFile(resultPath, JSON.stringify(evidence, null, 2) + '\n');
  } finally {
    try { if (browser) await browser.close(); }
    finally { if (temporary) await rm(temporary, {recursive: true, force: true}); }
  }
}
