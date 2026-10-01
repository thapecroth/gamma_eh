import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {access, mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {tmpdir} from 'node:os';
import {extname, join, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {build} from 'esbuild';
import {chromium} from '@playwright/test';
import {runtimeManifest} from './inference-runtime.mjs';
import {browserArguments, findChromium} from './browser-environment.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const modelFiles = ['manifest.json', 'labels.json', 'vocab.txt', 'model.onnx', 'model_quantized.onnx'];


function argumentsForRun() {
  const {values} = parseArgs({options: {
    'model-dir': {type: 'string', default: join(root, 'models/browser')},
    input: {type: 'string'}, output: {type: 'string'},
    modes: {type: 'string', default: 'rules,model,combined'}, passes: {type: 'string', default: '1,2'},
    limit: {type: 'string', default: '0'}, 'timeout-ms': {type: 'string', default: '120000'},
    webgpu: {type: 'boolean', default: false}, help: {type: 'boolean', default: false},
  }});
  if (values.help) {
    console.log('node scripts/evaluate-browser.mjs --input EVALUATION.jsonl --output artifacts/REPORT.json [--model-dir DIR] [--modes rules,model,combined] [--passes 1,2] [--limit N] [--webgpu]');
    return null;
  }
  if (!values.input || !values.output) throw new Error('INPUT_OUTPUT_REQUIRED');
  const modes = [...new Set(values.modes.split(','))];
  const passes = [...new Set(values.passes.split(',').map(Number))];
  const limit = Number(values.limit), timeoutMs = Number(values['timeout-ms']);
  if (!modes.length || modes.some(mode => !['rules', 'model', 'combined'].includes(mode)) ||
      !passes.length || passes.some(pass => !Number.isInteger(pass) || pass < 1 || pass > 3) ||
      !Number.isInteger(limit) || limit < 0 || !Number.isInteger(timeoutMs) || timeoutMs < 1000) {
    throw new Error('INVALID_EVALUATION_OPTIONS');
  }
  const output = resolve(values.output);
  if (!output.startsWith(join(root, 'artifacts') + sep)) throw new Error('OUTPUT_MUST_BE_IN_IGNORED_ARTIFACTS');
  return {input: resolve(values.input), output, modelDir: resolve(values['model-dir']), modes, passes,
    limit, timeoutMs, webgpu: values.webgpu};
}

async function inputs(path, limit) {
  const bytes = await readFile(path);
  const lines = bytes.toString('utf8').split(/\r?\n/u);
  if (lines.at(-1) === '') lines.pop();
  const rows = lines.map((line, index) => {
    let value;
    try { value = JSON.parse(line); } catch { throw new Error('CORRUPT_EVALUATION_JSONL'); }
    const references = value?.references ?? (typeof value?.target === 'string' ? [value.target] : undefined);
    if (!value || typeof value.source !== 'string' || !Array.isArray(references) || !references.length ||
        references.some(reference => typeof reference !== 'string' || !reference.trim()) ||
        (value.id !== undefined && typeof value.id !== 'string')) throw new Error('INVALID_EVALUATION_ROW');
    return {...value, id: value.id ?? `sample-${index}`, index, references,
      inputSha256: digest(Buffer.from(value.source, 'utf8'))};
  });
  if (!rows.length || new Set(rows.map(row => row.id)).size !== rows.length) throw new Error('EMPTY_OR_DUPLICATE_EVALUATION_IDS');
  return {rows: limit ? rows.slice(0, limit) : rows,
    evidence: {sha256: digest(bytes), totalRows: rows.length, selectedRows: limit ? Math.min(limit, rows.length) : rows.length,
      selection: limit ? 'First rows for smoke evaluation; incomplete population.' : 'Complete input population; no reference, vocabulary, or inference filtering.'}};
}

async function assets(modelDir) {
  const manifestBytes = await readFile(join(modelDir, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const files = new Map();
  const weights = {};
  for (const filename of modelFiles) {
    const path = join(modelDir, filename), bytes = await readFile(path);
    const sha256 = digest(bytes);
    if (filename !== 'manifest.json' && manifest.files?.[filename]?.sha256 !== sha256) {
      throw new Error('MODEL_ASSET_HASH_MISMATCH');
    }
    files.set('/models/' + filename, path);
    if (filename.endsWith('.onnx')) weights[filename] = {bytes: bytes.length, sha256};
  }
  const labels = JSON.parse(await readFile(join(modelDir, 'labels.json'), 'utf8'));
  if (!Array.isArray(labels) || labels[0] !== 'KEEP' || labels.some(label => typeof label !== 'string')) {
    throw new Error('INVALID_MODEL_LABELS');
  }
  return {files, evidence: {manifestSha256: digest(manifestBytes), weights, runtime: runtimeManifest,
    executedWeights: 'model.onnx',
    policy: {baseModel: manifest.base_model, baseRevision: manifest.base_revision,
      editSchema: manifest.editSchema ?? 1, maxSequenceLength: manifest.maxSequenceLength,
      confidenceThreshold: manifest.confidenceThreshold, confidenceThresholds: manifest.confidenceThresholds ?? {},
      disableModelEdits: manifest.disableModelEdits ?? false, maxPasses: manifest.maxPasses ?? 1,
      publicationAllowed: manifest.publication_allowed ?? true}, hashesVerified: true}};
}

function quantile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)];
}

function summarize(rows) {
  const durations = rows.map(row => row.elapsedMs).filter(value => typeof value === 'number' && Number.isFinite(value));
  const backends = {};
  for (const row of rows) backends[row.backend] = (backends[row.backend] ?? 0) + 1;
  return {sentences: rows.length, failures: rows.filter(row => row.failure).length,
    exactMatches: rows.filter(row => row.references.includes(row.corrected)).length,
    exactMatchRate: rows.filter(row => row.references.includes(row.corrected)).length / rows.length,
    modelRuns: rows.reduce((sum, row) => sum + row.modelRuns, 0), backends,
    latencyMs: {samples: durations.length, p50: quantile(durations, .5), p95: quantile(durations, .95)},
    latencyScope: 'Warm browser engine analysis, including all measured successes and failures; startup reported separately.'};
}

function failedRows(rows, code) {
  return rows.map(row => ({id: row.id, index: row.index, sourceSha256: row.inputSha256,
    source: row.source, references: row.references, corrected: row.source, suggestions: [],
    backend: 'unavailable', modelRuns: 0, elapsedMs: null, failure: {code}, failed: true, executed: false}));
}

async function run(options) {
  await mkdir(join(root, 'artifacts'), {recursive: true});
  try { await access(options.output); throw new Error('OUTPUT_ALREADY_EXISTS'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const input = await inputs(options.input, options.limit);
  const localAssets = await assets(options.modelDir);
  const temporary = await mkdtemp(join(tmpdir(), 'gamma-quality-browser-'));
  const evidence = {schema: 1, scope: 'Actual browser engine predictions; rules, neural, and combined modes scored separately.',
    input: input.evidence, inputSha256: input.evidence.sha256, model: localAssets.evidence,
    backendPreference: options.webgpu ? 'webgpu-with-wasm-fallback' : 'wasm',
    runs: [], network: {blockedRequests: 0}, protectedTextCanaries: [], passed: false};
  let browser, context, server, page;
  let runFailure;
  const sockets = new Set();
  const conditions = options.modes.flatMap(mode => options.passes.map(passes => ({mode, passes, rows: []})));
  try {
    const bundled = join(temporary, 'engine.mjs');
    await build({entryPoints: [join(root, 'packages/engine/src/index.ts')], outfile: bundled,
      bundle: true, format: 'esm', platform: 'browser', target: 'chrome116', logLevel: 'silent'});
    evidence.engineBundleSha256 = digest(await readFile(bundled));
    localAssets.files.set('/engine.mjs', bundled);
    const fixture = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Local engine evaluation</title><script type="module" src="/runner.mjs"></script></html>';
    server = createServer((request, response) => {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1');
      if (request.method !== 'GET' || url.search) { response.writeHead(403).end(); return; }
      response.setHeader('Cache-Control', 'no-store');
      response.setHeader('X-Content-Type-Options', 'nosniff');
      response.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self' blob:");
      if (url.pathname === '/') { response.setHeader('Content-Type', 'text/html; charset=utf-8'); response.end(fixture); return; }
      if (url.pathname === '/runner.mjs') {
        response.setHeader('Content-Type', 'text/javascript');
        response.end('import * as engine from "/engine.mjs"; globalThis.gammaEngine = engine;');
        return;
      }
      if (url.pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
      const path = localAssets.files.get(url.pathname);
      if (!path) { response.writeHead(404).end(); return; }
      response.setHeader('Content-Type', {'.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json',
        '.txt': 'text/plain; charset=utf-8'}[extname(path)] ?? 'application/octet-stream');
      createReadStream(path).on('error', () => response.destroy()).pipe(response);
    });
    server.on('connection', socket => { sockets.add(socket); socket.once('close', () => sockets.delete(socket)); });
    await new Promise((ready, rejectListen) => { server.once('error', rejectListen); server.listen(0, '127.0.0.1', ready); });
    const origin = `http://127.0.0.1:${server.address().port}`;
    const launchStarted = performance.now();
    browser = await chromium.launch({executablePath: await findChromium(), headless: true,
      args: [...browserArguments({...process.env, GAMMA_TEST_WEBGPU: options.webgpu ? '1' : '0'}),
        '--disable-background-networking', '--disable-component-update', '--disable-sync', '--no-default-browser-check']});
    evidence.browser = {version: browser.version(), launchMs: performance.now() - launchStarted};
    context = await browser.newContext({serviceWorkers: 'block'});
    await context.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url());
      const permitted = url.origin === origin && request.method() === 'GET' && !url.search &&
        (['/', '/runner.mjs', '/favicon.ico'].includes(url.pathname) || localAssets.files.has(url.pathname));
      if (permitted || url.protocol === 'blob:' || url.protocol === 'data:') await route.continue();
      else { evidence.network.blockedRequests++; await route.abort('blockedbyclient'); }
    });
    page = await context.newPage();
    page.setDefaultTimeout(options.timeoutMs);
    await page.goto(origin, {waitUntil: 'load'});
    await page.waitForFunction(() => Boolean(globalThis.gammaEngine));
    const engineOptions = {modelBaseUrl: origin + '/models/', preferWebGPU: options.webgpu};
    const startup = performance.now();
    evidence.browser.initialization = await page.evaluate(async settings => {
      const result = await globalThis.gammaEngine.analyzeText('The students has a notebook.', {...settings, mode: 'model', maxPasses: 1});
      return {backend: result.backend, modelRuns: result.modelRuns ?? 0, failed: Boolean(result.modelError), elapsedMs: result.elapsedMs};
    }, engineOptions);
    evidence.browser.initialization.totalMs = performance.now() - startup;
    // Safety canaries are artificial local fixtures, outside benchmark scoring.
    for (const condition of conditions) {
      const canaries = await page.evaluate(async ({settings, mode, passes}) => {
        const texts = ['`She have a freind.`', '```She have a freind.```', 'https://example.test/recieve', 'freind@example.test'];
        const rows = [];
        for (const text of texts) {
          const result = await globalThis.gammaEngine.analyzeText(text, {...settings, mode, maxPasses: passes});
          const corrected = globalThis.gammaEngine.applySuggestions(text, result.suggestions);
          rows.push({unchanged: corrected === text, suggestions: result.suggestions.length, modelRuns: result.modelRuns ?? 0,
            failed: Boolean(result.modelError)});
        }
        return rows;
      }, {settings: engineOptions, mode: condition.mode, passes: condition.passes});
      evidence.protectedTextCanaries.push({mode: condition.mode, passes: condition.passes, rows: canaries});
      for (let start = 0; start < input.rows.length; start += 20) {
        const batch = input.rows.slice(start, start + 20);
        const actual = await page.evaluate(async ({rows, settings, mode, passes, timeoutMs}) => {
          const results = [];
          for (const row of rows) {
            let timer;
            try {
              const result = await Promise.race([
                globalThis.gammaEngine.analyzeText(row.source, {...settings, mode, maxPasses: passes}),
                new Promise((_, rejectTimeout) => { timer = setTimeout(() => rejectTimeout(new Error('EVALUATION_TIMEOUT')), timeoutMs); }),
              ]);
              const corrected = globalThis.gammaEngine.applySuggestions(row.source, result.suggestions);
              results.push({id: row.id, index: row.index, inputSha256: row.inputSha256,
                source: row.source, corrected, suggestions: result.suggestions,
                backend: result.backend, modelRuns: result.modelRuns ?? 0, elapsedMs: result.elapsedMs,
                executed: true, failure: result.modelError ? {code: 'MODEL_INFERENCE_FAILED'} : null});
            } catch (error) {
              const timedOut = error instanceof Error && error.message === 'EVALUATION_TIMEOUT';
              results.push({id: row.id, index: row.index, inputSha256: row.inputSha256,
                source: row.source, corrected: row.source, suggestions: [],
                backend: 'unavailable', modelRuns: 0, elapsedMs: null, executed: true,
                failure: {code: timedOut ? 'EVALUATION_TIMEOUT' : 'ANALYSIS_OR_APPLY_FAILED'}});
              if (timedOut) break;
            } finally { clearTimeout(timer); }
          }
          return results;
        }, {rows: batch.map(({id, index, inputSha256, source}) => ({id, index, inputSha256, source})),
          settings: engineOptions, mode: condition.mode, passes: condition.passes, timeoutMs: options.timeoutMs});
        condition.rows.push(...actual.map((row, index) => {
          const {inputSha256, ...result} = row;
          return {...result, sourceSha256: inputSha256, references: batch[index].references, failed: Boolean(row.failure)};
        }));
        if (actual.some(row => row.failure?.code === 'EVALUATION_TIMEOUT')) throw new Error('EVALUATION_TIMEOUT');
        if (condition.rows.length % 100 === 0 || condition.rows.length === input.rows.length) {
          console.log(JSON.stringify({mode: condition.mode, passes: condition.passes, completed: condition.rows.length, total: input.rows.length}));
        }
      }
    }
  } catch {
    runFailure = 'BROWSER_EVALUATION_INTERRUPTED';
  } finally {
    if (runFailure) for (const condition of conditions) {
      condition.rows.push(...failedRows(input.rows.slice(condition.rows.length), runFailure));
    }
    for (const condition of conditions) condition.summary = summarize(condition.rows);
    evidence.runs = conditions.map(({mode, passes, rows, summary}) => ({mode, maxPasses: passes, predictions: rows, summary}));
    evidence.passed = !runFailure && evidence.network.blockedRequests === 0 &&
      conditions.every(condition => condition.summary.failures === 0) &&
      evidence.protectedTextCanaries.every(condition => condition.rows.every(row => row.unchanged && !row.failed));
    if (runFailure) evidence.failure = {code: runFailure};
    try { if (context) await context.close(); }
    finally {
      try { if (browser) await browser.close(); }
      finally {
        if (server?.listening) await new Promise(closed => {
          server.close(closed);
          for (const socket of sockets) socket.destroy();
        });
        await rm(temporary, {recursive: true, force: true});
        await mkdir(resolve(options.output, '..'), {recursive: true});
        await writeFile(options.output, JSON.stringify(evidence, null, 2) + '\n', {flag: 'wx', mode: 0o600});
      }
    }
  }
  console.log(JSON.stringify({passed: evidence.passed, conditions: conditions.map(({mode, passes, summary}) => ({mode, passes, ...summary}))}));
  if (!evidence.passed) process.exitCode = 1;
}

try {
  const options = argumentsForRun();
  if (options) await run(options);
} catch {
  // Never echo raw input, URLs, or exception bodies to the terminal.
  console.error('Browser evaluation failed. Check arguments, local assets, or the private artifact when one was created.');
  process.exitCode = 1;
}
