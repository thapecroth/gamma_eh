import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {build} from 'esbuild';
import {chromium} from '@playwright/test';
import {browserArguments, findChromium} from './browser-environment.mjs';

const {values} = parseArgs({options: {'model-dir': {type: 'string'}, output: {type: 'string'}}});
if (!values['model-dir'] || !values.output) throw new Error('MODEL_AND_OUTPUT_REQUIRED');
const model = resolve(values['model-dir']), output = resolve(values.output);
if (!output.startsWith(resolve('artifacts') + '/')) throw new Error('OUTPUT_MUST_BE_IGNORED');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = JSON.parse(await readFile(join(model, 'manifest.json'), 'utf8'));
const inputBytes = await readFile(join(model, 'inputs.json'));
if (digest(inputBytes) !== manifest.inputSha256) throw new Error('INPUT_HASH_MISMATCH');
const inputs = JSON.parse(inputBytes);
const directory = await mkdtemp(join(tmpdir(), 'gamma-generation-benchmark-'));
const files = new Map();
let browser, server;
const sockets = new Set();
try {
  for (const name of ['encoder_quantized.onnx', 'decoder_quantized.onnx']) {
    const path = join(model, name);
    if (digest(await readFile(path)) !== manifest.files[name].sha256) throw new Error('MODEL_HASH_MISMATCH');
    files.set('/' + name, path);
  }
  for (const name of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
    files.set('/runtime/' + name, resolve('node_modules/onnxruntime-web/dist', name));
  }
  const bundle = join(directory, 'runner.mjs');
  await build({stdin: {contents: 'import * as ort from "onnxruntime-web/wasm"; globalThis.ort = ort;',
    resolveDir: process.cwd()}, outfile: bundle, bundle: true, format: 'esm', platform: 'browser', logLevel: 'silent'});
  files.set('/runner.mjs', bundle);
  server = createServer((request, response) => {
    if (request.url === '/') {response.setHeader('Content-Type', 'text/html'); response.end('<script type="module" src="/runner.mjs"></script>'); return;}
    const file = files.get(request.url);
    if (!file) {response.writeHead(404).end(); return;}
    response.setHeader('Content-Type', file.endsWith('.mjs') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
    createReadStream(file).on('error', () => response.destroy()).pipe(response);
  });
  server.on('connection', socket => {sockets.add(socket); socket.once('close', () => sockets.delete(socket));});
  await new Promise((ready, reject) => {server.once('error', reject); server.listen(0, '127.0.0.1', ready);});
  const origin = 'http://127.0.0.1:' + server.address().port;
  browser = await chromium.launch({executablePath: await findChromium(), headless: true, args: browserArguments()});
  const context = await browser.newContext({serviceWorkers: 'block'});
  let blockedRequests = 0;
  await context.route('**/*', async route => {
    if (new URL(route.request().url()).origin === origin) await route.continue();
    else {blockedRequests++; await route.abort();}
  });
  const page = await context.newPage(); await page.goto(origin); await page.waitForFunction(() => Boolean(globalThis.ort));
  const result = await page.evaluate(async ({origin, inputs}) => {
    const ort = globalThis.ort;
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.wasmPaths = origin + '/runtime/';
    const before = performance.now();
    const encoder = await ort.InferenceSession.create(origin + '/encoder_quantized.onnx', {executionProviders: ['wasm']});
    const decoder = await ort.InferenceSession.create(origin + '/decoder_quantized.onnx', {executionProviders: ['wasm']});
    const coldInitializationMs = performance.now() - before;
    const rows = [];
    for (const input of inputs) {
      if (!input.inputIds) {rows.push({index: input.index, skipped: true, elapsedMs: 0}); continue;}
      const started = performance.now();
      const ids = new ort.Tensor('int64', BigInt64Array.from(input.inputIds, BigInt), [1, input.inputIds.length]);
      const mask = new ort.Tensor('int64', new BigInt64Array(input.inputIds.length).fill(1n), [1, input.inputIds.length]);
      const encoded = await encoder.run({input_ids: ids, attention_mask: mask});
      const generated = [0];
      for (let step = 0; step < 256; step++) {
        const previous = new ort.Tensor('int64', BigInt64Array.from(generated, BigInt), [1, generated.length]);
        const logits = (await decoder.run({input_ids: previous, encoder_hidden_states: encoded.hidden_states, attention_mask: mask})).logits;
        const vocabulary = logits.dims[2], start = (generated.length - 1) * vocabulary;
        let token = 0;
        for (let index = 1; index < vocabulary; index++) if (logits.data[start + index] > logits.data[start + token]) token = index;
        generated.push(token); previous.dispose(); logits.dispose();
        if (token === 1) break;
      }
      rows.push({index: input.index, skipped: false, elapsedMs: performance.now() - started, ids: generated,
        completed: generated.at(-1) === 1, nativeInt8TokenParity: JSON.stringify(generated) === JSON.stringify(input.expectedIds)});
      ids.dispose(); mask.dispose(); encoded.hidden_states.dispose();
    }
    await encoder.release(); await decoder.release();
    return {coldInitializationMs, rows};
  }, {origin, inputs});
  const measured = result.rows.filter(row => !row.skipped), times = measured.map(row => row.elapsedMs).sort((a, b) => a - b);
  const report = {...result, manifest, browser: browser.version(), backend: 'wasm', numThreads: 1, blockedRequests,
    totalRows: inputs.length, triggeredRows: measured.length, completedRows: measured.filter(row => row.completed).length,
    tokenParityRows: measured.filter(row => row.nativeInt8TokenParity).length,
    kernelLatencyMs: {p50: times[Math.ceil(times.length * .5) - 1] ?? null, p95: times[Math.ceil(times.length * .95) - 1] ?? null},
    meanKernelMsPerInput: result.rows.reduce((sum, row) => sum + row.elapsedMs, 0) / inputs.length};
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({objective: manifest.objective, totalRows: report.totalRows, triggeredRows: report.triggeredRows,
    completedRows: report.completedRows, tokenParityRows: report.tokenParityRows, coldInitializationMs: result.coldInitializationMs,
    kernelLatencyMs: report.kernelLatencyMs, meanKernelMsPerInput: report.meanKernelMsPerInput}));
} finally {
  if (browser) await browser.close();
  if (server?.listening) await new Promise(done => {server.close(done); for (const socket of sockets) socket.destroy();});
  await rm(directory, {recursive: true, force: true});
}
