import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { access, cp, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { chromium } from '@playwright/test';
import { buildPaths } from './paths.mjs';
import { runtimeManifest } from './inference-runtime.mjs';
import { browserArguments, findChromium } from './browser-environment.mjs';
import { extensionElement } from './extension-shadow.mjs';
import { webGPUAdapterEsbuildPlugin } from './webgpu-adapter-options.mjs';
import { verifyWindowsWebGPU } from './verify-windows-webgpu.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const temporary = await mkdtemp(join(tmpdir(), 'gamma-browser-check-'));
const {artifactDir, webOutput, extensionOutput} = buildPaths(root);
await mkdir(artifactDir, {recursive: true});
const failures = [];
const evidence = {web: {}, model: {}, extension: {}, network: []};
let context;
let server;

async function shadowElement(page, selector) {
  return extensionElement(context, page, selector);
}

async function waitPanel(page, pattern, timeout = 90_000, selector = '.checker-status') {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const element = await shadowElement(page, selector);
    if (element) {
      const {result} = await element.session.send('Runtime.callFunctionOn', {objectId: element.objectId,
        functionDeclaration: 'function() { return this.textContent; }', returnByValue: true});
      await element.session.detach();
      if (pattern.test(result.value ?? '')) return result.value;
    }
    await new Promise(resolvePoll => setTimeout(resolvePoll, 200));
  }
  throw new Error(`Extension panel did not reach ${pattern}`);
}

async function openSuggestions(page) {
  const badge = await shadowElement(page, 'button.badge');
  assert(badge, 'Expected an extension suggestion badge');
  await badge.session.send('Runtime.callFunctionOn', {objectId: badge.objectId, functionDeclaration: 'function() { this.click(); }'});
  await badge.session.detach();
}

async function acceptFirst(page) {
  let element = await shadowElement(page, 'button.accept');
  if (!element) {
    // Inline suggestions open their card from the badge; the earlier panel
    // already exposes its Accept button. Exercise the same user action.
    await openSuggestions(page);
    element = await shadowElement(page, 'button.accept');
  }
  assert(element, 'Expected an extension Accept button');
  await element.session.send('Runtime.callFunctionOn', {objectId: element.objectId, functionDeclaration: 'function() { this.click(); }'});
  await element.session.detach();
}

try {
  await build({entryPoints: [join(root, 'packages/engine/src/index.ts')], outfile: join(temporary, 'test-engine.mjs'),
    plugins: [webGPUAdapterEsbuildPlugin()], bundle: true, format: 'esm', platform: 'browser', target: 'chrome116'});
  await build({entryPoints: [join(root, 'tests/browser-model-probe.ts')], outfile: join(temporary, 'test-model-probe.mjs'),
    plugins: [webGPUAdapterEsbuildPlugin()], bundle: true, format: 'esm', platform: 'browser', target: 'chrome116'});
  const webDir = webOutput;
  for (const directory of [webOutput, extensionOutput]) {
    assert.deepEqual(JSON.parse(await readFile(join(directory, 'inference-runtime.json'), 'utf8')), runtimeManifest);
  }
  evidence.model.runtime = runtimeManifest;
  const modelManifest = JSON.parse(await readFile(join(webDir, 'models/manifest.json'), 'utf8'));
  for (const filename of ['model.onnx', 'model_quantized.onnx']) {
    for (const directory of [webDir, extensionOutput]) {
      const bytes = await readFile(join(directory, 'models', filename));
      assert.equal(createHash('sha256').update(bytes).digest('hex'), modelManifest.files[filename].sha256,
        'Built apps must contain the exact exported weights');
    }
  }
  evidence.model.builtWeightHashesVerified = true;
  const types = {'.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm'};
  server = createServer(async (request, response) => {
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    if (pathname === '/fixture.html') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><html lang="en"><title>Extension fixture</title><body><label>Draft<textarea id="draft">She have a freind.</textarea></label><label>Private<textarea id="private" data-private>She have a freind.</textarea></label><label>Payment<textarea id="payment" autocomplete="cc-number">She have a freind.</textarea></label><label>Disabled<textarea id="optout" spellcheck="false">She have a freind.</textarea></label><div id="plain" contenteditable="true" aria-label="Plain editor">She have a book.</div><div id="rich" contenteditable="true" aria-label="Rich editor"><strong>She have a book.</strong></div></body></html>');
      return;
    }
    const testModule = ['/test-engine.mjs', '/test-model-probe.mjs'].includes(pathname);
    const filename = testModule ? join(temporary, pathname.slice(1)) : resolve(webDir, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!testModule && !filename.startsWith(webDir + sep)) { response.writeHead(403).end(); return; }
    try {
      await access(filename);
      response.setHeader('Content-Type', types[extname(filename)] ?? 'application/octet-stream');
      createReadStream(filename).pipe(response);
    } catch { response.writeHead(404).end(); }
  });
  await new Promise((resolveReady, rejectListen) => {
    server.once('error', rejectListen);
    server.listen(0, '127.0.0.1', resolveReady);
  });
  const address = server.address();
  const origin = `http://127.0.0.1:${address.port}`;

  // Test-only copy gets localhost permission to exercise the real extension
  // pipeline without automating Chrome's native permission dialog. The shipping
  // manifest is checked below and is never altered.
  const extensionDir = join(temporary, 'extension');
  await cp(extensionOutput, extensionDir, {recursive: true});
  const shippingManifest = JSON.parse(await readFile(join(extensionDir, 'manifest.json'), 'utf8'));
  assert.equal(shippingManifest.host_permissions, undefined);
  assert.equal(shippingManifest.content_scripts, undefined);
  await writeFile(join(extensionDir, 'manifest.json'), JSON.stringify({...shippingManifest, host_permissions: ['http://127.0.0.1/*']}));
  context = await chromium.launchPersistentContext(join(temporary, 'profile'), {
    channel: 'chromium', executablePath: await findChromium(), headless: true,
    args: [`--disable-extensions-except=${extensionDir}`, `--load-extension=${extensionDir}`, ...browserArguments()],
    viewport: {width: 1440, height: 1050},
  });
  context.on('request', request => {
    const url = request.url();
    if (!url.startsWith(origin) && !url.startsWith('chrome-extension://') && !url.startsWith('data:')) evidence.network.push(url);
  });
  const page = await context.newPage();
  page.on('pageerror', error => failures.push(error.message));
  await page.goto(origin);
  evidence.model.manifest = await page.evaluate(async () => {
    const response = await fetch('/models/manifest.json');
    if (!response.ok) throw new Error('Model manifest unavailable');
    return response.json();
  });
  await page.getByRole('checkbox').focus();
  await page.getByRole('checkbox').press('Space');
  await page.getByRole('status').filter({hasText: /Local AI|Local rules/u}).first().waitFor({timeout: 90_000});
  evidence.web.initialBackend = await page.locator('.backend-status').textContent();
  evidence.web.modelWarning = await page.locator('.model-warning').count() ? await page.locator('.model-warning').textContent() : null;
  assert.match(evidence.web.initialBackend, /Local AI/u, 'Editor must execute the model, not silently pass via rules');
  await page.getByLabel('Your writing', {exact: true}).fill('She have a freind.');
  await page.getByRole('button', {name: 'Accept all suggestions'}).waitFor({state: 'visible'});
  await page.waitForFunction(() => document.querySelector('.accept-all-button')?.disabled === false);
  await page.getByRole('button', {name: 'Accept all suggestions'}).click();
  assert.equal(await page.getByLabel('Your writing', {exact: true}).inputValue(), 'She has a friend.');
  evidence.web.acceptAll = true;
  await page.getByLabel('Your writing', {exact: true}).fill('😀. The students has a notebook.');
  await page.waitForFunction(() => document.querySelector('.accept-all-button')?.disabled === false);
  await page.getByRole('button', {name: 'Accept all suggestions'}).click();
  assert.equal(await page.getByLabel('Your writing', {exact: true}).inputValue(), '😀. The students have a notebook.');
  evidence.web.modelCorrectionWithUTF16 = true;
  await page.getByRole('checkbox').focus();
  await page.getByRole('checkbox').press('Space');
  assert.equal(await page.getByRole('checkbox').isChecked(), false);
  await page.waitForFunction(() => document.querySelector('.backend-status')?.textContent?.includes('Local rules'));
  await page.getByLabel('Your writing', {exact: true}).fill('A clean sentence.');
  await page.getByText('No suggestions from this checker.').waitFor();
  evidence.web.rulesToggle = true;
  await page.getByLabel('Your writing', {exact: true}).fill('halo');
  await page.getByText('Did you mean “hello” as a greeting? “Halo” is also a valid word.').waitFor();
  await page.getByRole('button', {name: 'Accept', exact: true}).click();
  assert.equal(await page.getByLabel('Your writing', {exact: true}).inputValue(), 'hello');
  await page.getByLabel('Your writing', {exact: true}).fill('A halo surrounds the moon.');
  await page.getByText('No suggestions from this checker.').waitFor();
  await page.getByLabel('Your writing', {exact: true}).fill('Speling matters in a sentnce.');
  await page.waitForFunction(() => document.querySelectorAll('.suggestion-card').length === 2 && document.querySelector('.accept-all-button')?.disabled === false);
  await page.getByRole('button', {name: 'Accept all suggestions'}).click();
  assert.equal(await page.getByLabel('Your writing', {exact: true}).inputValue(), 'Spelling matters in a sentence.');
  evidence.web.dictionarySpellingWithoutAI = true;
  const protectedSpelling = '``speling ` sentnce\nwrold`` speling\u2011like speling\u203Fvalue speling\u200CValue';
  await page.getByLabel('Your writing', {exact: true}).fill(protectedSpelling);
  await page.getByText('No suggestions from this checker.').waitFor();
  assert.equal(await page.locator('.suggestion-card').count(), 0);
  assert.equal(await page.getByLabel('Your writing', {exact: true}).inputValue(), protectedSpelling);
  evidence.web.protectedDictionaryTokens = true;
  await page.getByLabel('Your writing', {exact: true}).fill('A little clarity goes a long way.\n\nI recieved your message, and we has a lot of ideas. My freind is writting about the project.\n\nWrite freely. You choose what to change.');
  await page.waitForFunction(() => document.querySelector('.accept-all-button')?.disabled === false);
  await page.getByLabel('Your writing', {exact: true}).blur();
  await page.screenshot({path: join(artifactDir, 'editor-desktop.png'), fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  evidence.web.mobileOverflow = false;
  await page.screenshot({path: join(artifactDir, 'editor-mobile.png'), fullPage: true});

  const regression = JSON.parse(await readFile(join(root, 'data/regression.json'), 'utf8'));
  Object.assign(evidence.model, await page.evaluate(async ({origin, cases}) => {
    const engine = await import(origin + '/test-engine.mjs');
    const rows = [];
    for (const item of cases) {
      const result = await engine.analyzeText(item.source, {modelBaseUrl: origin + '/models/', preferWebGPU: false});
      rows.push({...item, actual: engine.applySuggestions(item.source, result.suggestions), backend: result.backend,
        elapsedMs: result.elapsedMs, modelError: result.modelError,
        edits: result.suggestions.map(({original, replacement, source}) => ({original, replacement, source}))});
    }
    const gpuRows = [];
    for (const item of cases) {
      const result = await engine.analyzeText(item.source, {modelBaseUrl: origin + '/models/', preferWebGPU: true});
      gpuRows.push({...item, actual: engine.applySuggestions(item.source, result.suggestions), backend: result.backend, modelError: result.modelError});
    }
    const gpu = gpuRows.find(row => row.source === 'The students has a notebook.');
    return {rows, gpuRows, preferredBackend: gpu.backend, preferredCorrection: gpu.actual,
      hasWebGPU: 'gpu' in navigator, modelError: gpu.modelError};
  }, {origin, cases: regression.cases}));
  assert(evidence.model.rows.every(row => row.backend === 'wasm' && !row.modelError), 'All regression checks must execute actual WASM model');
  evidence.model.exactMatches = evidence.model.rows.filter(row => row.actual === row.target).length;
  evidence.model.total = evidence.model.rows.length;
  assert.equal(evidence.model.preferredCorrection, 'The students have a notebook.');
  if (process.env.GAMMA_TEST_WEBGPU === '1') {
    assert(evidence.model.gpuRows.every(row => row.backend === 'webgpu' && !row.modelError), 'Every explicit GPU regression must execute WebGPU');
    evidence.model.adapterInfo = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;
      const info = adapter.info;
      return {vendor: info.vendor, architecture: info.architecture, device: info.device,
        description: info.description, isFallbackAdapter: info.isFallbackAdapter};
    });
  }
  assert(evidence.model.gpuRows.every(row => row.actual === row.target && !row.modelError), 'Every preferred-backend correction must match');
  const reference = JSON.parse(await readFile(join(root, 'tests/fixtures/onnx-reference.json'), 'utf8'));
  assert.equal(reference.modelSha256, modelManifest.files['model.onnx'].sha256);
  evidence.model.parity = await page.evaluate(async ({origin, reference, gpuRequired}) => {
    const {JaxSession} = await import(origin + '/test-model-probe.mjs');
    const bytes = new Uint8Array(await (await fetch(origin + '/models/model.onnx')).arrayBuffer());
    const results = [];
    for (const preferGPU of [false, ...(gpuRequired ? [true] : [])]) {
      const session = await JaxSession.create(bytes, preferGPU);
      try {
        let maxLogitDifference = 0;
        let argmaxMatches = 0;
        let positions = 0;
        for (let repeat = 0; repeat < 2; repeat++) {
          for (const row of reference.rows) {
            const output = await session.run(row.ids);
            if (JSON.stringify(output.dims) !== JSON.stringify(row.dims)) throw new Error('Parity output shape mismatch');
            for (let i = 0; i < row.logits.length; i++) maxLogitDifference = Math.max(maxLogitDifference, Math.abs(row.logits[i] - output.data[i]));
            const width = row.dims[2];
            for (let position = 0; position < row.ids.length; position++) {
              const offset = position * width;
              let expected = 0, actual = 0;
              for (let label = 1; label < width; label++) {
                if (row.logits[offset + label] > row.logits[offset + expected]) expected = label;
                if (output.data[offset + label] > output.data[offset + actual]) actual = label;
              }
              argmaxMatches += Number(expected === actual);
              positions++;
            }
          }
        }
        results.push({backend: session.backend, maxLogitDifference, argmaxMatches, positions});
      } finally { session.dispose(); }
    }
    return results;
  }, {origin, reference, gpuRequired: process.env.GAMMA_TEST_WEBGPU === '1'});
  for (const row of evidence.model.parity) {
    assert(row.maxLogitDifference < 0.0005, 'JAX logits must match the independent FP32 ONNX reference');
    assert.equal(row.argmaxMatches, row.positions);
  }
  evidence.model.gpuFallback = [];
  evidence.model.windowsAdapters = await verifyWindowsWebGPU(context, origin, process.env.GAMMA_TEST_WEBGPU === '1');
  for (const mode of ['unavailable', ...(process.env.GAMMA_TEST_WEBGPU === '1' ? ['allocation', 'kernel'] : [])]) {
    const fallback = await context.newPage();
    fallback.on('pageerror', error => failures.push(error.message));
    try {
      await fallback.addInitScript(mode => {
        if (mode === 'unavailable') {
          Object.defineProperty(navigator, 'gpu', {value: {requestAdapter: async () => null}});
          return;
        }
        const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
        navigator.gpu.requestAdapter = async options => {
          const adapter = await requestAdapter(options);
          if (!adapter) throw new Error('GPU failure test needs an adapter');
          const requestDevice = adapter.requestDevice.bind(adapter);
          adapter.requestDevice = async options => {
            const device = await requestDevice(options);
            const method = mode === 'allocation' ? 'createBuffer' : 'createShaderModule';
            const original = device[method].bind(device);
            let calls = 0;
            Object.defineProperty(device, method, {
              value: (...args) => {
                calls++;
                // The backend's reusable four-byte buffer must succeed. Fail
                // the next allocation while loading actual model initializers.
                if (mode === 'allocation' && calls === 1) return original(...args);
                globalThis.__gammaGpuFailure = {mode, calls};
                throw new Error('Intentional GPU ' + mode + ' failure');
              },
            });
            return device;
          };
          return adapter;
        };
      }, mode);
      await fallback.goto(origin + '/fixture.html');
      const result = await fallback.evaluate(async origin => {
        const engine = await import(origin + '/test-engine.mjs');
        const result = await engine.analyzeText('The students has a notebook.', {modelBaseUrl: origin + '/models/'});
        return {backend: result.backend, modelError: result.modelError, actual: engine.applySuggestions(result.text, result.suggestions),
          modelEdit: result.suggestions.some(edit => edit.source === 'model'), injectedFailure: globalThis.__gammaGpuFailure};
      }, origin);
      assert.equal(result.backend, 'wasm', 'GPU failure must fall back to the real JAX WASM model');
      assert.equal(result.modelError, undefined);
      assert.equal(result.actual, 'The students have a notebook.');
      assert.equal(result.modelEdit, true);
      if (mode !== 'unavailable') {
        assert.equal(result.injectedFailure?.mode, mode);
        if (mode === 'allocation') assert.equal(result.injectedFailure.calls, 2);
        else assert(result.injectedFailure.calls >= 1, 'At least one real kernel compilation must have been attempted');
      }
      evidence.model.gpuFallback.push({mode, ...result});
    } finally { await fallback.close(); }
  }
  const modelEdit = evidence.model.rows.find(row => row.source === 'The students has a notebook.');
  assert(modelEdit.edits.some(edit => edit.source === 'model') && modelEdit.actual === modelEdit.target,
    'Model must correctly suggest an edit beyond deterministic rules');
  evidence.model.regressionFailures = evidence.model.rows.filter(row => row.actual !== row.target);
  assert.equal(evidence.model.regressionFailures.length, 0, 'Every grammar smoke case must match; misses are failures, not just reported accuracy');

  let background = context.serviceWorkers().find(worker => worker.url().includes('background.mjs'));
  if (!background) background = await context.waitForEvent('serviceworker', {predicate: worker => worker.url().includes('background.mjs')});
  const extensionId = new URL(background.url()).hostname;
  await background.evaluate(async () => {
    let hash = 2166136261;
    const pattern = 'http://127.0.0.1/*';
    for (const char of pattern) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
    await chrome.storage.local.set({enabled: true, useAI: true});
    await chrome.scripting.registerContentScripts([{id: `gamma-site-${(hash >>> 0).toString(16)}`, matches: [pattern], js: ['content.js'], allFrames: false, runAt: 'document_idle'}]);
  });
  const fixture = await context.newPage();
  fixture.on('pageerror', error => failures.push(error.message));
  await fixture.goto(origin + '/fixture.html');
  await fixture.locator('#draft').focus();
  await fixture.waitForFunction(() => Boolean(document.querySelector('[data-gamma-ignore]')), {timeout: 30_000});
  // CDP lets this test inspect the closed shadow panel without changing the
  // shipping extension to expose page text or private suggestion state.
  evidence.extension.backend = await waitPanel(fixture, /suggestions?.*Local AI/u);
  assert.match(evidence.extension.backend, process.env.GAMMA_TEST_WEBGPU === '1' ? /Local AI · WebGPU/u : /Local AI · CPU/u);
  await acceptFirst(fixture);
  await waitPanel(fixture, /1 suggestion/u);
  await acceptFirst(fixture);
  assert.equal(await fixture.locator('#draft').inputValue(), 'She has a friend.');
  evidence.extension.textareaCorrection = true;
  await fixture.locator('#draft').fill('my cat is hungry.');
  await waitPanel(fixture, /No suggestions from this checker.*Local AI/u);
  assert.equal(await fixture.locator('#draft').inputValue(), 'my cat is hungry.');
  evidence.extension.cleanSentencePreserved = true;
  await background.evaluate(() => chrome.storage.local.set({useAI: false}));
  await fixture.locator('#draft').fill('halo');
  await waitPanel(fixture, /1 suggestion.*Local rules/u);
  await openSuggestions(fixture);
  await waitPanel(fixture, /Did you mean “hello” as a greeting/u, 90_000, '.body');
  await acceptFirst(fixture);
  assert.equal(await fixture.locator('#draft').inputValue(), 'hello');
  await fixture.locator('#draft').fill('A halo surrounds the moon.');
  await waitPanel(fixture, /No suggestions from this checker/u);
  await fixture.locator('#draft').fill('😀. Speling in a sentnce.');
  await waitPanel(fixture, /2 suggestions.*Local rules/u);
  await acceptFirst(fixture);
  await waitPanel(fixture, /1 suggestion/u);
  await acceptFirst(fixture);
  assert.equal(await fixture.locator('#draft').inputValue(), '😀. Spelling in a sentence.');
  evidence.extension.dictionarySpellingWithoutAI = true;
  await fixture.locator('#draft').fill(protectedSpelling);
  await waitPanel(fixture, /No suggestions from this checker/u);
  assert.equal(await fixture.locator('#draft').inputValue(), protectedSpelling);
  evidence.extension.protectedDictionaryTokens = true;
  await background.evaluate(() => chrome.storage.local.set({useAI: true}));
  await fixture.locator('#draft').fill('😀. The students has a notebook.');
  await waitPanel(fixture, /1 suggestion.*Local AI/u);
  await acceptFirst(fixture);
  assert.equal(await fixture.locator('#draft').inputValue(), '😀. The students have a notebook.');
  evidence.extension.modelCorrectionWithUTF16 = true;
  for (const id of ['private', 'payment', 'optout']) {
    await fixture.locator('#' + id).focus();
    await fixture.waitForFunction(() => !document.querySelector('[data-gamma-ignore]'));
    assert.equal(await fixture.locator('#' + id).inputValue(), 'She have a freind.');
  }
  evidence.extension.sensitiveFieldsExcluded = true;
  await fixture.locator('#plain').focus();
  await waitPanel(fixture, /1 suggestion/u);
  await acceptFirst(fixture);
  assert.equal(await fixture.locator('#plain').textContent(), 'She has a book.');
  evidence.extension.plainContenteditableCorrection = true;
  await fixture.locator('#rich').focus();
  await waitPanel(fixture, /This rich editor is not supported yet/u);
  assert.equal(await fixture.locator('#rich strong').textContent(), 'She have a book.');
  evidence.extension.richDomPreserved = true;
  await background.evaluate(() => chrome.storage.local.set({enabled: false}));
  await fixture.waitForFunction(() => !document.querySelector('[data-gamma-ignore]'));
  await background.evaluate(() => chrome.storage.local.set({enabled: true}));
  await fixture.locator('#draft').fill('She have a book.');
  await fixture.locator('#draft').focus();
  await waitPanel(fixture, /1 suggestion/u);
  evidence.extension.pauseResume = true;
  await background.evaluate(async () => {
    const tabs = await chrome.tabs.query({url: 'http://127.0.0.1/*'});
    for (const tab of tabs) await chrome.tabs.sendMessage(tab.id, {target: 'content', action: 'site-disabled'}).catch(() => undefined);
  });
  await fixture.waitForFunction(() => !document.querySelector('[data-gamma-ignore]'));
  await background.evaluate(async () => {
    const tabs = await chrome.tabs.query({url: 'http://127.0.0.1/*'});
    for (const tab of tabs) await chrome.tabs.sendMessage(tab.id, {target: 'content', action: 'site-enabled'}).catch(() => undefined);
  });
  await waitPanel(fixture, /1 suggestion/u);
  evidence.extension.siteReenable = true;
  const offscreen = await background.evaluate(() => chrome.runtime.getContexts({contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT]}));
  assert.equal(offscreen.length, 1);
  evidence.extension.offscreenContext = true;
  evidence.extension.nativePermissionDialog = 'Not automated; test-only copy grants localhost. Shipping manifest remains optional-site-only.';
  evidence.extension.extensionId = extensionId;
  assert.deepEqual(evidence.network, [], 'Browser checks must not send external requests');
  assert.deepEqual(failures, [], 'Browser must have no uncaught page exceptions');
  evidence.passed = true;
  console.log(JSON.stringify(evidence, null, 2));
} catch (error) {
  evidence.passed = false;
  failures.push(error instanceof Error ? error.message : 'Browser check failed');
  throw error;
} finally {
  await writeFile(join(artifactDir, 'browser-smoke.json'), JSON.stringify({...evidence, errors: failures}, null, 2));
  if (context) await context.close();
  if (server?.listening) await new Promise(resolveClosed => server.close(resolveClosed));
  // This directory was created by mkdtemp for this run only.
  if (dirname(temporary) === tmpdir() && temporary.startsWith(join(tmpdir(), 'gamma-browser-check-'))) await rm(temporary, {recursive: true, force: true});
}
