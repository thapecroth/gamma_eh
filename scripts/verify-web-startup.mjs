import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, webkit } from '@playwright/test';
import { preview } from 'vite';
import { findChromium } from './browser-environment.mjs';
import { buildPaths } from './paths.mjs';
import { webBase } from './web-base.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const {webOutput, artifactDir} = buildPaths(root);
const browserName = process.env.GAMMA_STARTUP_BROWSER ?? 'chromium';
assert(['chromium', 'webkit'].includes(browserName), 'Use chromium or webkit');
const evidence = {browser: browserName, checks: {}, errors: [], uploads: []};
let server;
let browser;
const ready = page => page.locator('.status-dot.ready').waitFor({timeout: 60_000});
try {
  server = await preview({configFile: join(root, 'apps/web/vite.config.ts'),
    build: {outDir: webOutput}, preview: {host: '127.0.0.1', port: 0, open: false}});
  const url = `http://127.0.0.1:${server.httpServer.address().port}${webBase()}`;
  browser = browserName === 'webkit' ? await webkit.launch() : await chromium.launch({
    executablePath: await findChromium(), args: ['--no-sandbox', '--disable-webgpu']});
  const context = await browser.newContext();
  context.on('request', request => {
    if (!['http:', 'https:'].includes(new URL(request.url()).protocol)) return;
    if (!['GET', 'HEAD'].includes(request.method()) || request.postData()) evidence.uploads.push(request.method());
    assert(new URL(request.url()).origin === new URL(url).origin, 'Keep all assets on the local origin');
    assert(new URL(request.url()).pathname.startsWith(webBase()), 'Keep assets inside the Pages base path');
  });
  const page = await context.newPage();
  page.on('pageerror', error => evidence.errors.push(error.message));
  let releaseDownload;
  const heldDownload = new Promise(resolve => { releaseDownload = resolve; });
  await page.route('**/models/model.onnx', async route => {
    await heldDownload;
    await route.continue().catch(() => {}); // Turning AI off cancels this request.
  });
  await Promise.all([page.waitForRequest('**/models/model.onnx'), page.goto(url)]);
  await page.locator('.status-dot.loading-ai').waitFor();
  assert(await page.locator('.suggestion-card').count() > 0, 'Show rules during a stalled model download');
  const editor = page.getByLabel('Your writing', {exact: true});
  await editor.fill('😀. She have a freind.');
  await page.locator('.status-dot.loading-ai').waitFor();
  await page.getByRole('button', {name: 'Accept all suggestions', exact: true}).click();
  assert.equal(await editor.inputValue(), '😀. She has a friend.');
  evidence.checks.rulesDuringDownload = true;
  await page.getByRole('checkbox').uncheck();
  await ready(page);
  assert.match(await page.locator('.backend-status').textContent(), /Local rules/u);
  evidence.checks.cancelStalledAI = true;
  releaseDownload();
  await page.unroute('**/models/model.onnx');
  await page.getByRole('button', {name: 'Try local AI', exact: true}).click();
  await ready(page);
  assert.match(await page.locator('.backend-status').textContent(), /Local AI · CPU/u);
  assert((await page.locator('.suggestion-origin').allTextContents()).includes('LOCAL MODEL'));
  await context.setOffline(true);
  await editor.fill('😀. The students has a notebook.');
  await ready(page);
  await page.getByRole('button', {name: 'Accept all suggestions', exact: true}).click();
  assert.equal(await editor.inputValue(), '😀. The students have a notebook.');
  evidence.checks.realModelRetryAndOffline = true;
  await context.setOffline(false);
  await page.close();

  const retry = await context.newPage();
  retry.on('pageerror', error => evidence.errors.push(error.message));
  await retry.route('**/models/**', route => route.fulfill({status: 503, body: 'Unavailable'}));
  await retry.goto(url);
  await ready(retry);
  await retry.locator('.model-warning').waitFor();
  await retry.unroute('**/models/**');
  // Both clicks occur in one task, well inside the 450 ms typing debounce.
  await retry.getByRole('checkbox').evaluate(toggle => { toggle.click(); toggle.click(); });
  await retry.waitForFunction(() => document.querySelector('.backend-status')?.textContent.includes('Local AI · CPU'));
  assert.equal(await retry.locator('.model-warning').count(), 0);
  evidence.checks.immediateToggleRetry = true;
  await retry.close();

  // Hang JAX adapter initialization in Chromium. WebKit cannot reliably route
  // worker scripts, so hold its preferred worker's request before execution.
  // Both replacement workers must execute the actual WASM model.
  const stalledGPU = await context.newPage();
  stalledGPU.on('pageerror', error => evidence.errors.push(error.message));
  await stalledGPU.clock.install();
  await stalledGPU.addInitScript(holdRequest => {
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', event => {
          if (event.data.gammaTestAdapterPending) window.gammaTestAdapterPending = true;
        });
      }
      postMessage(message, ...args) {
        if (holdRequest && message.useAI && message.preferWebGPU !== false) {
          window.gammaTestAdapterPending = true;
          return;
        }
        super.postMessage(message, ...args);
      }
    };
  }, browserName === 'webkit');
  if (browserName === 'chromium') await stalledGPU.route('**/assets/inference-worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({response, body: `Object.defineProperty(navigator, 'gpu', {value: {
      requestAdapter() { self.postMessage({gammaTestAdapterPending: true}); return new Promise(() => {}); }
    }});\n${await response.text()}`});
  });
  await stalledGPU.goto(url);
  await stalledGPU.waitForFunction(() => window.gammaTestAdapterPending === true);
  await stalledGPU.locator('.status-dot.loading-ai').waitFor();
  await stalledGPU.clock.fastForward(30_001);
  await ready(stalledGPU);
  assert.match(await stalledGPU.locator('.backend-status').textContent(), /Local AI · CPU/u);
  assert.equal(await stalledGPU.locator('.model-warning').count(), 0);
  await stalledGPU.getByRole('button', {name: 'Try local AI', exact: true}).click();
  await ready(stalledGPU);
  assert((await stalledGPU.locator('.suggestion-origin').allTextContents()).includes('LOCAL MODEL'));
  evidence.checks.hangingAIRecoversOnCPU = browserName === 'webkit' ? 'stalled-worker' : 'stalled-gpu-adapter';
  await stalledGPU.close();
  assert.deepEqual(evidence.errors, []);
  assert.deepEqual(evidence.uploads, []);
  evidence.passed = true;
  console.log(JSON.stringify(evidence, null, 2));
} catch (error) {
  evidence.passed = false;
  evidence.errors.push(error instanceof Error ? error.message : 'Startup verification failed');
  throw error;
} finally {
  await mkdir(artifactDir, {recursive: true});
  await writeFile(join(artifactDir, `web-startup-${browserName}.json`), JSON.stringify(evidence, null, 2) + '\n');
  try { await browser?.close(); }
  finally {
    if (server) await new Promise(resolve => {
      server.httpServer.close(resolve);
      server.httpServer.closeAllConnections();
    });
  }
}
