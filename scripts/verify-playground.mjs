import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { preview } from 'vite';
import { browserArguments, findChromium } from './browser-environment.mjs';
import { buildPaths } from './paths.mjs';
import { webBase } from './web-base.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const { webOutput, artifactDir } = buildPaths(root);
const gpuRequired = process.env.GAMMA_TEST_WEBGPU === '1';
const base = webBase();
const evidence = { basePath: base, extensionLoaded: false, checks: {}, externalRequests: [], outsideBaseRequests: [], uploads: [], errors: [] };
await mkdir(artifactDir, { recursive: true });
let server;
let browser;

async function ready(page) {
  await page.locator('.status-dot.ready').waitFor({ timeout: 90_000 });
}

async function verifyUnderlines(page, expected) {
  const marks = page.locator('.writing-error');
  if (expected) assert.deepEqual(await marks.allTextContents(), expected);
  assert((await marks.count()) > 0, 'Expected errors to be underlined in the draft');
  const failures = await page.evaluate(() => {
    const editor = document.querySelector('.writing-editor');
    const layer = document.querySelector('.writing-highlight-layer');
    const style = getComputedStyle(editor);
    const bounds = editor.getBoundingClientRect();
    // Independently measure native textarea typography with a single text node.
    const reference = document.createElement('div');
    for (const property of ['boxSizing', 'font', 'letterSpacing', 'wordSpacing', 'padding', 'whiteSpace', 'overflowWrap', 'tabSize']) {
      reference.style[property] = style[property];
    }
    Object.assign(reference.style, { position: 'fixed', visibility: 'hidden', border: '0',
      left: `${bounds.left - editor.scrollLeft}px`, top: `${bounds.top - editor.scrollTop}px`, width: `${editor.clientWidth}px` });
    reference.textContent = editor.value + '\u200b';
    document.body.append(reference);
    const failures = [];
    try {
      if (layer.clientWidth !== editor.clientWidth || layer.clientHeight !== editor.clientHeight) failures.push('Highlight clipping must match the textarea content area');
      if (getComputedStyle(layer).pointerEvents !== 'none' || layer.getAttribute('aria-hidden') !== 'true') failures.push('Highlights must leave selection and accessibility to the textarea');
      for (const mark of document.querySelectorAll('.writing-error')) {
        const start = Number(mark.dataset.start);
        const end = Number(mark.dataset.end);
        if (mark.textContent !== editor.value.slice(start, end)) failures.push('Underline does not match its UTF-16 source range');
        const decoration = getComputedStyle(mark);
        if (decoration.textDecorationLine !== 'underline' || decoration.textDecorationStyle !== 'wavy' || decoration.textDecorationColor !== 'rgb(196, 63, 63)') failures.push('Expected a red wavy underline');
        const nativeRange = document.createRange();
        nativeRange.setStart(reference.firstChild, start);
        nativeRange.setEnd(reference.firstChild, end);
        const actualRange = document.createRange();
        actualRange.selectNodeContents(mark);
        const expectedRects = [...nativeRange.getClientRects()];
        const actualRects = [...actualRange.getClientRects()];
        if (expectedRects.length !== actualRects.length || expectedRects.some((rect, index) =>
          ['left', 'top', 'width', 'height'].some(key => Math.abs(rect[key] - actualRects[index][key]) > 1))) {
          failures.push(`Underline is misaligned for ${mark.textContent}`);
        }
      }
    } finally { reference.remove(); }
    return failures;
  });
  assert.deepEqual(failures, []);
}

try {
  let siteUrl = process.env.GAMMA_PLAYGROUND_URL;
  if (!siteUrl) {
    server = await preview({ configFile: join(root, 'apps/web/vite.config.ts'),
      build: { outDir: webOutput }, preview: { host: '127.0.0.1', port: 0, open: false } });
    siteUrl = `http://127.0.0.1:${server.httpServer.address().port}${base}`;
  }
  const site = new URL(siteUrl);
  assert(['http:', 'https:'].includes(site.protocol) && !site.username && !site.password && !site.search && !site.hash, 'Use an HTTP(S) site URL without credentials, query, or fragment');
  assert.equal(site.pathname, base, 'The site URL must match GAMMA_WEB_BASE');
  const origin = site.origin;
  evidence.siteUrl = site.href;
  browser = await chromium.launch({ executablePath: await findChromium(), headless: true,
    args: ['--disable-extensions', ...browserArguments(), ...(!gpuRequired ? ['--disable-webgpu'] : [])] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
  await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin });
  const requests = [];
  context.on('request', request => {
    const url = request.url();
    requests.push(url);
    const parsed = new URL(url);
    if (['http:', 'https:'].includes(parsed.protocol)) {
      if (parsed.origin !== origin) evidence.externalRequests.push(url);
      else if (!parsed.pathname.startsWith(base)) evidence.outsideBaseRequests.push(url);
    }
    if (!['GET', 'HEAD'].includes(request.method()) || request.postData() !== null) evidence.uploads.push(request.method());
  });
  await context.route('**/*', route => {
    const url = new URL(route.request().url());
    return ['http:', 'https:'].includes(url.protocol) && (url.origin !== origin || !url.pathname.startsWith(base)) ? route.abort() : route.continue();
  });
  const page = await context.newPage();
  page.on('pageerror', error => evidence.errors.push(error.message));
  await page.goto(site.href);
  await ready(page);
  const editor = page.getByLabel('Your writing', { exact: true });
  const acceptAll = page.getByRole('button', { name: 'Accept all suggestions', exact: true });
  const undo = page.getByRole('button', { name: 'Undo correction', exact: true });
  assert.equal(await page.getByRole('checkbox').isChecked(), true, 'Local AI must start enabled');
  assert.match(await page.locator('.backend-status').textContent(), gpuRequired ? /Local AI · WebGPU/u : /Local AI · CPU/u);
  assert.equal(await page.locator('.model-warning').count(), 0);
  assert(requests.some(url => new URL(url).pathname === `${base}models/model.onnx`), 'Default AI must execute the bundled model');
  assert.equal(context.serviceWorkers().length, 0);
  assert.equal(await page.evaluate(() => Boolean(globalThis.chrome?.runtime?.id)), false);
  evidence.checks.defaultLocalAIWithoutExtension = true;
  await verifyUnderlines(page);
  await page.getByRole('checkbox').uncheck();
  await ready(page);

  for (const name of ['Everyday writing', 'A spelling check', 'An email']) {
    await page.getByRole('button', { name, exact: true }).click();
    await ready(page);
    assert((await page.locator('.suggestion-card').count()) > 0, `${name} must produce suggestions`);
    assert.equal(await page.getByRole('button', { name, exact: true }).getAttribute('aria-pressed'), 'true');
  }
  const sample = await editor.inputValue();
  const initialCount = await page.locator('.suggestion-card').count();
  assert.equal(await page.locator('.writing-error').count(), initialCount);
  await page.getByRole('button', { name: /^Dismiss:/u }).first().click();
  assert.equal(await editor.inputValue(), sample);
  assert.equal(await page.locator('.suggestion-card').count(), initialCount - 1);
  assert.equal(await page.locator('.writing-error').count(), initialCount - 1, 'Dismiss must remove its underline');
  await page.getByRole('button', { name: 'Reset example', exact: true }).click();
  await ready(page);
  assert.equal(await editor.inputValue(), sample);
  assert.equal(await page.locator('.suggestion-card').count(), initialCount, 'Reset must refresh dismissed suggestions even when the text is identical');
  await verifyUnderlines(page);
  evidence.checks.examplesAndDismissReset = true;

  await editor.fill('😀. I would definitly help my freind. My freind is here.');
  assert.equal(await page.locator('.writing-error').count(), 0, 'Typing must clear stale underlines immediately');
  await ready(page);
  await verifyUnderlines(page, ['definitly', 'freind', 'freind']);
  await page.getByRole('button', { name: 'Accept', exact: true }).first().click();
  await ready(page);
  await verifyUnderlines(page, ['freind', 'freind']);
  evidence.checks.underlineUTF16RepeatedWordsAndAccept = true;

  await editor.fill(`${'A clear sentence.\n'.repeat(32)}😀\tI would definately help my freind.\n`);
  await ready(page);
  await verifyUnderlines(page, ['definately', 'freind']);
  await editor.evaluate(element => { element.scrollTop = element.scrollHeight; });
  await page.waitForFunction(() => document.querySelector('.writing-editor').scrollTop > 0);
  await verifyUnderlines(page);
  await editor.evaluate(element => { element.style.height = `${element.clientHeight + 80}px`; });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await verifyUnderlines(page);
  await editor.evaluate(element => element.style.removeProperty('height'));
  evidence.checks.underlineScrollResizeTabsAndTrailingNewline = true;

  await editor.fill('😀. She have a freind.');
  await ready(page);
  await page.getByRole('button', { name: 'Accept', exact: true }).first().click();
  assert.equal(await editor.inputValue(), '😀. She has a freind.');
  await undo.click();
  assert.equal(await editor.inputValue(), '😀. She have a freind.');
  await ready(page);
  await acceptAll.click();
  assert.equal(await editor.inputValue(), '😀. She has a friend.');
  await undo.click();
  assert.equal(await editor.inputValue(), '😀. She have a freind.');
  assert.equal(await undo.isDisabled(), true);
  await editor.fill('A fresh draft.');
  assert.equal(await undo.isDisabled(), true);
  await ready(page);
  assert.equal(await acceptAll.isDisabled(), true);
  evidence.checks.acceptUndoAndUTF16 = true;

  await page.getByRole('button', { name: 'Copy text', exact: true }).click();
  await page.getByText('Draft copied to clipboard.', { exact: true }).waitFor();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), 'A fresh draft.');
  await page.evaluate(() => { navigator.clipboard.writeText = async () => { throw new Error('Clipboard unavailable'); }; });
  await page.getByRole('button', { name: 'Copy text', exact: true }).click();
  await page.getByText('Couldn’t copy automatically. Select your draft and copy it instead.', { exact: true }).waitFor();
  assert.equal(await editor.evaluate(element => element.selectionEnd - element.selectionStart), 'A fresh draft.'.length);
  evidence.checks.clipboardAndFallback = true;

  await page.getByRole('button', { name: 'Clear', exact: true }).click();
  await ready(page);
  assert.equal(await editor.inputValue(), '');
  assert.equal(await acceptAll.isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: 'Copy text', exact: true }).isDisabled(), true);
  await page.getByText('Make room for your words.', { exact: true }).waitFor();
  evidence.checks.blankDraft = true;

  await page.getByRole('button', { name: 'Try local AI', exact: true }).click();
  assert.equal(await page.getByRole('checkbox').isChecked(), true);
  await ready(page);
  evidence.backend = await page.locator('.backend-status').textContent();
  assert.match(evidence.backend, gpuRequired ? /Local AI · WebGPU/u : /Local AI · CPU/u);
  if (gpuRequired) evidence.adapter = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const { vendor, architecture, description, isFallbackAdapter } = adapter.info;
    return { vendor, architecture, description, isFallbackAdapter };
  });
  assert.equal(await page.locator('.model-warning').count(), 0);
  assert((await page.locator('.suggestion-origin').allTextContents()).includes('LOCAL MODEL'), 'AI example must exercise an actual model-origin correction');
  assert(requests.some(url => new URL(url).pathname === `${base}models/model.onnx`));
  await acceptAll.click();
  assert.match(await editor.inputValue(), /^The students have a notebook\./u);
  evidence.checks.actualLocalModel = true;

  // Once the bundled weights have loaded, fresh inference needs no connection.
  await context.setOffline(true);
  await editor.fill('😀. The students has a notebook.');
  await ready(page);
  await acceptAll.click();
  assert.equal(await editor.inputValue(), '😀. The students have a notebook.');
  await ready(page);
  await page.getByRole('checkbox').focus();
  await page.getByRole('checkbox').press('Space');
  await ready(page);
  assert.match(await page.locator('.backend-status').textContent(), /Local rules/u);
  await editor.fill('She have a freind.');
  await ready(page);
  await acceptAll.click();
  assert.equal(await editor.inputValue(), 'She has a friend.');
  evidence.checks.offlineInferenceAndKeyboardToggle = true;
  await context.setOffline(false);

  await page.getByRole('button', { name: 'An email', exact: true }).click();
  await ready(page);
  await editor.blur();
  await page.screenshot({ path: join(artifactDir, 'playground-desktop.png'), fullPage: true });
  for (const width of [320, 390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `No horizontal overflow at ${width}px`);
    await verifyUnderlines(page);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: join(artifactDir, 'playground-mobile.png'), fullPage: true });
  evidence.checks.responsiveLayout = true;

  // Hold a real rule-worker reply, switch modes on the same draft, then release
  // the older reply. Its matching text must not override the newer AI result.
  const stale = await context.newPage();
  stale.on('pageerror', error => evidence.errors.push(error.message));
  await stale.addInitScript(() => {
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      set onmessage(handler) {
        super.onmessage = event => {
          if (!window.delayedReplyReady && event.data.result?.text === 'She have a freind.' && event.data.result.backend === 'rules') {
            window.releaseDelayedReply = () => handler.call(this, event);
            window.delayedReplyReady = true;
          } else handler.call(this, event);
        };
      }
    };
  });
  await stale.goto(site.href);
  await ready(stale);
  await stale.getByRole('checkbox').uncheck();
  await ready(stale);
  await stale.getByLabel('Your writing', { exact: true }).fill('She have a freind.');
  await stale.waitForFunction(() => window.delayedReplyReady === true);
  await stale.getByRole('checkbox').check();
  await ready(stale);
  assert.match(await stale.locator('.backend-status').textContent(), /Local AI/u);
  await stale.evaluate(async () => {
    window.releaseDelayedReply();
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
  assert.match(await stale.locator('.backend-status').textContent(), /Local AI/u, 'An older result with identical text must not override a newer mode');
  await verifyUnderlines(stale, ['have', 'freind']);
  await stale.close();
  evidence.checks.staleWorkerReplyIgnored = true;

  // A separate tab proves the failure path without a warmed-up model session.
  const fallback = await context.newPage();
  fallback.on('pageerror', error => evidence.errors.push(error.message));
  await fallback.route('**/models/**', route => route.fulfill({ status: 503, body: 'Unavailable' }));
  await fallback.goto(site.href);
  await ready(fallback);
  assert.equal(await fallback.getByRole('checkbox').isChecked(), true);
  await fallback.locator('.model-warning').waitFor();
  await fallback.getByLabel('Your writing', { exact: true }).fill('She have a freind.');
  await ready(fallback);
  await verifyUnderlines(fallback, ['have', 'freind']);
  await fallback.getByRole('button', { name: 'Accept all suggestions', exact: true }).click();
  assert.equal(await fallback.getByLabel('Your writing', { exact: true }).inputValue(), 'She has a friend.');
  evidence.checks.modelFailureFallsBackToRules = true;
  await ready(fallback);
  await fallback.unroute('**/models/**');
  await fallback.getByRole('checkbox').uncheck();
  await ready(fallback);
  await fallback.getByRole('checkbox').check();
  await ready(fallback);
  assert.match(await fallback.locator('.backend-status').textContent(), /Local AI/u);
  assert.equal(await fallback.locator('.model-warning').count(), 0);
  evidence.checks.modelRetry = true;
  await fallback.close();

  await editor.fill('This draft belongs only to this tab.');
  await page.reload();
  await ready(page);
  assert.equal(await editor.inputValue(), sample, 'Private drafts must not persist across a reload');
  assert.equal(await page.getByRole('checkbox').isChecked(), true);
  assert.deepEqual(evidence.externalRequests, []);
  assert.deepEqual(evidence.outsideBaseRequests, []);
  assert.deepEqual(evidence.uploads, []);
  assert.deepEqual(evidence.errors, []);
  evidence.checks.ephemeralDraftAndLocalRequests = true;
  evidence.passed = true;
  console.log(JSON.stringify(evidence, null, 2));
} catch (error) {
  evidence.passed = false;
  evidence.errors.push(error instanceof Error ? error.message : 'Playground verification failed');
  throw error;
} finally {
  try {
    await writeFile(join(artifactDir, 'playground-smoke.json'), JSON.stringify(evidence, null, 2) + '\n');
    await writeFile(join(artifactDir, `playground-smoke-${gpuRequired ? 'webgpu' : 'cpu'}.json`), JSON.stringify(evidence, null, 2) + '\n');
  }
  finally {
    try { if (browser) await browser.close(); }
    finally {
      if (server) await new Promise(resolve => {
        server.httpServer.close(resolve);
        server.httpServer.closeAllConnections();
      });
    }
  }
}
