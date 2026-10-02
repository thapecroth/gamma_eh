import assert from 'node:assert/strict';
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { browserArguments, findChromium } from './browser-environment.mjs';
import { extensionElement } from './extension-shadow.mjs';
import { acquireLock } from './agent-runtime.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = join(root, 'docs/assets/webstore');

async function artwork() {
  const browser = await chromium.launch({executablePath: await findChromium(), headless: true, args: browserArguments()});
  try {
    const page = await browser.newPage({deviceScaleFactor: 1});
    const svg = await readFile(join(root, 'apps/extension/icons/icon.svg'), 'utf8');
    for (const size of [16, 32, 48, 128]) {
      await page.setViewportSize({width: size, height: size});
      await page.setContent(`<style>html,body{margin:0;background:transparent}svg{display:block;width:100%;height:100%}</style>${svg}`);
      await page.screenshot({path: join(root, 'apps/extension/icons', `icon-${size}.png`), omitBackground: true});
    }
    await page.setViewportSize({width: 440, height: 280});
    await page.setContent(`<style>body{margin:0}</style>${await readFile(join(output, 'promo.svg'), 'utf8')}`);
    await page.screenshot({path: join(output, 'promo-440x280.png')});
  } finally { await browser.close(); }
}

const demo = `<!doctype html><html lang="en"><meta charset="utf-8"><title>Gamma EH writing demo</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5f2eb;color:#36312e;font:18px/1.6 Arial,sans-serif}main{max-width:940px;margin:72px auto}.brand{color:#76609b;font-weight:bold;letter-spacing:.02em}h1{font:48px Georgia,serif;margin:24px 0 12px}.intro{color:#625e58;margin:0 0 40px}.editor{background:#fff;border:1px solid #e2dbd1;border-radius:18px;padding:30px}label{display:block;font-size:14px;font-weight:bold;margin-bottom:12px}textarea{font:24px/1.7 Georgia,serif;padding:20px;width:100%;height:230px;resize:none;border:1px solid #d7d0df;border-radius:10px;background:#fff;color:#36312e}textarea:focus{outline:2px solid #d6cce2}.foot{font-size:14px;color:#625e58;margin-top:26px}p{max-width:760px}</style>
<main><div class="brand">γ &nbsp; gamma eh</div><h1>A little clarity, right where you write.</h1><p class="intro">Review a suggestion. Keep your words on your device.</p><section class="editor"><label for="draft">A writing example</label><textarea id="draft" aria-label="Writing example">She have a freind.</textarea></section><p class="foot">Demonstration page · fictional text · suggestions from the installed extension</p></main></html>`;

async function screenshots() {
  const temporary = await mkdtemp(join(tmpdir(), 'gamma-webstore-'));
  const server = createServer((_request, response) => { response.setHeader('Content-Type', 'text/html; charset=utf-8'); response.end(demo); });
  const sockets = new Set();
  server.on('connection', socket => { sockets.add(socket); socket.once('close', () => sockets.delete(socket)); });
  let context;
  try {
    await new Promise((ready, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', ready); });
    const origin = `http://127.0.0.1:${server.address().port}`;
    const extension = join(temporary, 'extension');
    await cp(join(root, 'dist/extension'), extension, {recursive: true});
    // Exercise automatic activation with the unchanged shipping manifest.
    context = await chromium.launchPersistentContext(join(temporary, 'profile'), {
      channel: 'chromium', executablePath: await findChromium(), headless: true,
      args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, ...browserArguments()],
      viewport: {width: 1280, height: 800}, deviceScaleFactor: 1, timeout: 20_000,
    });
    context.setDefaultTimeout(10_000);
    const requests = [];
    context.on('request', request => {
      if (!request.url().startsWith(origin + '/') && !request.url().startsWith('chrome-extension://')) requests.push(request.resourceType());
    });
    await context.route('**/*', route => route.request().url().startsWith(origin + '/') || route.request().url().startsWith('chrome-extension://') ? route.continue() : route.abort());
    const page = await context.newPage();
    await page.goto(origin);
    await page.locator('#draft').focus();
    // Automatic checking wakes the background worker on the first supported field.
    const background = context.serviceWorkers()[0] ?? await context.waitForEvent('serviceworker');
    const extensionId = new URL(background.url()).hostname;
    const popupPromise = context.waitForEvent('page');
    await background.evaluate(async ({url, origin}) => {
      const [tab] = await chrome.tabs.query({url: origin + '/*'});
      await chrome.tabs.update(tab.id, {active: true});
      await chrome.windows.update(tab.windowId, {focused: true});
      await chrome.tabs.create({url, active: false});
    }, {url: `chrome-extension://${extensionId}/popup.html`, origin});
    const popup = await popupPromise;
    await popup.locator('#site-name').filter({hasText: '127.0.0.1'}).waitFor();
    assert.equal(await popup.locator('#use-ai').isChecked(), true, 'Local AI must be enabled by default');
    await popup.locator('#site-detail').filter({hasText: 'Enabled automatically.'}).waitFor();
    assert.equal(await popup.locator('#enable-site').isVisible(), false);
    await page.locator('#draft').focus();
    async function shadow(selector, click = false) {
      const end = Date.now() + 10_000;
      while (Date.now() < end) {
        const node = await extensionElement(context, page, selector);
        if (node) {
          try {
            return (await node.session.send('Runtime.callFunctionOn', {objectId: node.objectId,
              functionDeclaration: click ? 'function(){this.click();return true}' : 'function(){return this.textContent}', returnByValue: true})).result.value;
          } finally { await node.session.detach(); }
        }
        await new Promise(ready => setTimeout(ready, 100));
      }
      throw new Error(`Missing extension UI: ${selector}`);
    }
    await shadow('.badge.has-issues');
    await shadow('.badge', true);
    assert.match(await shadow('.panel .new'), /has|friend/u);
    await page.screenshot({path: join(output, 'suggestions-1280x800.png')});
    for (let index = 0; index < 2; index++) {
      const previous = await page.locator('#draft').inputValue();
      await shadow('.badge.has-issues');
      await shadow('.badge', true);
      await shadow('.panel .accept', true);
      await page.waitForFunction(text => document.querySelector('#draft').value !== text, previous);
    }
    await page.waitForFunction(() => document.querySelector('#draft').value === 'She has a friend.');
    await shadow('.badge[aria-label*="No suggestions"]');
    await page.screenshot({path: join(output, 'correction-1280x800.png')});
    assert.deepEqual(requests, [], 'Store demo must not make external requests');
  } finally {
    await context?.close();
    if (server.listening) await new Promise(ready => { server.close(ready); for (const socket of sockets) socket.destroy(); });
    await rm(temporary, {recursive: true, force: true});
  }
}

if (!['artwork', 'screenshots'].includes(process.argv[2])) throw new Error('Usage: webstore-assets.mjs artwork|screenshots');
const lock = await acquireLock(root);
try {
  await mkdir(output, {recursive: true});
  if (process.argv[2] === 'artwork') await artwork();
  else await screenshots();
} finally { await lock.release(); }
