import assert from 'node:assert/strict';
import { cp, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { buildPaths } from './paths.mjs';
import { browserArguments, findChromium } from './browser-environment.mjs';
import { acquireLock, hash, jsonFile, runId, sourceStamp } from './agent-runtime.mjs';

export const fields = ['draft', 'private', 'payment', 'optout', 'plain', 'rich'];
const actions = ['popup', 'focus', 'fill', 'read', 'accept', 'dismiss', 'close', 'assertText', 'assertPanel', 'assertBackend', 'staleAccept'];
const stepSchema = {type: 'object', additionalProperties: false, required: ['action', 'field', 'value', 'expected'], properties: {
  action: {type: 'string', enum: actions}, field: {type: ['string', 'null'], enum: [...fields, 'enabled', 'useAI', null]},
  value: {type: ['string', 'boolean', 'null']}, expected: {type: ['string', 'null']},
}};
export const planSchema = {type: 'object', additionalProperties: false, required: ['reason', 'scenarios'], properties: {
  reason: {type: 'string'}, scenarios: {type: 'array', minItems: 1, maxItems: 4, items: {
    type: 'object', additionalProperties: false, required: ['id', 'steps'], properties: {
      id: {type: 'string', pattern: '^[a-z][a-z0-9-]{0,63}$'}, steps: {type: 'array', minItems: 2, maxItems: 20, items: stepSchema},
    },
  }},
}};
export const step = (action, field = null, value = null, expected = null) => ({action, field, value, expected});

export function validatePlan(plan) {
  if (!plan || Object.keys(plan).sort().join() !== 'reason,scenarios' || typeof plan.reason !== 'string' || !plan.reason.trim() || plan.reason.length > 2000
    || !Array.isArray(plan.scenarios) || plan.scenarios.length < 1 || plan.scenarios.length > 4) throw new Error('Malformed scenario plan');
  let count = 0;
  const ids = new Set();
  for (const scenario of plan.scenarios) {
    if (!scenario || Object.keys(scenario).sort().join() !== 'id,steps' || typeof scenario.id !== 'string' || !/^[a-z][a-z0-9-]{0,63}$/u.test(scenario.id) || ids.has(scenario.id)
      || !Array.isArray(scenario.steps) || scenario.steps.length < 2 || scenario.steps.length > 20) throw new Error('Malformed or duplicate scenario');
    ids.add(scenario.id);
    let assertions = 0;
    for (const item of scenario.steps) {
      count++;
      if (!item || Object.keys(item).sort().join() !== 'action,expected,field,value' || !actions.includes(item.action)) throw new Error('Out-of-scope browser action');
      const {action, field, value, expected} = item;
      if (typeof value === 'string' && value.length > 6000 || typeof expected === 'string' && expected.length > 6000) throw new Error('Browser text exceeds field limit');
      const requireNull = (...values) => { if (values.some(entry => entry !== null)) throw new Error(`Unexpected arguments for ${action}`); };
      if (action === 'popup') {
        if (!['enabled', 'useAI'].includes(field) || typeof value !== 'boolean') throw new Error('Invalid popup setting');
        requireNull(expected);
      } else if (['focus', 'read'].includes(action)) {
        if (!fields.includes(field)) throw new Error('Unknown fixture field'); requireNull(value, expected);
      } else if (['fill', 'staleAccept'].includes(action)) {
        if (!['draft', 'plain'].includes(field) || typeof value !== 'string') throw new Error('Only public plain fields may be filled');
        if (action === 'staleAccept' && (field !== 'draft' || expected !== value)) throw new Error('Stale acceptance must preserve mutated draft');
        if (action === 'fill') requireNull(expected);
      } else if (action === 'assertText') {
        if (!fields.includes(field) || typeof expected !== 'string') throw new Error('Invalid text assertion'); requireNull(value); assertions++;
      } else if (['assertPanel', 'assertBackend'].includes(action)) {
        if (!(action === 'assertPanel' ? ['visible', 'hidden'] : ['rules', 'ai']).includes(expected)) throw new Error('Invalid panel assertion');
        requireNull(field, value); assertions++;
      } else requireNull(field, value, expected);
    }
    if (!assertions) throw new Error('Every scenario needs an objective assertion');
  }
  if (count > 48) throw new Error('Scenario plan exceeds 48 actions');
  return plan;
}

const fixtureHtml = '<!doctype html><html lang="en"><title>Fictional Gamma EH fixture</title><body><h1>Agent extension fixture</h1><label>Draft<textarea id="draft">She have a freind.</textarea></label><label>Private<textarea id="private" data-private>She have a freind.</textarea></label><label>Payment<textarea id="payment" autocomplete="cc-number">She have a freind.</textarea></label><label>Opt out<textarea id="optout" spellcheck="false">She have a freind.</textarea></label><div id="plain" contenteditable="true" aria-label="Plain editor">She have a book.</div><div id="rich" contenteditable="true" aria-label="Rich editor"><strong>She have a book.</strong></div></body></html>';

export const mandatoryScenarios = [
  {id: 'rules-and-popup', steps: [step('popup', 'useAI', false), step('focus', 'draft'), step('assertBackend', null, null, 'rules'), step('accept'), step('accept'), step('assertText', 'draft', null, 'She has a friend.'), step('popup', 'enabled', false), step('assertPanel', null, null, 'hidden'), step('popup', 'enabled', true), step('fill', 'draft', 'She have a book.'), step('assertBackend', null, null, 'rules')]},
  {id: 'ai-opt-in', steps: [step('popup', 'useAI', true), step('focus', 'draft'), step('assertBackend', null, null, 'ai'), step('accept'), step('accept'), step('assertText', 'draft', null, 'She has a friend.'), step('popup', 'useAI', false), step('assertBackend', null, null, 'rules')]},
  {id: 'ai-model-offsets-and-stale', steps: [step('popup', 'useAI', true), step('fill', 'draft', '😀 The students has a notebook.'), step('assertBackend', null, null, 'ai'), step('staleAccept', 'draft', '😀 Newly changed text.', '😀 Newly changed text.'), step('fill', 'draft', '😀 The students has a notebook.'), step('assertBackend', null, null, 'ai'), step('accept'), step('assertText', 'draft', null, '😀 The students have a notebook.')]},
  {id: 'dismiss-preserves-text', steps: [step('focus', 'draft'), step('assertBackend', null, null, 'rules'), step('dismiss'), step('assertPanel', null, null, 'visible'), step('assertText', 'draft', null, 'She have a freind.')]},
  ...['private', 'payment', 'optout'].map(field => ({id: `protected-${field}`, steps: [step('focus', 'draft'), step('assertPanel', null, null, 'visible'), step('focus', field), step('assertPanel', null, null, 'hidden'), step('assertText', field, null, 'She have a freind.')]})),
  {id: 'plain-editable', steps: [step('focus', 'plain'), step('assertBackend', null, null, 'rules'), step('accept'), step('assertText', 'plain', null, 'She has a book.')]},
  {id: 'rich-preservation', steps: [step('focus', 'rich'), step('assertPanel', null, null, 'visible'), step('assertText', 'rich', null, 'She have a book.')]},
  {id: 'utf16-offsets', steps: [step('fill', 'draft', '😀 A freind.'), step('assertBackend', null, null, 'rules'), step('accept'), step('assertText', 'draft', null, '😀 A friend.')]},
  {id: 'stale-suggestion', steps: [step('focus', 'draft'), step('assertBackend', null, null, 'rules'), step('staleAccept', 'draft', '😀 Newly changed text.', '😀 Newly changed text.'), step('assertText', 'draft', null, '😀 Newly changed text.')]},
];

async function shadowNode(context, page, selector) {
  const session = await context.newCDPSession(page);
  const {root} = await session.send('DOM.getDocument', {depth: -1, pierce: true});
  function host(node) {
    if (node.attributes?.includes('data-gamma-ignore')) return node;
    for (const child of [...node.children ?? [], ...node.shadowRoots ?? []]) { const found = host(child); if (found) return found; }
  }
  const shadow = host(root)?.shadowRoots?.[0];
  const {nodeId} = shadow ? await session.send('DOM.querySelector', {nodeId: shadow.nodeId, selector}) : {};
  if (!nodeId) { await session.detach(); return null; }
  const {object} = await session.send('DOM.resolveNode', {nodeId});
  return {session, objectId: object.objectId};
}

async function panelText(context, page) {
  const node = await shadowNode(context, page, '.body');
  if (!node) return null;
  try { return (await node.session.send('Runtime.callFunctionOn', {objectId: node.objectId, functionDeclaration: 'function() { return this.textContent; }', returnByValue: true})).result.value; }
  finally { await node.session.detach(); }
}

async function waitForPanel(context, page, predicate, description, timeout = 60_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const text = await panelText(context, page);
    if (predicate(text)) return text;
    await new Promise(resolvePoll => setTimeout(resolvePoll, 100));
  }
  throw new Error(`Panel did not reach ${description}`);
}

export async function runExtensionScenarios({root, directory, plan = null, id = runId()}) {
  if (plan) validatePlan(plan);
  await mkdir(directory, {recursive: true, mode: 0o700});
  const stamp = await sourceStamp(root);
  const shippingPath = join(buildPaths(root).extensionOutput, 'manifest.json');
  const shippingBytes = await readFile(shippingPath);
  const temporary = await mkdtemp(join(directory, 'session-'));
  const evidence = {runId: id, ...stamp, planHash: hash(plan), manifestHash: hash(shippingBytes), fixtureHash: hash(fixtureHtml),
    permissionBoundary: 'Temporary extension copy has static localhost permission. Real popup enables scripts; native optional permission grant/revoke is not verified.',
    defaultSettings: null, scenarios: [], screenshots: [], blockedRequests: [], networkViolations: [], errors: [], passed: false};
  let context;
  let server;
  let fixture;
  let timeout;
  const sockets = new Set();
  let interrupted = false;
  const onSignal = () => {
    interrupted = true;
    evidence.interrupted = true;
    evidence.errors.push('Extension driver interrupted');
    void context?.close().catch(() => {});
    for (const socket of sockets) socket.destroy();
  };
  process.on('SIGINT', onSignal);
  process.on('SIGTERM', onSignal);
  try {
    const extension = join(temporary, 'extension');
    await cp(buildPaths(root).extensionOutput, extension, {recursive: true});
    const manifest = JSON.parse(shippingBytes.toString());
    assert.equal(manifest.host_permissions, undefined, 'Shipping extension must request optional host permissions');
    assert.equal(manifest.content_scripts, undefined);
    assert.equal(manifest.content_security_policy?.extension_pages,
      "script-src 'self' 'wasm-unsafe-eval'; object-src 'none'; worker-src 'self'; connect-src 'self'", 'Bundled-only extension CSP must remain intact');
    await writeFile(join(extension, 'manifest.json'), JSON.stringify({...manifest, host_permissions: ['http://127.0.0.1/*']}));
    evidence.fixtureGrantManifestHash = hash(await readFile(join(extension, 'manifest.json')));
    server = createServer((_request, response) => { response.setHeader('Content-Type', 'text/html'); response.end(fixtureHtml); });
    server.on('connection', socket => { sockets.add(socket); socket.once('close', () => sockets.delete(socket)); });
    await new Promise((ready, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', ready); });
    const origin = `http://127.0.0.1:${server.address().port}`;
    context = await chromium.launchPersistentContext(join(temporary, 'profile'), {channel: 'chromium', executablePath: await findChromium(), headless: true,
      args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, ...browserArguments()], viewport: {width: 1100, height: 800}, timeout: 20_000});
    if (interrupted) throw new Error('Extension driver interrupted');
    context.setDefaultTimeout(10_000);
    timeout = setTimeout(() => {
      evidence.errors.push('Extension driver exceeded its four-minute budget');
      void context.close().catch(() => {});
    }, 240_000);
    timeout.unref();
    await context.tracing.start({screenshots: true, snapshots: true});
    context.on('request', request => {
      const url = request.url();
      if (!url.startsWith(origin + '/') && !url.startsWith('chrome-extension://') && !url.startsWith('data:')) {
        evidence.networkViolations.push({kind: 'outside-fixture', resourceType: request.resourceType()});
      }
    });
    await context.route('**/*', route => {
      const url = route.request().url();
      if (url.startsWith(origin + '/') || url.startsWith('chrome-extension://') || url.startsWith('data:')) return route.continue();
      // Never retain a foreign URL, which could contain a secret.
      evidence.blockedRequests.push({kind: 'outside-fixture', resourceType: route.request().resourceType()});
      return route.abort('blockedbyclient');
    });
    context.on('page', page => page.on('pageerror', error => evidence.errors.push(error.message.slice(0, 500))));
    let background = context.serviceWorkers().find(worker => worker.url().endsWith('/background.mjs'));
    background ??= await context.waitForEvent('serviceworker', {predicate: worker => worker.url().endsWith('/background.mjs'), timeout: 20_000});
    const extensionId = new URL(background.url()).hostname;
    fixture = await context.newPage();
    await fixture.goto(origin);
    // Keep the fixture active. The popup is a real extension page in an inactive
    // tab, allowing its unchanged chrome.tabs.query handler to target the fixture.
    const popupPromise = context.waitForEvent('page', {timeout: 10_000});
    const popupTabId = await background.evaluate(async ({url, origin}) => {
      const [tab] = await chrome.tabs.query({url: origin + '/*'});
      await chrome.tabs.update(tab.id, {active: true});
      await chrome.windows.update(tab.windowId, {focused: true});
      return (await chrome.tabs.create({url, active: false})).id;
    }, {url: `chrome-extension://${extensionId}/popup.html`, origin});
    const popup = await popupPromise;
    assert(popup, 'Popup tab was not created');
    await popup.waitForURL(`chrome-extension://${extensionId}/popup.html`);
    await popup.locator('#site-name').filter({hasText: '127.0.0.1'}).waitFor();
    evidence.defaultSettings = {enabled: await popup.locator('#enabled').isChecked(), useAI: await popup.locator('#use-ai').isChecked()};
    assert.deepEqual(evidence.defaultSettings, {enabled: true, useAI: false}, 'Fresh install must keep AI opt-in');
    await popup.locator('#enable-site').click();
    await popup.getByRole('status').filter({hasText: 'Ready.'}).waitFor();
    const registered = await background.evaluate(() => chrome.scripting.getRegisteredContentScripts());
    assert.equal(registered.length, 1, 'Real popup handler must register the site');
    evidence.popupEnableHandler = true;
    evidence.popupTabId = popupTabId;

    async function setting(name, value) {
      const locator = popup.locator(name === 'useAI' ? '#use-ai' : '#enabled');
      if (await locator.isChecked() === value) return; // An unchanged default needs no storage write.
      await locator.setChecked(value);
      await background.evaluate(async ({name, value}) => {
        const end = Date.now() + 5000;
        while (Date.now() < end) {
          if ((await chrome.storage.local.get(name))[name] === value) return;
          await new Promise(ready => setTimeout(ready, 50));
        }
        throw new Error('Popup setting was not persisted');
      }, {name, value});
    }
    async function text(field) { return fields.indexOf(field) < 4 ? fixture.locator('#' + field).inputValue() : fixture.locator('#' + field).textContent(); }
    async function clickPanel(selector, mutate = null) {
      await waitForPanel(context, fixture, value => value !== null && !value.includes('Checking locally'), 'completed analysis');
      const node = await shadowNode(context, fixture, selector);
      assert(node, `Expected panel ${selector}`);
      try {
        await node.session.send('Runtime.callFunctionOn', {objectId: node.objectId,
          functionDeclaration: mutate === null ? 'function() { this.click(); }' : 'function(text) { document.querySelector("#draft").value = text; this.click(); }', arguments: mutate === null ? [] : [{value: mutate}]});
      } finally { await node.session.detach(); }
    }
    async function executeStep(item) {
      const {action, field, value, expected} = item;
      if (action === 'popup') await setting(field, value);
      else if (action === 'focus') await fixture.locator('#' + field).focus();
      else if (action === 'fill') await fixture.locator('#' + field).fill(value);
      else if (action === 'accept' || action === 'dismiss' || action === 'close') await clickPanel('button.' + action);
      else if (action === 'staleAccept') { await clickPanel('button.accept', value); assert.equal(await text(field), expected, 'Old suggestion must not overwrite changed text'); }
      else if (action === 'assertText') assert.equal(await text(field), expected);
      else if (action === 'assertPanel') await waitForPanel(context, fixture, panel => expected === 'hidden' ? panel === null : panel !== null && !panel.includes('Checking locally'), expected);
      else if (action === 'assertBackend') await waitForPanel(context, fixture, panel => panel?.includes(expected === 'ai' ? 'Local AI' : 'Local rules'), expected + ' backend');
      return {action, ...(field ? {field, text: fields.includes(field) ? await text(field) : null} : {}), panel: await panelText(context, fixture)};
    }
    for (const [kind, scenarios] of [['mandatory', mandatoryScenarios], ['luna', plan?.scenarios ?? []]]) {
      for (const scenario of scenarios) {
        const result = {kind, id: scenario.id, passed: false, actions: []};
        evidence.scenarios.push(result);
        await setting('enabled', true); await setting('useAI', false);
        await fixture.reload();
        const beforeRich = await fixture.locator('#rich').innerHTML();
        for (const item of scenario.steps) result.actions.push(await executeStep(item));
        assert.equal(await fixture.locator('#rich').innerHTML(), beforeRich, 'Unsupported rich DOM must remain intact');
        if (scenario.id === 'rich-preservation') assert.match(await panelText(context, fixture), /rich editor is not supported/u);
        const filename = `${kind}-${scenario.id}.png`;
        await fixture.screenshot({path: join(directory, filename), fullPage: true});
        evidence.screenshots.push(filename);
        result.passed = true;
      }
    }
    assert.deepEqual(evidence.blockedRequests, [], 'Extension must make no requests outside bundled assets and fictional fixture');
    assert.deepEqual(evidence.networkViolations, [], 'Browser observed a request outside the allowed fixture');
    assert.deepEqual(evidence.errors, [], 'No uncaught browser errors');
    assert.equal(hash(await readFile(shippingPath)), evidence.manifestHash, 'Shipping manifest must stay unchanged');
    assert.deepEqual(await sourceStamp(root), stamp, 'Source changed during browser verification');
    evidence.passed = true;
  } catch (error) {
    evidence.errors.push(error instanceof Error ? error.message : 'Extension driver failed');
    if (fixture && !fixture.isClosed()) {
      evidence.failurePanel = await panelText(context, fixture).catch(() => null);
      await fixture.screenshot({path: join(directory, 'failure.png'), fullPage: true}).then(() => evidence.screenshots.push('failure.png')).catch(() => {});
    }
  }
  finally {
    clearTimeout(timeout);
    try {
      if (context) {
        await context.tracing.stop({path: join(directory, 'trace.zip')}).catch(() => {});
        await context.close().catch(() => { evidence.passed = false; evidence.errors.push('Browser cleanup failed'); });
      }
    } finally {
      if (server?.listening) await new Promise(closed => { server.close(closed); for (const socket of sockets) socket.destroy(); });
      await rm(temporary, {recursive: true, force: true});
      await jsonFile(join(directory, 'browser.json'), evidence);
      process.off('SIGINT', onSignal);
      process.off('SIGTERM', onSignal);
    }
  }
  return evidence;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const options = {};
  while (args.length) {
    const flag = args.shift();
    if (!['--root', '--directory', '--plan', '--run-id'].includes(flag) || !args[0]) throw new Error('Invalid extension driver arguments');
    options[flag] = args.shift();
  }
  const root = resolve(options['--root'] ?? fileURLToPath(new URL('../', import.meta.url)));
  const directory = resolve(options['--directory'] ?? join(root, 'artifacts/agents', `extension-${runId()}`));
  const plan = options['--plan'] ? JSON.parse(await readFile(options['--plan'], 'utf8')) : null;
  const lock = await acquireLock(root);
  try {
    const evidence = await runExtensionScenarios({root, directory, plan, id: options['--run-id'] ?? runId()});
    console.log(JSON.stringify({passed: evidence.passed, directory, errors: evidence.errors}));
    if (!evidence.passed) process.exitCode = evidence.interrupted ? 130 : 1;
  } finally { await lock.release(); }
}
