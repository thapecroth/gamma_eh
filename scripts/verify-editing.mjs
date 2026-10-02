import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import {chromium} from '@playwright/test';
import {acquireLock, sourceStamp} from './agent-runtime.mjs';
import {browserArguments, findChromium} from './browser-environment.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const lock = await acquireLock(root);
const evidence = {scope: 'Fictional browser edit-integrity fixtures; no model or store installation', checks: [], passed: false};
let browser;
try {
  evidence.source = await sourceStamp(root);
  const bundle = await build({entryPoints: [join(root, 'apps/extension/src/editable.ts')], bundle: true, write: false,
    format: 'iife', globalName: 'gammaEditable', platform: 'browser', target: 'chrome116'});
  browser = await chromium.launch({executablePath: await findChromium(), headless: true, args: browserArguments()});
  const page = await browser.newPage();
  await page.setContent('<!doctype html><html lang="en"><title>Fictional editing fixture</title><textarea id="textarea" aria-label="Draft"></textarea><div id="plain" contenteditable="plaintext-only" aria-label="Plain draft"></div><textarea id="other" aria-label="Other draft">Other fictional writing.</textarea>');
  await page.addScriptTag({content: bundle.outputFiles[0].text});
  const source = 'She have a book.';
  const corrected = 'She has a book.';
  async function replace(id, options = {}) {
    return page.evaluate(({id, source, options}) => {
      const field = document.getElementById(id);
      if (options.before === 'redirect-focus') {
        field.addEventListener('focus', () => document.getElementById('other').focus(), {once: true});
      }
      if (options.before) field.addEventListener('beforeinput', event => {
        if (options.before === 'cancel') event.preventDefault();
        if (options.before === 'mutate') {
          if (field instanceof HTMLTextAreaElement) field.value = 'A newer fictional draft.';
          else field.textContent = 'A newer fictional draft.';
        }
        if (options.before === 'private') field.setAttribute('data-private', '');
        if (options.before === 'detach') field.remove();
      }, {once: true});
      const changed = gammaEditable.replaceText(field, source, {id: 'fictional', start: 4, end: 8, original: 'have', replacement: 'has',
        category: 'grammar', confidence: 1, source: 'rule', message: 'Fictional agreement correction.'});
      return {changed, text: gammaEditable.readText(field)};
    }, {id, source, options});
  }
  for (const id of ['textarea', 'plain']) {
    const field = page.locator('#' + id);
    await field.focus();
    await field.pressSequentially(source);
    assert.deepEqual(await replace(id), {changed: true, text: corrected});
    await field.press('Control+z');
    assert.equal(await field.inputValue().catch(() => field.textContent()), source, `${id}: native Undo must restore the accepted correction`);
    await field.press('Control+Shift+z');
    assert.equal(await field.inputValue().catch(() => field.textContent()), corrected, `${id}: native Redo must restore the correction`);
    evidence.checks.push(`${id}-native-undo-redo`);
    for (const before of ['cancel', 'mutate', 'private', 'detach', 'redirect-focus']) {
      await field.fill(source);
      if (before === 'redirect-focus') await page.locator('#other').focus();
      const result = await replace(id, {before});
      assert.equal(result.changed, false, `${id}: ${before} must reject the edit`);
      assert.equal(result.text, before === 'mutate' ? 'A newer fictional draft.' : source);
      assert.equal(await page.locator('#other').inputValue(), 'Other fictional writing.');
      if (before === 'private') await field.evaluate(element => element.removeAttribute('data-private'));
      if (before === 'detach') {
        await page.evaluate(id => {
          const element = document.createElement(id === 'textarea' ? 'textarea' : 'div');
          element.id = id;
          if (id === 'plain') element.contentEditable = 'plaintext-only';
          document.body.append(element);
        }, id);
      }
      evidence.checks.push(`${id}-beforeinput-${before}`);
    }
  }
  evidence.passed = true;
} finally {
  try {
    if (browser) await browser.close();
    await mkdir(join(root, 'artifacts'), {recursive: true});
    await writeFile(join(root, 'artifacts/editing-integrity.json'), JSON.stringify(evidence, null, 2) + '\n');
  } finally { await lock.release(); }
}
console.log(JSON.stringify(evidence, null, 2));
