import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { acquireLock, changedProductPaths, git, hash, productScopeAllowed, publicationAllowed, requiredScenarioIds, runCommand, sourceStamp, validateReceipt, validateReview } from '../scripts/agent-runtime.mjs';
import { parseOptions } from '../scripts/agent-harness.mjs';
import { step, validatePlan } from '../scripts/extension-driver.mjs';

const plan = {reason: 'Protect clean fictional writing', scenarios: [{id: 'clean-text', steps: [step('fill', 'draft', 'A clean sentence.'), step('assertText', 'draft', null, 'A clean sentence.')]}]};
const stamp = {commit: 'commit', treeHash: 'tree'};
function receipt() {
  return {runId: 'run', ...stamp, planHash: hash(plan), passed: true, popupEnableHandler: true, errors: [], blockedRequests: [], networkViolations: [],
    scenarios: [...requiredScenarioIds.map(id => ({kind: 'mandatory', id, passed: true})), {kind: 'luna', id: 'clean-text', passed: true}]};
}

test('rejects arbitrary code/navigation, malformed operations and assertion-free plans', () => {
  assert.equal(validatePlan(plan), plan);
  for (const action of ['evaluate', 'navigate', 'shell']) {
    assert.throws(() => validatePlan({reason: 'bad', scenarios: [{id: 'bad', steps: [step(action), step('assertPanel', null, null, 'hidden')]}]}), /Out-of-scope/u);
  }
  assert.throws(() => validatePlan({reason: 'bad', scenarios: [{id: '../escape', steps: plan.scenarios[0].steps}]}), /Malformed/u);
  assert.throws(() => validatePlan({reason: 'bad', scenarios: [{id: 'no-assert', steps: [step('focus', 'draft'), step('read', 'draft')]}]}), /objective assertion/u);
  assert.throws(() => validatePlan({reason: 'bad', scenarios: [{id: 'private-write', steps: [step('fill', 'private', 'anything'), step('assertPanel', null, null, 'hidden')]}]}), /Only public/u);
});

test('caps plans, retries and disallows accidental publication in verify-only mode', () => {
  assert.throws(() => parseOptions(['loop', '--max-iterations', '4']), /1 to 3/u);
  assert.throws(() => parseOptions(['loop', '--max-iterations', '0']), /1 to 3/u);
  assert.throws(() => parseOptions(['loop', '--verify-only', '--publish']), /cannot/u);
  assert.throws(() => parseOptions(['e2e', '--publish']), /Loop flags/u);
  assert.throws(() => parseOptions(['loop', '--unknown']), /Unknown/u);
  const tooMany = {reason: 'bad', scenarios: [0, 1, 2].map(index => ({id: `case-${index}`, steps: [...Array.from({length: 16}, () => step('read', 'draft')), step('assertText', 'draft', null, 'x')]}))};
  assert.throws(() => validatePlan(tooMany), /48 actions/u);
});

test('all current required and planned scenarios must pass; stale receipts never pass', () => {
  assert.equal(validateReceipt(receipt(), {id: 'run', stamp, plan}).passed, true);
  for (const key of ['runId', 'commit', 'treeHash', 'planHash']) {
    const evidence = receipt(); evidence[key] = 'stale';
    assert.throws(() => validateReceipt(evidence, {id: 'run', stamp, plan}), /stale/u);
  }
  for (const mutate of [e => e.scenarios.pop(), e => e.scenarios.shift(), e => { e.scenarios[0].passed = false; }, e => e.errors.push('error'), e => e.networkViolations.push({kind: 'outside-fixture'})]) {
    const evidence = receipt(); mutate(evidence);
    assert.throws(() => validateReceipt(evidence, {id: 'run', stamp, plan}), /Failed/u);
  }
});

test('Luna cannot waive a failed machine gate or return an old review', () => {
  const evidence = receipt();
  const review = {runId: 'run', evidenceHash: hash(evidence), verdict: 'pass', findings: []};
  assert.equal(validateReview(review, evidence), review);
  assert.throws(() => validateReview({...review, evidenceHash: 'old'}, evidence), /stale/u);
  assert.throws(() => validateReview({...review, findings: [{kind: 'bug', description: 'Observed corruption'}]}, evidence), /contradicts/u);
  const failed = {...evidence, passed: false};
  assert.throws(() => validateReview({...review, evidenceHash: hash(failed)}, failed), /contradicts/u);
  assert.throws(() => validateReview(null, evidence), /Missing/u);
});

test('publication requires changes and matching source evidence from every gate', () => {
  const gates = Object.fromEntries(['check', 'browser', 'agent'].map(name => [name, {passed: true, treeHash: 'same'}]));
  assert.equal(publicationAllowed(gates, true), true);
  assert.equal(publicationAllowed(gates, false), false);
  assert.equal(publicationAllowed({...gates, agent: {passed: true, treeHash: 'old'}}, true), false);
  assert.equal(publicationAllowed({...gates, browser: {passed: false, treeHash: 'same'}}, true), false);
  assert.equal(publicationAllowed({}, true), false);
});

async function fixture(operation) {
  const root = await mkdtemp(join(tmpdir(), 'gamma-harness-unit-'));
  try {
    await git(root, 'init', '--quiet');
    await writeFile(join(root, '.gitignore'), 'artifacts/\n');
    await writeFile(join(root, 'package.json'), '{}\n');
    await git(root, 'add', '.');
    await git(root, '-c', 'user.name=Harness Test', '-c', 'user.email=harness@example.invalid', '-c', 'core.hooksPath=/dev/null', 'commit', '--quiet', '-m', 'test: fixture');
    return await operation(root);
  } finally { await rm(root, {recursive: true, force: true}); }
}

test('staging cannot hide edits to protected gate configuration', () => fixture(async root => {
  await writeFile(join(root, 'package.json'), '{"scripts":{"check":"true"}}\n');
  await git(root, 'add', 'package.json');
  assert.deepEqual(await changedProductPaths(root), ['package.json']);
  assert.equal(productScopeAllowed(await changedProductPaths(root)), false);
  assert.equal(productScopeAllowed(['tests/agent-harness.node.mjs']), false);
  assert.equal(productScopeAllowed(['apps/extension/src/content.ts', 'tests/engine.test.ts']), true);
}));

test('source stamps cover unstaged, staged and new source, excluding artifacts', () => fixture(async root => {
  const initial = await sourceStamp(root);
  await mkdir(join(root, 'artifacts'));
  await writeFile(join(root, 'artifacts', 'old.json'), '{}');
  assert.deepEqual(await sourceStamp(root), initial);
  await writeFile(join(root, 'package.json'), '{"changed":true}');
  const dirty = await sourceStamp(root);
  assert.notEqual(dirty.treeHash, initial.treeHash);
  await git(root, 'add', 'package.json');
  assert.deepEqual(await sourceStamp(root), dirty);
  await writeFile(join(root, 'new.txt'), 'new source');
  assert.notEqual((await sourceStamp(root)).treeHash, dirty.treeHash);
}));

test('shared locks reject another run and validate inherited ownership', () => fixture(async root => {
  const lock = await acquireLock(root, null);
  try {
    await assert.rejects(acquireLock(root, null), /Another harness/u);
    await assert.rejects(acquireLock(root, 'invalid-token'), /Invalid/u);
    const inherited = await acquireLock(root, lock.token);
    await inherited.release();
    await assert.rejects(acquireLock(root, null), /Another harness/u);
  } finally { await lock.release(); }
  const next = await acquireLock(root, null); await next.release();
}));

test('nonzero processes and log failures reject before a retry can start', async () => {
  await assert.rejects(runCommand(process.execPath, ['-e', 'process.exit(7)']), error => error.result?.code === 7);
  await assert.rejects(runCommand(process.execPath, ['-e', 'process.exit(130)']), error => error.result?.interrupted === true);
  const directory = await mkdtemp(join(tmpdir(), 'gamma-process-unit-'));
  try {
    await assert.rejects(runCommand(process.execPath, ['-e', 'setInterval(()=>{},1000)'], {log: join(directory, 'missing', 'log.txt'), timeoutMs: 3000}), error => error.code === 'ENOENT');
  } finally { await rm(directory, {recursive: true, force: true}); }
});

test('timeouts stop detached descendants even when the leader exits first', {skip: process.platform !== 'linux'}, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'gamma-process-tree-'));
  const pidPath = join(directory, 'pid');
  let pid;
  try {
    const program = `const {spawn}=require('node:child_process'); const {writeFileSync}=require('node:fs'); const child=spawn(process.execPath,['-e','process.on("SIGTERM",()=>{});setInterval(()=>{},1000)'],{stdio:'ignore',detached:true});writeFileSync(${JSON.stringify(pidPath)},String(child.pid));setInterval(()=>{},1000);`;
    await assert.rejects(runCommand(process.execPath, ['-e', program], {timeoutMs: 700}), error => error.result?.timedOut === true);
    pid = Number(await readFile(pidPath, 'utf8'));
    const end = Date.now() + 2000;
    while (Date.now() < end) {
      try {
        process.kill(pid, 0);
        if (process.platform === 'linux' && /\) Z /u.test(await readFile(`/proc/${pid}/stat`, 'utf8'))) return;
      } catch (error) { if (error.code === 'ESRCH' || error.code === 'ENOENT') return; throw error; }
      await new Promise(ready => setTimeout(ready, 25));
    }
    assert.fail('Timed-out descendant remained alive');
  } finally {
    if (pid) try { process.kill(pid, 'SIGKILL'); } catch { /* already stopped */ }
    await rm(directory, {recursive: true, force: true});
  }
});

test('immediate launcher exit cannot orphan a detached process between samples', {skip: process.platform !== 'linux'}, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'gamma-fast-process-'));
  let pid;
  try {
    const pidPath = join(directory, 'pid');
    const program = `const {spawn}=require('node:child_process'); const {writeFileSync}=require('node:fs'); const child=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore',detached:true});writeFileSync(${JSON.stringify(pidPath)},String(child.pid));process.exit(0);`;
    await runCommand(process.execPath, ['-e', program]);
    pid = Number(await readFile(pidPath, 'utf8'));
    try {
      const stat = await readFile(`/proc/${pid}/stat`, 'utf8');
      assert.match(stat, /\) [ZX] /u, 'Detached orphan must be stopped before the command resolves');
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  } finally {
    if (pid) try { process.kill(pid, 'SIGKILL'); } catch { /* already stopped */ }
    await rm(directory, {recursive: true, force: true});
  }
});
