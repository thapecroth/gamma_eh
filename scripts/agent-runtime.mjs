import { createHash, randomUUID } from 'node:crypto';
import { execFile, spawn } from 'node:child_process';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { hostname } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

const execute = promisify(execFile);
export const hash = value => createHash('sha256').update(typeof value === 'string' || Buffer.isBuffer(value) ? value : JSON.stringify(value)).digest('hex');
export const runId = () => `${new Date().toISOString().replace(/[:.]/gu, '-')}-${randomUUID().slice(0, 8)}`;
export async function jsonFile(path, value) { await writeFile(path, JSON.stringify(value, null, 2) + '\n', {mode: 0o600}); }
export async function git(root, ...args) { return (await execute('git', args, {cwd: root, maxBuffer: 10 * 1024 * 1024})).stdout.trim(); }

export async function changedProductPaths(root) {
  const paths = [];
  for (const args of [['diff', '--name-only', '-z'], ['diff', '--cached', '--name-only', '-z'], ['ls-files', '--others', '--exclude-standard', '-z']]) {
    const {stdout} = await execute('git', args, {cwd: root});
    paths.push(...stdout.split('\0').filter(Boolean));
  }
  return [...new Set(paths)];
}

export function productScopeAllowed(paths) {
  return paths.every(path => /^(apps\/|packages\/|tests\/|docs\/)/u.test(path) && !/^tests\/(agent-|extension-driver)/u.test(path));
}

export async function sourceStamp(root) {
  const commit = await git(root, 'rev-parse', 'HEAD');
  const {stdout} = await execute('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard'], {cwd: root, maxBuffer: 10 * 1024 * 1024});
  const digest = createHash('sha256');
  for (const path of [...new Set(stdout.split('\0').filter(Boolean))].sort()) {
    digest.update(path + '\0');
    try { digest.update(await readFile(join(root, path))); }
    catch (error) { if (error.code !== 'ENOENT') throw error; digest.update('<deleted>'); }
    digest.update('\0');
  }
  return {commit, treeHash: digest.digest('hex')};
}

export function boundedInteger(value, name, maximum) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1 || number > maximum) throw new Error(`${name} must be an integer from 1 to ${maximum}`);
  return number;
}

export async function acquireLock(root, token = process.env.GAMMA_AGENT_LOCK_TOKEN) {
  const common = resolve(root, await git(root, 'rev-parse', '--git-common-dir'));
  const directory = join(common, 'gamma-agent.lock');
  if (token) {
    const owner = JSON.parse(await readFile(join(directory, 'owner.json'), 'utf8'));
    if (owner.token !== token || owner.host !== hostname()) throw new Error('Invalid inherited harness lock');
    try { process.kill(owner.pid, 0); } catch { throw new Error('Harness lock owner is no longer running'); }
    return {token, release: async () => {}};
  }
  try { await mkdir(directory); }
  catch (error) { if (error.code === 'EEXIST') throw new Error(`Another harness owns ${directory}. Inspect its owner.json; remove only after confirming that run has stopped.`); throw error; }
  const owner = {token: randomUUID(), pid: process.pid, host: hostname(), startedAt: new Date().toISOString()};
  try { await jsonFile(join(directory, 'owner.json'), owner); }
  catch (error) { await rm(directory, {recursive: true, force: true}); throw error; }
  return {token: owner.token, release: async () => {
    const current = JSON.parse(await readFile(join(directory, 'owner.json'), 'utf8'));
    if (current.token === owner.token) await rm(directory, {recursive: true, force: true});
  }};
}

// Every child is a process group: interruption/timeout also stops descendants.
export async function runCommand(command, args, {cwd, timeoutMs = 600_000, log, input, environment = {}} = {}) {
  const {createWriteStream} = await import('node:fs');
  const stream = log ? createWriteStream(log, {mode: 0o600}) : null;
  return new Promise((resolveRun, rejectRun) => {
    const child = spawn(command, args, {cwd, env: {...process.env, ...environment}, detached: process.platform !== 'win32', stdio: ['pipe', 'pipe', 'pipe']});
    let timedOut = false;
    let interrupted = false;
    let forceTimer;
    let logError;
    const stop = () => {
      try { process.kill(process.platform === 'win32' ? child.pid : -child.pid, 'SIGTERM'); } catch { /* already stopped */ }
      forceTimer ??= setTimeout(() => {
        try { process.kill(process.platform === 'win32' ? child.pid : -child.pid, 'SIGKILL'); } catch { /* already stopped */ }
      }, 2000);
      forceTimer.unref();
    };
    const onSignal = () => { interrupted = true; stop(); };
    process.on('SIGINT', onSignal);
    process.on('SIGTERM', onSignal);
    const timer = setTimeout(() => { timedOut = true; stop(); }, timeoutMs);
    const cleanup = () => {
      clearTimeout(timer); clearTimeout(forceTimer);
      process.off('SIGINT', onSignal); process.off('SIGTERM', onSignal);
    };
    stream?.on('error', error => { logError = error; stop(); });
    child.stdout.on('data', chunk => stream?.write(chunk));
    child.stderr.on('data', chunk => stream?.write(chunk));
    child.stdin.on('error', () => {});
    child.on('error', error => { cleanup(); stream?.end(); rejectRun(error); });
    child.on('close', (code, signal) => {
      if (timedOut || interrupted || logError) {
        // The leader may exit on SIGTERM while a descendant ignores it.
        try { process.kill(process.platform === 'win32' ? child.pid : -child.pid, 'SIGKILL'); } catch { /* group stopped */ }
      }
      cleanup();
      const result = {command, args, code, signal, timedOut, interrupted};
      const finish = () => {
        if (logError) rejectRun(logError);
        else if (code === 0 && !timedOut && !interrupted) resolveRun(result);
        else rejectRun(Object.assign(new Error(timedOut ? `${command} timed out` : interrupted ? `${command} interrupted` : `${command} failed (${code ?? signal})`), {result}));
      };
      if (stream && !stream.destroyed) stream.end(finish); else finish();
    });
    child.stdin.end(input);
  });
}

export async function invokeCodex({root, directory, name, prompt, schema, model, effort = 'max', writable = false, images = [], timeoutMs = 300_000}) {
  const schemaPath = join(directory, `${name}.schema.json`);
  const outputPath = join(directory, `${name}.json`);
  await rm(outputPath, {force: true});
  await jsonFile(schemaPath, schema);
  const args = ['exec', '--ignore-user-config', '--ephemeral', '--model', model, '--sandbox', writable ? 'workspace-write' : 'read-only',
    '-c', `model_reasoning_effort="${effort}"`, '-c', 'approval_policy="never"', '--cd', root,
    '--output-schema', schemaPath, '--output-last-message', outputPath];
  for (const screenshot of images) args.push('--image', screenshot);
  args.push('-');
  try {
    await runCommand(process.env.GAMMA_CODEX_BIN ?? 'codex', args, {cwd: root, input: prompt, timeoutMs, log: join(directory, `${name}.private.log`)});
  } catch (error) { error.agentFailure = true; throw error; }
  // Never parse stdout: CLI emits progress and can echo the result twice.
  let result;
  try { result = JSON.parse(await readFile(outputPath, 'utf8')); }
  catch { throw Object.assign(new Error(`${name} did not produce a valid JSON result`), {agentFailure: true}); }
  return result;
}

export function validateReview(review, evidence) {
  if (!review || review.runId !== evidence.runId || review.evidenceHash !== hash(evidence)
    || !['pass', 'fail'].includes(review.verdict) || !Array.isArray(review.findings)
    || review.findings.some(item => !item || !['bug', 'limitation'].includes(item.kind) || typeof item.description !== 'string' || !item.description.trim())) throw new Error('Missing, malformed, or stale Luna review');
  if (review.verdict === 'pass' && (!evidence.passed || review.findings.some(item => item.kind === 'bug'))) throw new Error('Luna pass contradicts browser evidence or findings');
  if (review.verdict !== 'pass' || !evidence.passed) throw new Error('Extension E2E did not pass both machine and Luna gates');
  return review;
}

export function publicationAllowed(gates, changed) {
  const source = gates.check?.treeHash;
  return Boolean(changed && source && ['check', 'browser', 'agent'].every(name => gates[name]?.passed === true && gates[name].treeHash === source));
}

export const requiredScenarioIds = ['rules-and-popup', 'ai-opt-in', 'dismiss-preserves-text', 'protected-private', 'protected-payment', 'protected-optout', 'plain-editable', 'rich-preservation', 'utf16-offsets', 'stale-suggestion'];

export function validateReceipt(evidence, {id, stamp, plan}) {
  const expected = [...requiredScenarioIds.map(name => `mandatory:${name}`), ...plan.scenarios.map(row => `luna:${row.id}`)].sort();
  const actual = Array.isArray(evidence?.scenarios) ? evidence.scenarios.map(row => `${row.kind}:${row.id}`).sort() : null;
  if (!evidence || evidence.runId !== id || evidence.commit !== stamp.commit || evidence.treeHash !== stamp.treeHash || evidence.planHash !== hash(plan)
    || !Array.isArray(evidence.scenarios) || !evidence.scenarios.length || evidence.scenarios.some(row => row.passed !== true)
    || JSON.stringify(actual) !== JSON.stringify(expected) || evidence.errors?.length !== 0 || evidence.networkViolations?.length !== 0
    || evidence.blockedRequests?.length !== 0 || evidence.popupEnableHandler !== true || evidence.passed !== true) throw new Error('Failed or stale browser receipt');
  return evidence;
}
