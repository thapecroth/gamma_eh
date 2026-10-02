import assert from 'node:assert/strict';
import {access, mkdir, readFile, writeFile} from 'node:fs/promises';
import {join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {git, jsonFile, runCommand, runId} from './agent-runtime.mjs';
import {policy} from './inference-analysis.mjs';

// Real-browser smoke for the infrastructure, with fictional inputs only. A
// self-comparison without the frozen corpus must never promote a candidate.
const root = fileURLToPath(new URL('../', import.meta.url));
const directory = join(root, 'artifacts/inference-analysis', `verification-${runId()}`);
await mkdir(directory, {recursive: true, mode: 0o700});
const output = join(directory, 'analysis');
const evidence = {schema: 1, passed: false, backend: 'wasm', checks: {}};
try {
  let code = 0;
  try {
    await runCommand(process.execPath, [join(root, 'scripts/analyze-inference.mjs'),
      '--baseline', root, '--iterations', '5', '--profile-ms', '500', '--output', output],
    {cwd: root, timeoutMs: 600000, log: join(directory, 'analysis.private.log')});
  } catch (error) {
    if (error.result?.code !== 3 || error.result.timedOut || error.result.interrupted) throw error;
    code = error.result.code;
  }
  assert.equal(code, 3, 'Diagnostic self-comparison must be inconclusive');
  const report = JSON.parse(await readFile(join(output, 'report.json'), 'utf8'));
  assert.equal(report.decision.status, 'inconclusive');
  assert.equal(report.quality.complete, false);
  assert.equal(report.parity.baseline.smoke.passed, true);
  assert.equal(report.parity.baseline.timing.passed, true);
  assert.equal(report.parity.baseline.logits.passed, true);
  const html = await readFile(join(output, 'report.html'), 'utf8');
  assert(html.includes('<svg') && html.includes('including garbage collection'), 'Report must render CPU and elapsed evidence');
  for (const name of ['candidate', 'baseline']) {
    assert.equal(report.trials[name].length, policy.minimumTrials);
    assert.equal(report.trees[name].probe.warmCounters.runtimeCalls, 0);
    assert.equal(report.trees[name].probe.editedCounters.runtimeCalls, 1);
    assert.equal(report.trees[name].probe.coldRuntimeCalls, 24);
    for (const [workload, profile] of Object.entries(report.profiles[name])) {
      if (workload !== 'cold') assert(profile.samples >= policy.minimumProfileSamples);
      const cpu = JSON.parse(await readFile(join(output, profile.artifacts.cpu), 'utf8'));
      assert(cpu.samples.length > 0);
      const svg = await readFile(join(output, profile.artifacts.flamegraph), 'utf8');
      assert(svg.startsWith('<svg') && svg.includes('Sampled renderer CPU'));
      const trace = JSON.parse(await readFile(join(output, profile.artifacts.wall), 'utf8'));
      assert(trace.traceEvents.length > 0 && trace.traceEvents.every(event => event.ph === 'X' && event.dur >= 0));
    }
  }
  for (const [name, args, environment] of [
    ['invalid-options', ['--trials', '4'], {}],
    ['alternate-model', [], {GAMMA_MODEL_DIR: 'unapproved-model'}],
  ]) {
    let failed = false;
    try { await runCommand(process.execPath, [join(root, 'scripts/analyze-inference.mjs'), ...args],
      {cwd: root, timeoutMs: 30000, environment, log: join(directory, `${name}.private.log`)}); }
    catch (error) { if (error.result?.code !== 1 || error.result.timedOut || error.result.interrupted) throw error; failed = true; }
    assert(failed, `${name} must exit invalid`);
  }
  const lock = join(resolve(root, await git(root, 'rev-parse', '--git-common-dir')), 'gamma-agent.lock');
  async function verifyInterruption(name, phase, preload) {
    const log = join(directory, `${name}.private.log`);
    let signaled = false;
    const poll = setInterval(async () => {
      if (signaled) return;
      try {
        const text = await readFile(log, 'utf8');
        const markerLine = text.split('\n').find(line => line.includes(`"phase":"${phase}"`));
        if (!markerLine) return;
        const marker = JSON.parse(markerLine);
        const started = JSON.parse(text.split('\n').find(line => line.includes('"phase":"analysis-start"')));
        const owner = JSON.parse(await readFile(join(lock, 'owner.json'), 'utf8'));
        if (signaled || owner.pid !== started.pid || (preload && marker.pid !== started.pid)) return;
        process.kill(started.pid, 'SIGTERM'); signaled = true;
      } catch { /* wait for the owned analysis to acquire its lock and reach the marker */ }
    }, 100);
    try {
      let code = 0;
      try { await runCommand(process.execPath, [...(preload ? ['--import', preload] : []),
        join(root, 'scripts/analyze-inference.mjs'), '--baseline', root,
        '--iterations', '5', '--profile-ms', '500', '--output', join(directory, name)],
      {cwd: root, timeoutMs: preload ? 600000 : 60000, log}); }
      catch (error) { if (error.result?.code !== 1 || error.result.timedOut || error.result.interrupted) throw error; code = 1; }
      assert(signaled && code === 1, 'Interruption must invalidate the run');
      const interrupted = JSON.parse(await readFile(join(directory, name, 'report.json'), 'utf8'));
      assert.equal(interrupted.decision.status, 'invalid');
      if (preload) assert.equal(interrupted.decision.reason, 'analysis-interrupted');
      await assert.rejects(access(lock), {code: 'ENOENT'});
    } finally { clearInterval(poll); }
  }
  await verifyInterruption('interruption', 'unprofiled-timing');
  evidence.checks = {realBrowser: true, correctCacheWorkloads: true, exactOutputParity: true,
    realCpuSamples: true, flamegraphs: true, wallTraces: true, incompleteQualityCannotPromote: true,
    invalidOptionsRejected: true, alternateModelsRejected: true, interruptionInvalidatesAndReleasesLock: true,
    cleanupInterrupted: false};
  // Pause only the first browser.close(), reached by final analysis cleanup.
  // Context closes stay untouched, and a repeated close from onSignal delegates.
  const preload = join(directory, 'cleanup.preload.mjs');
  await writeFile(preload, `import {chromium} from '@playwright/test';
const launch = chromium.launch;
chromium.launch = async function (...args) {
  const browser = await launch.apply(this, args);
  const close = browser.close.bind(browser);
  let first = true;
  browser.close = async function (...args) {
    if (first) {
      first = false;
      await new Promise(resolve => {
        const finish = () => { clearTimeout(timer); process.off('SIGTERM', finish); resolve(); };
        const timer = setTimeout(finish, 5000);
        process.once('SIGTERM', finish);
        console.log(JSON.stringify({phase: 'browser-cleanup-await', pid: process.pid}));
      });
    }
    return close(...args);
  };
  return browser;
};
`, {mode: 0o600});
  await verifyInterruption('cleanup-interruption', 'browser-cleanup-await', preload);
  evidence.checks.cleanupInterrupted = true;
  evidence.report = relative(root, join(output, 'report.json'));
  evidence.passed = true;
} catch {
  evidence.failure = 'INFERENCE_ANALYSIS_VERIFICATION_FAILED';
  process.exitCode = 1;
} finally {
  await jsonFile(join(root, 'artifacts/inference-analysis-verification.json'), evidence);
  console.log(JSON.stringify(evidence));
}
