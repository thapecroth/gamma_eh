import assert from 'node:assert/strict';
import {mkdir, readFile} from 'node:fs/promises';
import {join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {jsonFile, runCommand, runId} from './agent-runtime.mjs';
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
  evidence.checks = {realBrowser: true, correctCacheWorkloads: true, exactOutputParity: true,
    realCpuSamples: true, flamegraphs: true, wallTraces: true, incompleteQualityCannotPromote: true};
  evidence.report = relative(root, join(output, 'report.json'));
  evidence.passed = true;
} catch {
  evidence.failure = 'INFERENCE_ANALYSIS_VERIFICATION_FAILED';
  process.exitCode = 1;
} finally {
  await jsonFile(join(root, 'artifacts/inference-analysis-verification.json'), evidence);
  console.log(JSON.stringify(evidence));
}
