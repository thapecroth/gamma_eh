import assert from 'node:assert/strict';
import {mkdir, readFile, realpath, symlink, writeFile} from 'node:fs/promises';
import {join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {acquireLock, changedProductPaths, git, jsonFile, runCommand, runId, sourceStamp} from './agent-runtime.mjs';
import {digest, frozenQuality, policy, profileTree, validateFrozenQuality} from './inference-analysis.mjs';

// Controlled delays exercise promotion plumbing; they are not an optimization.
const root = fileURLToPath(new URL('../', import.meta.url));
const {values} = parseArgs({options: {'quality-dir': {type: 'string', default: join(root, 'data/imported/jfleg-evaluation')}}});
const directory = join(root, 'artifacts/inference-promotion', runId());
const owned = [], reports = {};
const evidence = {schema: 1, passed: false, backend: 'wasm', purpose: 'promotion-plumbing-only', delayMs: 5,
  checks: {}, reports: {}};
const edits = [
  {path: 'packages/engine/src/index.ts', anchor: 'export async function analyzeText(text: string, options?: EngineOptions): Promise<AnalysisResult> {\n', indent: '  '},
  {path: 'packages/engine/src/jax-runtime.ts', anchor: '  async run(ids: number[]): Promise<{data: Float32Array; dims: number[]}> {\n', indent: '    '},
];
let lock, originalRoot, phase = 'setup', interrupted = false;
const onSignal = () => { interrupted = true; evidence.passed = false; process.exitCode = 1; };
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(signal, onSignal);

async function analyze(name, args, expectedCode, output = join(directory, name), timeoutMs = 300000) {
  assert(!interrupted, 'Promotion verification interrupted');
  phase = name;
  const log = join(directory, `${name}.private.log`);
  let code;
  try {
    ({code} = await runCommand(process.execPath, [join(root, 'scripts/analyze-inference.mjs'),
      '--iterations', '5', '--profile-ms', '500', '--output', output, ...args],
    {cwd: root, timeoutMs, log, environment: {GAMMA_AGENT_LOCK_TOKEN: lock.token}}));
  } catch (error) {
    if (!error.result || error.result.timedOut || error.result.interrupted) throw error;
    code = error.result.code;
  }
  const lines = (await readFile(log, 'utf8')).trimEnd().split('\n');
  let result;
  for (const line of lines.reverse()) {
    try { const parsed = JSON.parse(line); if (typeof parsed.report === 'string') { result = parsed; break; } }
    catch { /* progress and exception lines are not receipts */ }
  }
  assert(result, 'Child must produce a machine receipt');
  const bytes = await readFile(result.report), report = JSON.parse(bytes.toString('utf8'));
  evidence.observed = {phase, code, status: report.decision.status, reason: report.decision.reason};
  assert.equal(code, expectedCode, 'Unexpected actual child exit code; pressure cannot be waived');
  assert.equal(result.exitCode, code);
  assert.equal(report.decision.exitCode, code);
  assert.equal(report.decision.status, {0: 'accepted', 1: 'invalid', 2: 'rejected'}[code]);
  evidence.reports[name] = {path: relative(root, result.report), sha256: digest(bytes), exitCode: code, status: report.decision.status};
  reports[name] = report;
  console.log(JSON.stringify({phase: 'promotion-control', name, code, status: report.decision.status}));
  return report;
}

async function verifyComparison(control, expected) {
  const report = reports[control];
  assert.equal(report.backend, 'wasm');
  assert.equal(report.environment.underPressure, false, 'Pressure invalidates promotion verification');
  assert.deepEqual(report.policy, policy, 'Trusted gate policy changed');
  assert.equal(report.quality.complete, true);
  assert.deepEqual(report.quality.selection, ['dev']);
  assert.equal(report.quality.rows, frozenQuality.dev.rows);
  assert.deepEqual(report.quality.conditions, ['model:1', 'model:2', 'combined:1', 'combined:2']);
  assert.equal(report.options['include-holdout'], undefined);
  for (const name of ['baseline', 'anchor']) {
    for (const type of ['smoke', 'timing', 'logits', 'quality']) assert.equal(report.parity[name][type].passed, true);
    assert.equal(report.parity[name].quality.comparisons, frozenQuality.dev.rows * 4);
  }
  for (const [name, tree] of Object.entries(expected)) {
    const recorded = report.trees[name], current = await sourceStamp(tree);
    assert.equal(recorded.commit, evidence.commit);
    assert.equal(recorded.treeHash, current.treeHash, 'Receipt source must still be current');
    assert.deepEqual(recorded.dirtyPaths, (await changedProductPaths(tree)).sort());
    for (const key of ['modelAssets', 'guarded', 'lockSha256', 'dependencyInputsSha256']) assert.deepEqual(recorded[key], report.trees.candidate[key]);
    for (const [file, hash] of Object.entries(recorded.modelAssets)) assert.equal(digest(await readFile(join(tree, 'models/browser', file))), hash);
    assert.equal(recorded.probe.warmCounters.runtimeCalls, 0);
    assert.equal(recorded.probe.editedCounters.runtimeCalls, 1);
    assert.equal(report.trials[name].length, policy.minimumTrials);
    assert.deepEqual(Object.keys(report.profiles[name]).sort(), ['cold', 'edited', 'fresh', 'nearby', 'unchanged']);
    for (const [workload, sampling] of Object.entries(report.profiles[name])) {
      const cpu = JSON.parse(await readFile(join(directory, control, sampling.artifacts.cpu), 'utf8'));
      assert.equal(digest(JSON.stringify(cpu)), sampling.profileSha256);
      const actual = profileTree(cpu);
      assert.equal(actual.sampledUs, sampling.sampledUs);
      assert(cpu.samples.length > 0 && (workload === 'cold' || cpu.samples.length >= policy.minimumProfileSamples));
    }
  }
}

try {
  await mkdir(join(root, 'artifacts/inference-promotion'), {recursive: true});
  await mkdir(directory, {mode: 0o700});
  // Own the shared lock. Child analyses borrow its token and cannot release it.
  lock = await acquireLock(root, '');
  originalRoot = await sourceStamp(root);
  evidence.commit = originalRoot.commit;
  phase = 'quality-inputs';
  const quality = await realpath(values['quality-dir']);
  const manifestBytes = await readFile(join(quality, 'manifest.json')), devBytes = await readFile(join(quality, 'dev.jsonl'));
  validateFrozenQuality(JSON.parse(manifestBytes.toString('utf8')), devBytes, 'dev');
  const dependencies = await realpath(join(root, 'node_modules'));
  const fast = join(directory, 'fast'), slow = join(directory, 'slow');
  phase = 'fixture-creation';
  for (const tree of [fast, slow]) {
    await git(root, 'worktree', 'add', '--detach', tree, evidence.commit); owned.push(tree);
    await symlink(dependencies, join(tree, 'node_modules'), 'dir');
  }
  assert.deepEqual(await changedProductPaths(fast), []);
  phase = 'delay-injection';
  for (const {path, anchor, indent} of edits) {
    const original = await readFile(join(fast, path), 'utf8');
    assert.equal(original.split(anchor).length, 2, 'Fixture signature must occur exactly once');
    assert(!original.includes('INFERENCE_PROMOTION_E2E_DELAY'), 'Shipping code must not contain fixture delays');
    const delay = `${indent}// INFERENCE_PROMOTION_E2E_DELAY: disposable fixture only.\n${indent}await new Promise<void>(resolve => setTimeout(resolve, 5));\n`;
    const modified = original.replace(anchor, anchor + delay);
    await writeFile(join(slow, path), modified);
    assert.equal((await readFile(join(slow, path), 'utf8')).replace(delay, ''), original);
  }
  assert.deepEqual((await changedProductPaths(slow)).sort(), edits.map(edit => edit.path).sort(), 'Only declared delay fixtures may change');
  const positiveTrees = {candidate: fast, baseline: slow, anchor: slow};
  const negativeTrees = {candidate: slow, baseline: fast, anchor: fast};
  const args = trees => Object.entries(trees).flatMap(([name, tree]) => [`--${name}`, tree]);
  await analyze('positive', [...args(positiveTrees), '--quality-dir', quality], 0);
  await verifyComparison('positive', positiveTrees); evidence.checks.acceptedExit0 = true;
  await analyze('negative', [...args(negativeTrees), '--quality-dir', quality], 2);
  await verifyComparison('negative', negativeTrees); evidence.checks.rejectedExit2 = true;

  const modelPath = join(slow, 'models/browser/model.onnx'), model = await readFile(modelPath), corrupted = Buffer.from(model);
  corrupted[0] ^= 1;
  await writeFile(modelPath, corrupted);
  try {
    const invalid = await analyze('baseline-assets', args({candidate: fast, baseline: slow}), 1);
    assert.equal(invalid.decision.failure, 'Model hash mismatch: model.onnx');
    evidence.checks.ownBaselineAssetsRejected = true;
  } finally { await writeFile(modelPath, model); }
  const badQuality = join(directory, 'bad-quality'); await mkdir(badQuality, {mode: 0o700});
  await writeFile(join(badQuality, 'manifest.json'), manifestBytes, {mode: 0o600});
  await writeFile(join(badQuality, 'dev.jsonl'), devBytes.subarray(devBytes.indexOf(10) + 1), {mode: 0o600});
  const frozen = await analyze('frozen-quality', [...args(positiveTrees), '--quality-dir', badQuality], 1);
  assert.equal(frozen.decision.failure, 'Frozen quality hash mismatch'); evidence.checks.frozenQualityRejected = true;
  const positivePath = join(directory, 'positive/report.json'), previous = digest(await readFile(positivePath));
  await analyze('reused-output', args(positiveTrees), 1, join(directory, 'positive'), 30000);
  assert.equal(digest(await readFile(positivePath)), previous, 'Existing receipt must not be overwritten');
  evidence.checks.existingReceiptPreserved = true;

  const runtimePath = join(fast, 'packages/engine/src/jax-runtime.ts'), runtime = await readFile(runtimePath, 'utf8');
  let mutated = false, pending, mutationError;
  const poll = setInterval(() => {
    if (mutated || pending || mutationError) return;
    pending = (async () => {
      let text;
      try { text = await readFile(join(directory, 'source-drift.private.log'), 'utf8'); }
      catch (error) { if (error.code === 'ENOENT') return; throw error; }
      if (!text.includes('"phase":"unprofiled-timing"')) return;
      await writeFile(runtimePath, runtime + '\n// INFERENCE_PROMOTION_E2E_DRIFT: owned fixture only.\n'); mutated = true;
    })().catch(error => { mutationError = error; }).finally(() => { pending = undefined; });
  }, 100);
  try {
    const drift = await analyze('source-drift', ['--candidate', fast], 1);
    if (mutationError) throw mutationError;
    assert(mutated, 'Source drift must occur after compilation and timing');
    assert.equal(drift.decision.failure, 'Source changed during analysis'); evidence.checks.sourceDriftRejected = true;
    const changed = await sourceStamp(fast);
    assert.notEqual(changed.treeHash, reports.positive.trees.candidate.treeHash);
    evidence.staleReceipt = {receiptTreeHash: reports.positive.trees.candidate.treeHash, currentTreeHash: changed.treeHash};
    evidence.checks.staleReceiptHasDifferentSource = true;
  } finally { clearInterval(poll); await pending; await writeFile(runtimePath, runtime); }
  assert(!interrupted, 'Promotion verification interrupted'); evidence.passed = true;
} catch (error) {
  evidence.failure = {phase, reason: 'INFERENCE_PROMOTION_VERIFICATION_FAILED'}; process.exitCode = 1;
  await writeFile(join(directory, 'failure.private.log'), String(error.stack ?? error), {mode: 0o600}).catch(() => {});
} finally {
  try {
    for (const tree of owned.reverse()) await git(root, 'worktree', 'remove', '--force', tree);
    if (originalRoot) assert.deepEqual(await sourceStamp(root), originalRoot, 'Shipping checkout changed');
    evidence.checks.ownedFixturesRemovedAndRootPreserved = true;
  } catch {
    evidence.passed = false; evidence.failure = {phase: 'cleanup', reason: 'OWNED_FIXTURE_CLEANUP_FAILED'}; process.exitCode = 1;
  }
  try { await lock?.release(); } catch {
    evidence.passed = false; evidence.failure = {phase: 'cleanup', reason: 'PARENT_LOCK_CLEANUP_FAILED'}; process.exitCode = 1;
  }
  if (interrupted) { evidence.passed = false; evidence.failure = {phase, reason: 'PROMOTION_VERIFICATION_INTERRUPTED'}; }
  await jsonFile(join(root, 'artifacts/inference-promotion-verification.json'), evidence);
  console.log(JSON.stringify(evidence));
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.off(signal, onSignal);
}
