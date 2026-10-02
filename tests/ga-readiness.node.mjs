import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { readinessChecks, releaseChannel, sourceIdentity, validateReadiness, verifyStableRelease } from '../scripts/ga-readiness.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const now = new Date('2026-10-01T12:00:00Z');

// Fictional receipts exercise the gate, never serve as product launch evidence.
function fixture() {
  const identity = {sha256: 'a'.repeat(64), model_manifest_sha256: 'b'.repeat(64)};
  const assets = {chrome: 'c'.repeat(64), firefox: 'd'.repeat(64), web: 'e'.repeat(64)};
  const record = {
    schema_version: 1, status: 'ready', owner: 'Fictional release owner', approved_by: 'Fictional reviewer', approved_at: '2026-10-01T11:00:00Z',
    scope: {surfaces: ['chrome', 'web'], platforms: ['linux'], local_ai: 'experimental'},
    policy: {approved: true, max_evidence_age_days: 90, min_edit_precision: 0.95, max_clean_changed_rate: 0.02, min_predicted_edits: 25},
    release: {tag: 'v0.1.2', source_commit: 'f'.repeat(40), source_sha256: identity.sha256, model_manifest_sha256: identity.model_manifest_sha256, assets: {...assets}}, gates: {},
  };
  const receipts = {};
  function update(gate, mutate) {
    const filename = `docs/ga-evidence/${gate}.json`;
    const receipt = JSON.parse(receipts[filename]);
    mutate(receipt);
    receipts[filename] = JSON.stringify(receipt);
    record.gates[gate].sha256 = hash(receipts[filename]);
  }
  for (const [gate, checks] of Object.entries(readinessChecks)) {
    const filename = `docs/ga-evidence/${gate}.json`;
    const required = gate === 'platforms' ? ['linux-cpu', 'linux-webgpu'] : checks;
    receipts[filename] = JSON.stringify({schema_version: 1, gate, tag: record.release.tag, source_sha256: identity.sha256,
      model_manifest_sha256: identity.model_manifest_sha256, assets, performed_at: '2026-09-30T10:00:00Z', reviewed_by: 'Fictional independent reviewer',
      checks: required.map(id => ({id, passed: true, notes: 'Fictional acceptance evidence for validator regression.'}))});
    record.gates[gate] = {status: 'passed', owner: 'Fictional owner', evidence: filename, sha256: hash(receipts[filename])};
  }
  update('quality', receipt => {
    const metrics = {sentences: 747, clean_sentences: 182, predicted_edits: 100, edit_precision: 0.97, clean_changed_rate: 0.01, meaning_changing_errors: 0};
    receipt.metrics = {development: {rules: {...metrics}, combined: {...metrics}}, test: {rules: {...metrics}, combined: {...metrics}}};
    receipt.population = 'Fictional natural references';
    receipt.methodology = 'Fictional independently reviewed natural development/test scoring';
    receipt.experimental_ai_limitations = 'Optional AI is experimental; no broad accuracy claim.';
  });
  update('platforms', receipt => {
    receipt.devices = [{os: 'linux', cpu: 'Fictional CPU', gpu: 'Fictional GPU', browser: 'Fictional Chrome 116', webgpu_adapter: 'physical', cold_start_ms: 200, warm_p50_ms: 5, warm_p95_ms: 50, peak_memory_mb: 100}];
    receipt.performance_budgets = 'Fictional approved startup/latency/memory targets';
    receipt.performance_budgets_passed = true;
  });
  update('distribution', receipt => {receipt.store_url = 'https://chromewebstore.google.com/detail/fictional/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'; receipt.web_url = 'https://fictional.example/editor/';});
  update('pilot', receipt => {receipt.participants = 12; receipt.acceptance_criteria = 'Fictional predeclared pilot criteria'; receipt.acceptance_criteria_met = true;});
  const options = {tag: record.release.tag, identity, assets, now, readEvidence: async filename => {
    if (!receipts[filename]) throw new Error('Missing receipt');
    return Buffer.from(receipts[filename]);
  }};
  return {record, receipts, options, update};
}

test('release channels default to experimental and reject misspellings', () => {
  assert.equal(releaseChannel(), 'experimental');
  assert.equal(releaseChannel('stable'), 'stable');
  assert.throws(() => releaseChannel('ga'), /channel/u);
});

test('readiness schema accepts canonical tags and receipt paths only', async () => {
  const schema = JSON.parse(await readFile(new URL('../docs/ga-readiness.schema.json', import.meta.url), 'utf8'));
  const tag = new RegExp(schema.properties.release.properties.tag.pattern, 'u');
  const receipt = new RegExp(schema.$defs.gate.properties.evidence.pattern, 'u');
  assert.equal(tag.test('v0.1.2'), true);
  assert.equal(tag.test('v0x1x2'), false);
  assert.equal(tag.test('v01.1.2'), false);
  assert.equal(receipt.test('docs/ga-evidence/privacy.json'), true);
  assert.equal(receipt.test('docs/ga-evidence/privacyXjson'), false);
  assert.equal(receipt.test('../privacy.json'), false);
});

test('ready evidence passes; the committed product record remains pending', async () => {
  const {record, options} = fixture();
  assert.equal((await validateReadiness(record, options)).tag, 'v0.1.2');
  const pending = JSON.parse(await readFile(new URL('../docs/ga-readiness.json', import.meta.url), 'utf8'));
  await assert.rejects(validateReadiness(pending, options), /pending or blocked/u);
});

test('rejects absent, unreviewed, tampered, stale, and future receipts', async () => {
  for (const [mutate, expected] of [
    [f => {delete f.record.gates.privacy;}, /incomplete/u],
    [f => {f.record.gates.privacy.status = 'pending';}, /privacy/u],
    [f => {f.record.gates.privacy.evidence = '../private.json';}, /privacy/u],
    [f => {delete f.receipts['docs/ga-evidence/privacy.json'];}, /Missing/u],
    [f => {f.receipts['docs/ga-evidence/privacy.json'] += ' ';}, /hash mismatch/u],
    [f => f.update('privacy', r => {r.reviewed_by = ''; }), /not reviewed/u],
    [f => f.update('privacy', r => {r.performed_at = '2026-01-01T10:00:00Z';}), /stale/u],
    [f => f.update('privacy', r => {r.performed_at = '2026-10-02T10:00:00Z';}), /future/u],
    [f => f.update('privacy', r => {r.checks[0].passed = false;}), /missing, duplicate or failed/u],
    [f => f.update('privacy', r => {r.checks.push({...r.checks[0], passed: false});}), /duplicate/u],
  ]) {
    const f = fixture(); mutate(f);
    await assert.rejects(validateReadiness(f.record, f.options), expected);
  }
});

test('rejects stale release sources/models, wrong tags/assets, and weaker quality targets', async () => {
  for (const [mutate, expected] of [
    [f => {f.record.release.source_sha256 = '0'.repeat(64);}, /source or model/u],
    [f => {f.record.release.model_manifest_sha256 = '0'.repeat(64);}, /source or model/u],
    [f => {f.record.release.tag = 'v0.1.3';}, /exact release tag/u],
    [f => {f.record.release.assets.chrome = '0'.repeat(64);}, /archive/u],
    [f => {f.record.policy.approved = false;}, /policy/u],
    [f => {f.record.policy.max_evidence_age_days = '90';}, /policy/u],
    [f => {f.record.policy.min_predicted_edits = '25';}, /weaker/u],
    [f => {f.record.policy.min_edit_precision = '0.95';}, /weaker/u],
    [f => {f.record.policy.min_edit_precision = 0.7;}, /weaker/u],
    [f => f.update('quality', r => {r.metrics.development.rules.edit_precision = 0.719;}), /misses/u],
    [f => f.update('quality', r => {r.metrics.development.rules.clean_changed_rate = 38 / 216;}), /misses/u],
    [f => f.update('quality', r => {r.metrics.test.rules.predicted_edits = 8;}), /misses/u],
    [f => f.update('quality', r => {r.metrics.test.combined.meaning_changing_errors = 1;}), /meaning-changing/u],
    [f => f.update('quality', r => {r.metrics.test.rules.clean_sentences = r.metrics.test.rules.sentences + 1;}), /metrics/u],
    [f => f.update('privacy', r => {r.assets.chrome = '0'.repeat(64);}), /different assets/u],
  ]) {
    const f = fixture(); mutate(f);
    await assert.rejects(validateReadiness(f.record, f.options), expected);
  }
});

test('optional AI remains experimental; claiming AI GA enforces both shipping modes', async () => {
  const f = fixture();
  f.update('quality', r => {r.metrics.test.combined.edit_precision = 0.7;});
  await validateReadiness(f.record, f.options);
  f.record.scope.local_ai = 'ga';
  await assert.rejects(validateReadiness(f.record, f.options), /test\/combined/u);
});

test('rejects software GPU, absent store/update coverage, and incomplete pilot signoff', async () => {
  for (const [gate, mutate, expected] of [
    ['platforms', r => {r.devices[0].webgpu_adapter = 'swiftshader';}, /physical-device/u],
    ['platforms', r => {r.devices[0].warm_p50_ms = 100;}, /physical-device/u],
    ['distribution', r => {r.checks = r.checks.filter(c => c.id !== 'chrome-store-update');}, /missing, duplicate or failed/u],
    ['distribution', r => {r.store_url = 'https://example.com/';}, /installation URLs/u],
    ['pilot', r => {r.acceptance_criteria_met = false;}, /pilot/u],
  ]) {
    const f = fixture(); f.update(gate, mutate);
    await assert.rejects(validateReadiness(f.record, f.options), expected);
  }
});

test('trusted snapshot verifier rejects a tag that replaces its own provenance check', async t => {
  const root = await mkdtemp(join(tmpdir(), 'gamma-ga-trusted-snapshot-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const git = (...args) => execFileSync('git', args, {cwd: root, encoding: 'utf8', stdio: 'pipe'}).trim();
  const put = async (name, content) => {await mkdir(dirname(join(root, name)), {recursive: true}); await writeFile(join(root, name), content);};
  const verifier = await readFile(new URL('../scripts/verify-release-snapshot.mjs', import.meta.url));
  git('init', '-q'); git('config', 'user.name', 'Fictional Test'); git('config', 'user.email', 'test@example.invalid');
  const metadata = ['package.json', 'package-lock.json', 'apps/extension/manifest.json'];
  for (const name of metadata) await put(name, JSON.stringify({version: '0.1.1', ...(name === 'package-lock.json' ? {packages: {'': {version: '0.1.1'}}} : {})}));
  await put('scripts/verify-release-snapshot.mjs', verifier);
  git('add', '.'); git('commit', '-qm', 'fictional trusted main');
  await put('trusted.mjs', verifier);
  for (const name of metadata) {
    const value = JSON.parse(await readFile(join(root, name), 'utf8')); value.version = '0.1.2';
    if (value.packages) value.packages[''].version = '0.1.2';
    await put(name, JSON.stringify(value));
  }
  git('add', ...metadata); git('commit', '-qm', 'fictional version-only snapshot');
  execFileSync(process.execPath, ['trusted.mjs'], {cwd: root, stdio: 'pipe'});
  await put('scripts/verify-release-snapshot.mjs', 'process.exit(0);\n');
  git('add', 'scripts'); git('commit', '--amend', '-qm', 'fictional tampered snapshot');
  execFileSync(process.execPath, ['scripts/verify-release-snapshot.mjs'], {cwd: root, stdio: 'pipe'});
  assert.throws(() => execFileSync(process.execPath, ['trusted.mjs'], {cwd: root, stdio: 'pipe'}));
});

test('version-only snapshot shares reviewed source identity; runtime changes invalidate it', async t => {
  const root = await mkdtemp(join(tmpdir(), 'gamma-ga-source-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const git = (...args) => execFileSync('git', args, {cwd: root, encoding: 'utf8', stdio: 'pipe'}).trim();
  const put = async (name, content) => {await mkdir(dirname(join(root, name)), {recursive: true}); await writeFile(join(root, name), typeof content === 'string' ? content : JSON.stringify(content));};
  git('init', '-q'); git('config', 'user.name', 'Fictional Test'); git('config', 'user.email', 'test@example.invalid');
  await put('package.json', {version: '0.1.1', name: 'fictional'});
  await put('package-lock.json', {version: '0.1.1', packages: {'': {version: '0.1.1'}}});
  await put('apps/extension/manifest.json', {version: '0.1.1'});
  await put('models/browser/manifest.json', {files: {}});
  await put('packages/engine/index.js', 'Fictional shipping source');
  git('add', '.'); git('commit', '-qm', 'fixture source');
  const source = sourceIdentity(root, 'HEAD');
  for (const name of ['package.json', 'package-lock.json', 'apps/extension/manifest.json']) {
    const value = JSON.parse(await readFile(join(root, name), 'utf8')); value.version = '0.1.2';
    if (name === 'package-lock.json') value.packages[''].version = '0.1.2';
    await put(name, value);
  }
  git('add', '.'); git('commit', '-qm', 'chore(release): v0.1.2'); git('tag', 'v0.1.2');
  const snapshot = sourceIdentity(root, 'HEAD');
  assert.notEqual(snapshot.commit, source.commit); assert.equal(snapshot.sha256, source.sha256);
  const f = fixture();
  f.record.release.source_commit = source.commit; f.record.release.source_sha256 = source.sha256; f.record.release.model_manifest_sha256 = source.model_manifest_sha256;
  for (const kind of ['chrome', 'firefox', 'web']) {
    const bytes = Buffer.from(`Fictional ${kind} archive`); await put(`assets/gamma-eh-${kind}-v0.1.2.zip`, bytes.toString()); f.record.release.assets[kind] = hash(bytes);
  }
  for (const gate of Object.keys(readinessChecks)) {
    f.update(gate, r => {r.source_sha256 = source.sha256; r.model_manifest_sha256 = source.model_manifest_sha256; r.assets = f.record.release.assets;});
    await put(f.record.gates[gate].evidence, f.receipts[f.record.gates[gate].evidence]);
  }
  await put('docs/ga-readiness.json', f.record);
  assert.equal((await verifyStableRelease(root, {tag: 'v0.1.2', assetDirectory: join(root, 'assets'), now})).source_sha256, source.sha256);
  await put('packages/engine/index.js', 'Changed shipping source'); git('add', 'packages'); git('commit', '-qm', 'changed runtime'); git('tag', '-f', 'v0.1.2');
  assert.notEqual(sourceIdentity(root, 'HEAD').sha256, source.sha256);
  await assert.rejects(verifyStableRelease(root, {tag: 'v0.1.2', assetDirectory: join(root, 'assets'), now}), /reviewed source differs/u);
});
