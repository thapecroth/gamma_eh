import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { lstat, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const sha256 = value => typeof value === 'string' && /^[a-f0-9]{64}$/u.test(value);
const git = (root, ...args) => execFileSync('git', args, {cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']}).trim();
const sourcePaths = ['apps', 'packages', 'licenses', 'models', 'scripts', 'package.json', 'package-lock.json', 'LICENSE', 'NOTICE',
  'README.md', 'docs/privacy.md', 'docs/chrome-web-store.md', 'docs/support.md', 'docs/launch-scope.md', 'docs/playground.md'];
const normalizedVersions = new Set(['package.json', 'package-lock.json', 'apps/extension/manifest.json']);
export const readinessChecks = {
  quality: ['natural-development', 'natural-test', 'independent-reference', 'frozen-test', 'licensed-populations', 'meaning-preservation'],
  'editing-safety': ['stale-text', 'utf16', 'accept-all', 'undo-redo', 'ime', 'caret-selection', 'sensitive-fields', 'typing-navigation', 'long-text'],
  platforms: [],
  distribution: ['chrome-store-install', 'chrome-store-update', 'settings-preserved', 'site-access', 'uninstall', 'web-production-assets', 'web-recovery'],
  privacy: ['no-draft-network', 'no-draft-storage', 'no-draft-logs', 'bundled-code', 'permissions', 'policy'],
  accessibility: ['keyboard', 'screen-reader', 'contrast', 'zoom', 'responsive', 'status-announcements'],
  pilot: ['representative-users', 'usefulness', 'false-suggestions', 'installation', 'repeat-use', 'consent-no-default-draft-collection'],
  operations: ['support-contact', 'onboarding', 'known-limitations', 'incident-owner', 'higher-version-hotfix', 'rollback-rehearsal', 'zero-critical-defects'],
};

export function releaseChannel(value = 'experimental') {
  if (!['experimental', 'stable'].includes(value)) throw new Error('Release channel must be experimental or stable.');
  return value;
}

// Evidence can be committed after testing the release. Bind the complete shipping
// source, not the evidence commit. Version-only release snapshots share this digest.
export function sourceIdentity(root, ref) {
  const commit = git(root, 'rev-parse', `${ref}^{commit}`);
  const files = git(root, 'ls-tree', '-r', '--name-only', commit, '--', ...sourcePaths).split('\n').filter(Boolean).sort();
  if (!files.includes('package.json') || !files.includes('models/browser/manifest.json')) throw new Error('Incomplete release source tree.');
  const digest = createHash('sha256');
  for (const filename of files) {
    const bytes = execFileSync('git', ['show', `${commit}:${filename}`], {cwd: root, maxBuffer: 64 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe']});
    let content = bytes;
    if (normalizedVersions.has(filename)) {
      const value = JSON.parse(bytes.toString());
      delete value.version;
      if (filename === 'package-lock.json') delete value.packages[''].version;
      content = Buffer.from(JSON.stringify(value));
    }
    digest.update(filename + '\0').update(content).update('\0');
  }
  const model = execFileSync('git', ['show', `${commit}:models/browser/manifest.json`], {cwd: root, stdio: ['ignore', 'pipe', 'pipe']});
  return {commit, sha256: digest.digest('hex'), model_manifest_sha256: hash(model)};
}

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}
const nonempty = value => typeof value === 'string' && value.trim().length > 0;
const validDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/u.test(value)
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString() === value.replace('Z', '.000Z');

// The checked-in schema also describes pending records. This function enforces
// promotion requirements and receipt contents without a downloaded validator.
export async function validateReadiness(record, {identity, tag, assets, readEvidence, now = new Date()} = {}) {
  requireValue(record?.schema_version === 1, 'Unsupported GA readiness schema.');
  requireValue(record.status === 'ready', 'GA readiness is pending or blocked.');
  requireValue(/^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.test(tag ?? '') && record.release?.tag === tag, 'GA evidence must name the exact release tag.');
  requireValue(nonempty(record.owner) && nonempty(record.approved_by), 'GA requires named release ownership and approval.');
  requireValue(validDate(record.approved_at) && new Date(record.approved_at) <= now, 'GA approval date is missing or invalid.');
  requireValue(record.release.source_sha256 === identity.sha256 && record.release.model_manifest_sha256 === identity.model_manifest_sha256, 'GA evidence is stale: release source or model changed.');
  requireValue(/^[a-f0-9]{40}$/u.test(record.release.source_commit ?? ''), 'GA requires a source commit.');
  requireValue(record.scope?.surfaces?.length === 2 && record.scope.surfaces.includes('chrome') && record.scope.surfaces.includes('web'), 'GA scope must include Chrome and the web editor.');
  requireValue(['experimental', 'ga'].includes(record.scope.local_ai), 'Local AI scope must be explicit.');
  requireValue(Array.isArray(record.scope.platforms) && record.scope.platforms.length > 0
    && new Set(record.scope.platforms).size === record.scope.platforms.length
    && record.scope.platforms.every(os => ['windows', 'macos', 'linux'].includes(os)), 'GA requires a declared device matrix.');
  requireValue(record.policy?.approved === true && Number.isSafeInteger(record.policy.max_evidence_age_days)
    && record.policy.max_evidence_age_days > 0 && record.policy.max_evidence_age_days <= 90, 'GA evidence policy must be approved and expire within 90 days.');
  requireValue(Number.isFinite(record.policy.min_edit_precision) && record.policy.min_edit_precision >= 0.95 && record.policy.min_edit_precision <= 1
    && Number.isFinite(record.policy.max_clean_changed_rate) && record.policy.max_clean_changed_rate >= 0 && record.policy.max_clean_changed_rate <= 0.02
    && Number.isSafeInteger(record.policy.min_predicted_edits) && record.policy.min_predicted_edits >= 25, 'GA quality policy is weaker than the published precision/clean-text/support gates.');
  const expectedAssets = ['chrome', 'firefox', 'web'];
  requireValue(expectedAssets.every(kind => sha256(record.release.assets?.[kind]) && record.release.assets[kind] === assets?.[kind]), 'GA evidence must match every immutable release archive.');
  requireValue(record.gates && Object.keys(record.gates).length === Object.keys(readinessChecks).length, 'GA gate set is incomplete.');
  for (const [gate, checks] of Object.entries(readinessChecks)) {
    const item = record.gates[gate];
    requireValue(item?.status === 'passed' && nonempty(item.owner) && sha256(item.sha256)
      && /^docs\/ga-evidence\/[a-zA-Z0-9_-]+\.json$/u.test(item.evidence ?? ''), `GA gate ${gate} needs a reviewed evidence receipt.`);
    const bytes = await readEvidence(item.evidence);
    requireValue(hash(bytes) === item.sha256, `GA gate ${gate} receipt hash mismatch.`);
    const receipt = JSON.parse(bytes.toString());
    requireValue(receipt.schema_version === 1 && receipt.gate === gate && nonempty(receipt.reviewed_by), `GA gate ${gate} receipt is not reviewed.`);
    requireValue(receipt.tag === tag && receipt.source_sha256 === identity.sha256 && receipt.model_manifest_sha256 === identity.model_manifest_sha256
      && expectedAssets.every(kind => receipt.assets?.[kind] === assets[kind]), `GA gate ${gate} receipt belongs to different assets.`);
    requireValue(validDate(receipt.performed_at) && new Date(receipt.performed_at) <= new Date(record.approved_at)
      && (now - new Date(receipt.performed_at)) / 86400000 <= record.policy.max_evidence_age_days, `GA gate ${gate} receipt is stale or future-dated.`);
    const requiredChecks = gate === 'platforms' ? record.scope.platforms.flatMap(os => [`${os}-cpu`, `${os}-webgpu`]) : checks;
    requireValue(Array.isArray(receipt.checks) && receipt.checks.every(check => nonempty(check.id))
      && new Set(receipt.checks.map(check => check.id)).size === receipt.checks.length
      && requiredChecks.every(id => receipt.checks.some(check => check.id === id && check.passed === true && nonempty(check.notes))), `GA gate ${gate} has missing, duplicate or failed checks.`);
    if (gate === 'quality') {
      for (const split of ['development', 'test']) {
        for (const mode of ['rules', 'combined']) {
          const metrics = receipt.metrics?.[split]?.[mode];
          requireValue(Number.isSafeInteger(metrics?.sentences) && metrics.sentences > 0 && Number.isSafeInteger(metrics.clean_sentences) && metrics.clean_sentences > 0 && metrics.clean_sentences <= metrics.sentences
            && Number.isSafeInteger(metrics.predicted_edits) && metrics.predicted_edits >= 0 && Number.isSafeInteger(metrics.meaning_changing_errors) && metrics.meaning_changing_errors === 0
            && Number.isFinite(metrics.edit_precision) && metrics.edit_precision >= 0 && metrics.edit_precision <= 1
            && Number.isFinite(metrics.clean_changed_rate) && metrics.clean_changed_rate >= 0 && metrics.clean_changed_rate <= 1, `GA quality needs natural ${split}/${mode} metrics and zero meaning-changing errors.`);
          if (mode === 'rules' || record.scope.local_ai === 'ga') {
            requireValue(metrics.edit_precision >= record.policy.min_edit_precision && metrics.clean_changed_rate <= record.policy.max_clean_changed_rate
              && metrics.predicted_edits >= record.policy.min_predicted_edits, `GA quality ${split}/${mode} misses the declared targets.`);
          }
        }
      }
      requireValue(nonempty(receipt.population) && nonempty(receipt.methodology) && nonempty(receipt.experimental_ai_limitations), 'GA quality requires population, methodology, and optional-AI limitations.');
    }
    if (gate === 'platforms') {
      requireValue(Array.isArray(receipt.devices) && record.scope.platforms.every(os => receipt.devices.some(device => device.os === os
        && device.webgpu_adapter === 'physical' && nonempty(device.cpu) && nonempty(device.gpu) && nonempty(device.browser)
        && device.warm_p50_ms <= device.warm_p95_ms
        && ['cold_start_ms', 'warm_p50_ms', 'warm_p95_ms', 'peak_memory_mb'].every(key => Number.isFinite(device[key]) && device[key] > 0))), 'GA requires physical-device measurements, not SwiftShader checks.');
      requireValue(receipt.performance_budgets_passed === true && nonempty(receipt.performance_budgets), 'GA device performance budgets need review.');
    }
    if (gate === 'distribution') requireValue(/^https:\/\/chromewebstore\.google\.com\/detail\/[a-zA-Z0-9/_-]+$/u.test(receipt.store_url ?? '')
      && /^https:\/\/[^\s?@#]+$/u.test(receipt.web_url ?? ''), 'GA requires verified store and HTTPS web installation URLs.');
    if (gate === 'pilot') requireValue(Number.isSafeInteger(receipt.participants) && receipt.participants > 0 && nonempty(receipt.acceptance_criteria)
      && receipt.acceptance_criteria_met === true, 'GA needs a completed representative pilot and reviewed acceptance criteria.');
  }
  return {tag, source_sha256: identity.sha256, model_manifest_sha256: identity.model_manifest_sha256, assets, approved_by: record.approved_by};
}

export async function verifyStableRelease(root, {tag, assetDirectory, recordPath = 'docs/ga-readiness.json', now} = {}) {
  const record = JSON.parse(await readFile(join(root, recordPath), 'utf8'));
  // Resolve only a tag, never a caller-provided git revision expression.
  requireValue(/^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.test(tag ?? ''), 'Expected a canonical release tag.');
  const identity = sourceIdentity(root, `refs/tags/${tag}`);
  requireValue(/^[a-f0-9]{40}$/u.test(record.release?.source_commit ?? ''), 'GA source commit is missing.');
  git(root, 'merge-base', '--is-ancestor', record.release.source_commit, identity.commit);
  const approvedSource = sourceIdentity(root, record.release.source_commit);
  requireValue(approvedSource.sha256 === identity.sha256, 'GA reviewed source differs from the tagged release.');
  const assets = {};
  for (const kind of ['chrome', 'firefox', 'web']) {
    const filename = join(assetDirectory, `gamma-eh-${kind}-${tag}.zip`);
    requireValue((await lstat(filename)).isFile() && !(await lstat(filename)).isSymbolicLink(), 'Expected ordinary release archives.');
    assets[kind] = hash(await readFile(filename));
  }
  return validateReadiness(record, {tag, identity, assets, now, readEvidence: async filename => {
    const info = await lstat(join(root, filename));
    requireValue(info.isFile() && !info.isSymbolicLink(), 'Expected an ordinary GA evidence file.');
    return readFile(join(root, filename));
  }});
}

async function main() {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const [mode, tag, directory] = process.argv.slice(2);
  if (mode === 'identity' && tag) console.log(JSON.stringify(sourceIdentity(root, tag), null, 2));
  else if (mode === 'verify' && tag && directory) console.log(JSON.stringify(await verifyStableRelease(root, {tag, assetDirectory: resolve(directory)}), null, 2));
  else throw new Error('Usage: node scripts/ga-readiness.mjs identity REF | verify vX.Y.Z ASSET_DIRECTORY');
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
