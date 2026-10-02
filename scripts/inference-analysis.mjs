import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

export const digest = bytes => createHash('sha256').update(bytes).digest('hex');
export const policy = Object.freeze({version: 1, workloadVersion: 1, seed: 143271,
  weights: {unchanged: 0.1, edited: 0.6, fresh: 0.3}, minimumSpeedup: 0.05,
  maximumRegression: 0.15, maximumP95Regression: 0.25, confidenceTolerance: 1e-4, logitTolerance: 5e-4,
  bootstrapSamples: 10000, minimumTrials: 5, minimumProfileSamples: 100,
  hostPressure: {loadCpuFraction: 0.5, memoryFullAvg10: 1, ioFullAvg10: 5}, browserOperationTimeoutMs: 300000});
export const frozenQuality = Object.freeze({revision: 'ee06ff806a208aba815ac45313f4e750a48330a5',
  dev: {rows: 754, sha256: 'b5581c7aa7dfcbacdf39b41256e3b01e13e85ace621baa766ac2b3868723f3a5'},
  test: {rows: 747, sha256: '22504390dc7921e1a4e85d77847b89953965ce385d22b428efec764e2abe6760'}});

export function percentile(values, quantile) {
  assert(values.length && values.every(Number.isFinite), 'Invalid measurements');
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * quantile))];
}

export function compareOutputs(before, after) {
  assert(Array.isArray(before) && before.length > 0 && before.length === after?.length, 'Missing parity population');
  assert(before.every(row => typeof row.id === 'string') && new Set(before.map(row => row.id)).size === before.length, 'Invalid parity IDs');
  let maxConfidenceDifference = 0;
  for (let index = 0; index < before.length; index++) {
    const left = before[index], right = after[index];
    assert.equal(left.id, right.id, 'Parity input/order mismatch');
    assert.equal(left.corrected, right.corrected, 'Correction parity failed');
    assert(Array.isArray(left.suggestions) && Array.isArray(right.suggestions), 'Missing edits');
    assert.equal(left.suggestions.length, right.suggestions.length, 'Edit count parity failed');
    for (let edit = 0; edit < left.suggestions.length; edit++) {
      const {confidence: a, ...aEdit} = left.suggestions[edit];
      const {confidence: b, ...bEdit} = right.suggestions[edit];
      assert.deepEqual(aEdit, bEdit, 'Edit field parity failed');
      assert(Number.isFinite(a) && Number.isFinite(b), 'Nonfinite confidence');
      const difference = Math.abs(a - b);
      assert(difference < policy.confidenceTolerance, 'Confidence parity failed');
      maxConfidenceDifference = Math.max(maxConfidenceDifference, difference);
    }
  }
  return {passed: true, comparisons: before.length, maxConfidenceDifference};
}

export function compareLogits(before, after) {
  assert(before?.length > 0 && before.length === after?.length, 'Missing logit population');
  let maximumDifference = 0;
  for (let index = 0; index < before.length; index++) {
    const a = before[index], b = after[index];
    assert.equal(a.length, b.length);
    assert.deepEqual(a.dims, b.dims, 'Logit shape mismatch');
    assert(a.dims.length === 3 && a.dims[0] === 1 && a.dims[1] === a.length && a.dims[2] > 0, 'Invalid logits shape');
    assert(a.values.length === a.length * a.dims[2] && b.values.length === a.values.length, 'Truncated logits');
    const labels = a.dims[2];
    for (let start = 0; start < a.values.length; start += labels) {
      let bestA = 0, bestB = 0;
      for (let label = 0; label < labels; label++) {
        assert(Number.isFinite(a.values[start + label]) && Number.isFinite(b.values[start + label]), 'Nonfinite logits');
        maximumDifference = Math.max(maximumDifference, Math.abs(a.values[start + label] - b.values[start + label]));
        if (a.values[start + label] > a.values[start + bestA]) bestA = label;
        if (b.values[start + label] > b.values[start + bestB]) bestB = label;
      }
      assert.equal(bestA, bestB, 'Logit argmax changed');
    }
  }
  assert(maximumDifference < policy.logitTolerance, 'Logit numerical parity failed');
  return {passed: true, lengths: before.map(row => row.length), maximumDifference};
}

export function validateFrozenQuality(manifest, bytes, split) {
  const expected = frozenQuality[split];
  assert(expected && manifest?.schema === 1 && manifest.dataset === 'JFLEG' && manifest.revision === frozenQuality.revision &&
    manifest.evaluation_only === true && manifest.training_allowed === false && manifest.license === 'CC-BY-NC-SA-4.0', 'Untrusted quality manifest');
  assert.deepEqual(manifest.splits?.[split], expected, 'Quality manifest split mismatch');
  assert.equal(digest(bytes), expected.sha256, 'Frozen quality hash mismatch');
  const rows = bytes.toString('utf8').trimEnd().split('\n').map(line => JSON.parse(line));
  assert.equal(rows.length, expected.rows, 'Incomplete quality population');
  rows.forEach((row, index) => assert(row.id === `jfleg-${split}-${index}` && typeof row.source === 'string' && row.source.trim() &&
    row.references?.length === 4 && row.references.every(value => typeof value === 'string' && value.trim()) &&
    row.source_revision === frozenQuality.revision && row.evaluation_only === true, 'Invalid frozen quality row'));
  return rows;
}

export function validateAssetSet(files) {
  const manifest = JSON.parse(files['manifest.json'].toString('utf8'));
  assert.notEqual(manifest.disableModelEdits, true, 'Disabled model cannot establish inference performance');
  const hashes = {};
  for (const name of ['manifest.json', 'labels.json', 'vocab.txt', 'model.onnx']) {
    assert(files[name]?.length > 0, `Missing model asset: ${name}`);
    hashes[name] = digest(files[name]);
    if (name !== 'manifest.json') assert.equal(hashes[name], manifest.files?.[name]?.sha256, `Model hash mismatch: ${name}`);
  }
  return hashes;
}

// Resample complete paired trials, preserving correlation among workloads. The
// lower bound of benefit must exceed 5%; a point estimate alone never promotes.
export function timingGate(before, after, qualityComplete, anchorGuard = true) {
  if (!qualityComplete) return {status: 'inconclusive', exitCode: 3, reason: 'complete-frozen-development-quality-required'};
  if (!anchorGuard) return {status: 'rejected', exitCode: 2, reason: 'immutable-anchor-regression'};
  if (!before || before.length < policy.minimumTrials || after?.length !== before.length) return {status: 'inconclusive', exitCode: 3, reason: 'insufficient-paired-trials'};
  const ratios = [], regressions = {}, p95Regressions = {};
  for (const name of [...Object.keys(policy.weights), 'nearby', 'cold']) {
    const values = before.map((row, index) => {
      const a = name === 'cold' ? row.coldMs : row[name]?.medianMs, b = name === 'cold' ? after[index].coldMs : after[index][name]?.medianMs;
      assert(Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0, 'Invalid timing population');
      return b / a;
    });
    regressions[name] = percentile(values, 0.5) - 1;
    if (name !== 'cold') {
      p95Regressions[name] = percentile(before.map((row, index) => {
        const a = row[name]?.p95Ms, b = after[index][name]?.p95Ms;
        assert(Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0, 'Missing p95 guard measurements');
        return b / a;
      }), 0.5) - 1;
    }
  }
  for (let index = 0; index < before.length; index++) ratios.push(Object.entries(policy.weights)
    .reduce((sum, [name, weight]) => sum + weight * Math.log(after[index][name].medianMs / before[index][name].medianMs), 0));
  let seed = policy.seed;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const bootstrap = Array.from({length: policy.bootstrapSamples}, () => {
    let sum = 0;
    for (let index = 0; index < ratios.length; index++) sum += ratios[Math.floor(random() * ratios.length)];
    return 1 - Math.exp(sum / ratios.length);
  });
  const speedup = 1 - Math.exp(ratios.reduce((sum, value) => sum + value, 0) / ratios.length);
  const interval = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)];
  const regression = Object.values(regressions).some(value => value > policy.maximumRegression) ||
    Object.values(p95Regressions).some(value => value > policy.maximumP95Regression);
  const status = regression || interval[1] < policy.minimumSpeedup ? 'rejected' : interval[0] >= policy.minimumSpeedup ? 'accepted' : 'inconclusive';
  return {status, exitCode: {accepted: 0, rejected: 2, inconclusive: 3}[status], reason: regression ? 'workload-regression' :
    status === 'accepted' ? 'paired-speedup-with-quality-parity' : status === 'rejected' ? 'insufficient-speedup' : 'timing-uncertainty',
    speedup, speedup95: interval, regressions, p95Regressions, pairedTrials: ratios.length};
}

export const escapeHtml = value => String(value).replace(/[&<>"']/gu, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));

// Decode only standard v3 generated-to-original locations. Names remain the
// profiler's real function names; maps improve file/line refs, not fabricated stacks.
export function sourceResolver(map) {
  if (!map) return frame => `${frame.url || '(native)'}:${frame.lineNumber + 1}:${frame.columnNumber + 1}`;
  assert(map.version === 3 && Array.isArray(map.sources) && typeof map.mappings === 'string', 'Invalid source map');
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let source = 0, line = 0, column = 0;
  const lines = map.mappings.split(';').map(encoded => {
    let generated = 0;
    return encoded.split(',').filter(Boolean).map(segment => {
      const fields = []; let value = 0, shift = 0;
      for (const character of segment) {
        const digit = alphabet.indexOf(character); assert(digit >= 0, 'Invalid source map VLQ');
        value += (digit & 31) * 2 ** shift;
        if (digit & 32) shift += 5;
        else { fields.push(value & 1 ? -(value >> 1) : value >> 1); value = 0; shift = 0; }
      }
      assert(shift === 0 && fields.length > 0, 'Truncated source map VLQ');
      generated += fields[0];
      if (fields.length < 4) return {generated};
      source += fields[1]; line += fields[2]; column += fields[3];
      assert(source >= 0 && source < map.sources.length && line >= 0 && column >= 0, 'Invalid source map location');
      return {generated, source: map.sources[source], line, column};
    });
  });
  return frame => {
    if (frame.lineNumber >= 0 && frame.url?.endsWith('.mjs')) {
      const row = lines[frame.lineNumber]?.filter(item => item.generated <= frame.columnNumber).at(-1);
      if (row?.source) return `${row.source}:${row.line + 1}:${row.column + 1}`;
    }
    return `${frame.url || '(native)'}:${frame.lineNumber + 1}:${frame.columnNumber + 1}`;
  };
}

export function profileTree(profile, resolveSource = sourceResolver(null)) {
  assert(Array.isArray(profile?.nodes) && profile.nodes.length && Array.isArray(profile.samples) &&
    Array.isArray(profile.timeDeltas) && profile.samples.length === profile.timeDeltas.length &&
    Number.isFinite(profile.startTime) && Number.isFinite(profile.endTime) && profile.endTime >= profile.startTime, 'Malformed CPU profile');
  const nodes = new Map(), parents = new Map();
  for (const node of profile.nodes) {
    assert(Number.isInteger(node.id) && !nodes.has(node.id) && typeof node.callFrame?.functionName === 'string', 'Malformed CPU node');
    nodes.set(node.id, node);
  }
  for (const node of profile.nodes) for (const child of node.children ?? []) {
    assert(nodes.has(child) && !parents.has(child), 'Malformed CPU parents'); parents.set(child, node.id);
  }
  const roots = profile.nodes.filter(node => !parents.has(node.id));
  assert(roots.length === 1, 'CPU profile requires one root');
  for (const node of profile.nodes) {
    const seen = new Set(); let id = node.id;
    while (id !== undefined) { assert(!seen.has(id), 'CPU profile parent cycle'); seen.add(id); id = parents.get(id); }
  }
  const root = {name: 'Sampled renderer CPU (including GC)', value: 0, self: 0, children: new Map()};
  const excluded = {idleUs: 0, programUs: 0, rootUs: 0};
  let sampledUs = 0, activeSamples = 0, garbageCollectionUs = 0;
  profile.samples.forEach((id, index) => {
    const delta = profile.timeDeltas[index];
    assert(nodes.has(id) && Number.isFinite(delta) && delta >= 0, 'Malformed CPU sample delta');
    sampledUs += delta;
    const stack = []; let current = id;
    while (current !== undefined) { stack.unshift(nodes.get(current)); current = parents.get(current); }
    const leaf = stack.at(-1).callFrame.functionName;
    if (leaf === '(garbage collector)') garbageCollectionUs += delta;
    const category = {'(idle)': 'idleUs', '(program)': 'programUs', '(root)': 'rootUs'}[leaf];
    if (category) { excluded[category] += delta; return; }
    activeSamples++; root.value += delta;
    let branch = root;
    for (const node of stack.slice(1)) {
      const frame = node.callFrame, ref = resolveSource(frame), name = frame.functionName || '(anonymous)';
      const key = name + '\0' + ref;
      if (!branch.children.has(key)) branch.children.set(key, {name, ref, value: 0, self: 0, children: new Map()});
      branch = branch.children.get(key); branch.value += delta;
    }
    branch.self += delta;
  });
  assert(sampledUs <= (profile.endTime - profile.startTime) * 1.1 + 10000, 'CPU samples exceed capture duration');
  return {root, excluded, garbageCollectionUs, sampledUs, activeSamples, hotspotsReliable: activeSamples >= policy.minimumProfileSamples, samples: profile.samples.length,
    wallUs: profile.endTime - profile.startTime, unaccountedUs: Math.max(0, profile.endTime - profile.startTime - sampledUs)};
}

export function flamegraphSvg(tree, title) {
  const width = 1280, rowHeight = 24; let maximumDepth = 0; const rectangles = [];
  function walk(branch, x, depth) {
    maximumDepth = Math.max(maximumDepth, depth);
    const size = tree.root.value ? width * branch.value / tree.root.value : 0;
    if (size >= 0.1) {
      const color = 25 + parseInt(digest(branch.name).slice(0, 2), 16) % 40;
      const description = `${branch.name}\n${branch.ref ?? ''}\nInclusive ${(branch.value / 1000).toFixed(3)} ms; self ${(branch.self / 1000).toFixed(3)} ms`;
      rectangles.push(`<g><title>${escapeHtml(description)}</title><rect x="${x.toFixed(2)}" y="${50 + depth * rowHeight}" width="${size.toFixed(2)}" height="23" fill="hsl(${color},80%,72%)" stroke="white"/><text x="${(x + 3).toFixed(2)}" y="${66 + depth * rowHeight}">${escapeHtml(branch.name.slice(0, Math.max(0, Math.floor(size / 7) - 1)))}</text></g>`);
    }
    for (const child of [...branch.children.values()].sort((a, b) => a.name.localeCompare(b.name))) { walk(child, x, depth + 1); x += tree.root.value ? width * child.value / tree.root.value : 0; }
  }
  walk(tree.root, 0, 0);
  const height = 85 + (maximumDepth + 1) * rowHeight;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><style>text{font:12px monospace}rect:hover{stroke:#111;stroke-width:2}</style><rect width="100%" height="100%" fill="#fff"/><text x="8" y="20">${escapeHtml(title)}${tree.hotspotsReliable ? '' : ' — Sparse active CPU samples: hotspot ranking is unreliable'}</text><text x="8" y="38">${tree.activeSamples} active samples, ${(tree.root.value / 1000).toFixed(1)} ms CPU. Width includes GC; excludes idle, program and unsampled wall time. Hover for source.</text>${rectangles.join('')}</svg>`;
}
