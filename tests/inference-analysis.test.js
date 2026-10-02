import {describe, expect, it} from 'vite-plus/test';
import {compareLogits, compareOutputs, digest, flamegraphSvg, policy, profileTree, sourceResolver,
  timingGate, validateAssetSet, validateFrozenQuality} from '../scripts/inference-analysis.mjs';

const frame = (id, name, children = []) => ({id, callFrame: {functionName: name, url: 'engine.mjs', lineNumber: 0, columnNumber: 0}, children});
const profile = overrides => ({nodes: [frame(1, '(root)', [2, 4]), frame(2, 'recursive <unsafe>', [3]),
  frame(3, 'recursive <unsafe>'), frame(4, '(idle)')], startTime: 0, endTime: 6000,
  samples: [3, 4, 2], timeDeltas: [1000, 3000, 2000], ...overrides});
const output = (confidence = 0.9) => [{id: 'case', corrected: 'Fine.', suggestions: [{start: 0, end: 1, original: 'F', replacement: 'F', confidence}]}];
const trials = ratios => ratios.map(ratio => ({coldMs: 100 * ratio,
  ...Object.fromEntries(['unchanged', 'edited', 'fresh', 'nearby'].map(name => [name, {medianMs: 10 * ratio, p95Ms: 11 * ratio}]))}));

describe('trusted inference analysis', () => {
  it('keeps recursion and source refs while excluding idle from CPU width', () => {
    const tree = profileTree(profile());
    expect(tree.root.value).toBe(3000);
    expect(tree.excluded.idleUs).toBe(3000);
    expect(tree.sampledUs).toBe(6000);
    expect(tree.negativeDeltaSamples).toBe(0);
    expect(tree.reorderedSamples).toBe(0);
    expect(tree.root.children.values().next().value.children.size).toBe(1);
    const svg = flamegraphSvg(tree, '<script>alert("title")</script>');
    expect(svg).toContain('&lt;script&gt;');
    expect(svg).toContain('recursive &lt;unsafe&gt;');
    expect(svg).not.toContain('<script>');
    expect(svg).toContain('engine.mjs:1:1');
  });
  it('rejects dangling IDs, cyclic parents, invalid deltas and truncated profiles', () => {
    for (const malformed of [profile({samples: [99], timeDeltas: [1]}), profile({timeDeltas: [1]}),
      profile({timeDeltas: [-1, 1, 1]}), profile({timeDeltas: [NaN, 1, 1]}),
      profile({timeDeltas: [3000, -4000, 3000]}), profile({timeDeltas: [7000, -6000, 1000]}),
      profile({nodes: [frame(1, '(root)'), frame(2, 'a', [3]), frame(3, 'b', [2])]}),
      profile({nodes: [frame(1, '(root)', [2, 3]), frame(2, 'a', [3]), frame(3, 'b')]}),
      profile({timeDeltas: [100000, 1, 1]})]) expect(() => profileTree(malformed)).toThrow();
  });
  it('sorts timestamp/sample pairs without changing raw evidence or total elapsed time', () => {
    const capture = profile({nodes: [frame(1, '(root)', [2, 3, 4]), frame(2, 'a'), frame(3, 'b'), frame(4, '(idle)')],
      startTime: 100, endTime: 120, samples: [2, 3, 4], timeDeltas: [10, -1, 5]});
    const raw = structuredClone(capture), tree = profileTree(capture);
    const frames = new Map([...tree.root.children.values()].map(node => [node.name, node.self]));
    expect(capture).toEqual(raw);
    expect(tree.negativeDeltaSamples).toBe(1);
    expect(tree.reorderedSamples).toBe(2);
    expect(frames.get('a')).toBe(1);
    expect(frames.get('b')).toBe(9);
    expect(tree.excluded.idleUs).toBe(4);
    expect(tree.sampledUs).toBe(capture.timeDeltas.reduce((sum, delta) => sum + delta, 0));
    expect(tree.root.value + tree.excluded.idleUs).toBe(tree.sampledUs);
  });
  it('preserves the observed -1 microsecond inversion and stable equal-time ordering', () => {
    const nodes = [frame(1, '(root)', [2, 3]), frame(2, 'a'), frame(3, 'b')];
    const tree = profileTree(profile({nodes, samples: [2, 3, 2, 2, 3], timeDeltas: [7, 8, -1, 9, 17]}));
    const frames = new Map([...tree.root.children.values()].map(node => [node.name, node.self]));
    expect(tree.negativeDeltaSamples).toBe(1);
    expect(tree.reorderedSamples).toBe(2);
    expect(tree.sampledUs).toBe(40);
    expect(tree.root.value).toBe(40);
    expect(frames.get('a')).toBe(22);
    expect(frames.get('b')).toBe(18);
    const equal = profileTree(profile({nodes, samples: [2, 3, 2], timeDeltas: [10, 0, 5]}));
    expect(equal.reorderedSamples).toBe(0);
    expect([...equal.root.children.values()].map(node => [node.name, node.self])).toEqual([['a', 15], ['b', 0]]);
  });
  it('accounts for zero deltas, empty samples and all-idle profiles without invented CPU', () => {
    expect(profileTree(profile({timeDeltas: [0, 3000, 2000]})).root.value).toBe(2000);
    const empty = profileTree(profile({samples: [], timeDeltas: []}));
    expect(empty.activeSamples).toBe(0);
    expect(empty.unaccountedUs).toBe(6000);
    expect(flamegraphSvg(empty, 'Empty')).toContain('0 active samples');
    const idle = profileTree(profile({samples: [4], timeDeltas: [6000]}));
    expect(idle.root.value).toBe(0);
    expect(idle.excluded.idleUs).toBe(6000);
  });
  it('includes garbage collection in CPU width so allocation bottlenecks remain visible', () => {
    const capture = profile({nodes: [frame(1, '(root)', [2, 4]), frame(2, '(garbage collector)'), frame(4, '(idle)')],
      samples: [2, 4], timeDeltas: [4000, 2000]});
    const tree = profileTree(capture);
    expect(tree.root.value).toBe(4000);
    expect(tree.garbageCollectionUs).toBe(4000);
    expect(tree.excluded.idleUs).toBe(2000);
    expect(flamegraphSvg(tree, 'Allocation')).toContain('(garbage collector)');
  });
  it('counts frames named like Object prototype properties as real CPU', () => {
    for (const name of ['constructor', 'toString', '__proto__']) {
      const tree = profileTree(profile({nodes: [frame(1, '(root)', [2]), frame(2, name)],
        samples: [2], timeDeltas: [6000]}));
      expect(tree.root.value).toBe(6000);
      expect([...tree.root.children.values()][0].name).toBe(name);
    }
  });
  it('maps generated frame refs without renaming symbols', () => {
    const resolver = sourceResolver({version: 3, sources: ['src/model.ts'], mappings: 'AAAA'});
    expect(resolver(frame(2, 'analyzePass').callFrame)).toBe('src/model.ts:1:1');
    expect(() => sourceResolver({version: 3, sources: [], mappings: '?'})).toThrow();
  });
  it('requires every output/edit field and finite confidence parity', () => {
    expect(compareOutputs(output(), output(0.90000001)).passed).toBe(true);
    for (const rows of [[], [{...output()[0], corrected: 'Wrong.'}], output(NaN), output(0.5),
      [{...output()[0], id: 'other'}], [{...output()[0], suggestions: []}],
      [{...output()[0], suggestions: [{...output()[0].suggestions[0], start: 1}]}]]) expect(() => compareOutputs(output(), rows)).toThrow();
  });
  it('requires real logit dimensions, finite values and identical argmax', () => {
    const logits = [{length: 3, dims: [1, 3, 2], values: [1, 0, 1, 0, 1, 0]}];
    expect(compareLogits(logits, logits).maximumDifference).toBe(0);
    for (const row of [{...logits[0], dims: [1, 4, 2]}, {...logits[0], values: [1]},
      {...logits[0], values: [NaN, 0, 1, 0, 1, 0]}, {...logits[0], values: [0, 1, 1, 0, 1, 0]},
      {...logits[0], values: [1.01, 0, 1, 0, 1, 0]}]) expect(() => compareLogits(logits, [row])).toThrow();
  });
  it('validates all executed model assets and rejects disabled/stale policies', () => {
    const files = {'labels.json': Buffer.from('["KEEP"]'), 'vocab.txt': Buffer.from('[CLS]\n[SEP]\n[UNK]\n'), 'model.onnx': Buffer.from('weights')};
    const manifest = {files: Object.fromEntries(Object.entries(files).map(([name, bytes]) => [name, {sha256: digest(bytes)}]))};
    files['manifest.json'] = Buffer.from(JSON.stringify(manifest));
    expect(validateAssetSet(files)['model.onnx']).toBe(digest(files['model.onnx']));
    expect(() => validateAssetSet({...files, 'model.onnx': Buffer.from('different')})).toThrow(/hash mismatch/u);
    expect(() => validateAssetSet({...files, 'manifest.json': Buffer.from(JSON.stringify({...manifest, disableModelEdits: true}))})).toThrow(/Disabled/u);
  });
  it('rejects reduced or relabeled quality populations before scoring', () => {
    expect(() => validateFrozenQuality({schema: 1, dataset: 'JFLEG', revision: 'wrong'}, Buffer.from('{}\n'), 'dev')).toThrow();
    expect(() => validateFrozenQuality({schema: 1, dataset: 'other'}, Buffer.from('{}\n'), 'test')).toThrow();
  });
  it('promotes only paired speedup with complete quality and immutable anchor safety', () => {
    const baseline = trials([1, 1, 1, 1, 1]);
    expect(timingGate(baseline, trials([0.8, 0.8, 0.8, 0.8, 0.8]), true).status).toBe('accepted');
    expect(timingGate(baseline, trials([1, 1, 1, 1, 1]), true).status).toBe('rejected');
    expect(timingGate(baseline, trials([0.7, 1.2, 0.7, 1.2, 0.7]), true).status).toBe('inconclusive');
    expect(timingGate(baseline, trials([0.8, 0.8, 0.8, 0.8, 0.8]), false).status).toBe('inconclusive');
    expect(timingGate(baseline, trials([0.8, 0.8, 0.8, 0.8, 0.8]), true, false).status).toBe('rejected');
    expect(timingGate(baseline.slice(0, 4), baseline.slice(0, 4), true).status).toBe('inconclusive');
    const candidate = trials([0.8, 0.8, 0.8, 0.8, 0.8]);
    candidate.forEach(row => { row.nearby.medianMs = 20; });
    expect(timingGate(baseline, candidate, true).reason).toBe('workload-regression');
    const tail = trials([0.8, 0.8, 0.8, 0.8, 0.8]);
    tail.forEach(row => { row.edited.p95Ms = 30; });
    expect(timingGate(baseline, tail, true).reason).toBe('workload-regression');
    expect(policy.weights.edited).toBe(0.6);
  });
});
