import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';
import { applySuggestions } from '../packages/engine/src/edits';
import { analyzeModel } from '../packages/engine/src/model';
import { analyzeRules } from '../packages/engine/src/rules';

const runtime = vi.hoisted(() => ({
  run: vi.fn<(ids: number[]) => Promise<{data: Float32Array; dims: number[]}>>(),
  create: vi.fn(),
}));
vi.mock('../packages/engine/src/jax-runtime', () => ({JaxSession: {create: runtime.create}}));

const labels = ['KEEP', 'REPLACE:friend', 'APPEND_EXACT:,', 'REPLACE:are'];
const words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'a', 'freind', 'friend', 'called',
  '.', 'hello', 'world', 'alpha', 'my', 'is', 'hungry'];
let predictions: Record<string, string>;
let key = 0;
let options: {modelBaseUrl: string; preferWebGPU: boolean};

beforeEach(() => {
  options = {modelBaseUrl: `/cache-model-${key++}/`, preferWebGPU: false};
  predictions = {freind: 'REPLACE:friend'};
  runtime.create.mockReset().mockResolvedValue({backend: 'wasm', run: runtime.run});
  runtime.run.mockReset().mockImplementation(async ids => {
    const data = new Float32Array(ids.length * labels.length).fill(-4);
    for (const [position, id] of ids.entries()) {
      const prediction = predictions[words[id]] ?? 'KEEP';
      data[position * labels.length + labels.indexOf(prediction)] = 4;
    }
    return {data, dims: [1, ids.length, labels.length]};
  });
  vi.stubGlobal('fetch', async (url: string) => {
    if (url.endsWith('manifest.json')) return new Response(JSON.stringify({confidenceThreshold: .8, maxSequenceLength: 64, editSchema: 2}));
    if (url.endsWith('labels.json')) return new Response(JSON.stringify(labels));
    if (url.endsWith('vocab.txt')) return new Response(words.join('\n'));
    return new Response(new Uint8Array([0]));
  });
});
afterEach(() => vi.unstubAllGlobals());

describe('session-local raw logits reuse', () => {
  it('reuses unchanged sentences and runs only the edited sentence', async () => {
    const text = 'A freind called. Hello world.';
    const first = await analyzeModel(text, options);
    expect(first.modelRuns).toBe(2);
    const repeated = await analyzeModel(text, options);
    expect(repeated.modelRuns).toBe(0);
    expect(repeated.suggestions).toEqual(first.suggestions);
    expect(repeated.backend).toBe('wasm');
    const edited = 'A freind called. Hello alpha.';
    const result = await analyzeModel(edited, options);
    expect(result.modelRuns).toBe(1);
    expect(applySuggestions(edited, result.suggestions)).toBe('A friend called. Hello alpha.');
    expect((await analyzeModel(text, options)).modelRuns).toBe(0);
    expect(runtime.run).toHaveBeenCalledTimes(3);
  });

  it('decodes current casing, normalized source words, and shifted UTF-16 offsets', async () => {
    await analyzeModel('A freind called.', options);
    const accented = await analyzeModel('A Fréind called.', options);
    expect(accented.modelRuns).toBe(0);
    expect(accented.suggestions).toEqual([]);
    const capitalized = await analyzeModel('  A FREIND called.', options);
    expect(capitalized.modelRuns).toBe(0);
    expect(capitalized.suggestions[0]).toMatchObject({start: 4, end: 10, original: 'FREIND', replacement: 'FRIEND'});
    const shifted = await analyzeModel('😀. A FREIND called.', options);
    expect(shifted.modelRuns).toBe(1); // Only the new emoji sentence.
    expect(shifted.suggestions[0]).toMatchObject({start: 6, end: 12, original: 'FREIND', replacement: 'FRIEND'});
    expect(applySuggestions('😀. A FREIND called.', shifted.suggestions)).toBe('😀. A FRIEND called.');
  });

  it('reevaluates full-source verb guards when unknown words share token IDs', async () => {
    predictions = {is: 'REPLACE:are'};
    const first = await analyzeModel('My cats is hungry.', options);
    expect(applySuggestions('My cats is hungry.', first.suggestions)).toBe('My cats are hungry.');
    const valid = await analyzeModel('My cat is hungry.', options);
    expect(valid.modelRuns).toBe(0);
    expect(valid.suggestions).toEqual([]);
    expect(runtime.run).toHaveBeenCalledTimes(1);
  });

  it('reevaluates thresholds and owns logits independently of runtime output', async () => {
    const first = await analyzeModel('A freind called.', options);
    expect(first.suggestions).toHaveLength(1);
    const output = await runtime.run.mock.results[0].value;
    output.data.fill(NaN);
    const strict = await analyzeModel('A freind called.', {...options, confidenceThreshold: 1});
    expect(strict.modelRuns).toBe(0);
    expect(strict.suggestions).toEqual([]);
    const relaxed = await analyzeModel('A freind called.', options);
    expect(relaxed.modelRuns).toBe(0);
    expect(relaxed.suggestions).toEqual(first.suggestions);
  });

  it('rebuilds insertion snapshots for current whitespace and rejects stale drafts', async () => {
    predictions = {hello: 'APPEND_EXACT:,'};
    const first = await analyzeModel('Hello world.', options);
    const current = 'Hello\tworld.';
    const second = await analyzeModel(current, options);
    expect(second.modelRuns).toBe(0);
    expect(second.suggestions[0].checkedText).toBe(current);
    expect(applySuggestions(current, second.suggestions)).toBe('Hello,\tworld.');
    expect(() => applySuggestions(current, first.suggestions)).toThrow(/changed/u);
  });

  it('skips current protected windows and rule deletions before cache lookup', async () => {
    await analyzeModel('Hello. A freind called.', options);
    const protectedResult = await analyzeModel('`Hello. A freind called.`', options);
    expect(protectedResult.modelRuns).toBe(0);
    expect(protectedResult.suggestions).toEqual([]);
    const repeated = 'A freind called called.';
    expect((await analyzeModel(repeated, options)).suggestions).toHaveLength(1);
    const rules = analyzeRules(repeated);
    expect(rules.some(edit => !edit.replacement)).toBe(true);
    const guarded = await analyzeModel(repeated, options, rules);
    expect(guarded.modelRuns).toBe(0);
    expect(guarded.suggestions).toEqual([]);
    expect(runtime.run).toHaveBeenCalledTimes(3);
  });

  it('keeps different model URLs and backend sessions independent', async () => {
    await analyzeModel('A freind called.', options);
    expect((await analyzeModel('A freind called.', {...options, modelBaseUrl: options.modelBaseUrl + 'other/'})).modelRuns).toBe(1);
    expect((await analyzeModel('A freind called.', {...options, preferWebGPU: true})).modelRuns).toBe(1);
    expect((await analyzeModel('A freind called.', options)).modelRuns).toBe(0);
    expect(runtime.create).toHaveBeenCalledTimes(3);
  });

  it.each(['shape', 'length', 'nan', 'infinity', 'failure'])('does not cache %s outputs and recovers on retry', async kind => {
    runtime.run.mockImplementationOnce(async ids => {
      if (kind === 'failure') throw new Error('Runtime failed');
      const data = new Float32Array(ids.length * labels.length);
      if (kind === 'nan') data[0] = NaN;
      if (kind === 'infinity') data[0] = Infinity;
      return {data: kind === 'length' ? data.subarray(1) : data,
        dims: [1, ids.length + (kind === 'shape' ? 1 : 0), labels.length]};
    });
    await expect(analyzeModel('A freind called.', options)).rejects.toThrow(/output|Runtime failed/u);
    const recovered = await analyzeModel('A freind called.', options);
    expect(recovered.modelRuns).toBe(1);
    expect(recovered.suggestions).toHaveLength(1);
    expect((await analyzeModel('A freind called.', options)).modelRuns).toBe(0);
    expect(runtime.run).toHaveBeenCalledTimes(2);
  });
});
