import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';
import { analyzeText, applySuggestions } from '../packages/engine/src/index';

const runtime = vi.hoisted(() => ({runs: 0, predict: (_ids: number[], _position: number): number => 0}));
vi.mock('../packages/engine/src/jax-runtime', () => ({
  JaxSession: {create: async () => ({backend: 'wasm', run: async (ids: number[]) => {
    runtime.runs++;
    const values = new Float32Array(ids.length * 6);
    for (let position = 0; position < ids.length; position++) values[position * 6 + runtime.predict(ids, position)] = 12;
    return {data: values, dims: [1, ids.length, 6]};
  }})},
}));

let manifest: Record<string, unknown>;
let key = 0;
const labels = ['KEEP', 'CASE:TITLE', 'CASE:LOWER', 'SUFFIX:ADD_S', 'APPEND_EXACT:,', 'REPLACE_EXACT:had'];
const vocab = '[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n.\nworld\nbooks\n,\nshe\nhave\na\nbook\n';
const options = () => ({modelBaseUrl: '/test-model-' + key++ + '/', preferWebGPU: false});

beforeEach(() => {
  runtime.runs = 0;
  runtime.predict = () => 0;
  manifest = {confidenceThreshold: .8, maxSequenceLength: 64, editSchema: 2};
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('manifest.json')) return new Response(JSON.stringify(manifest));
    if (url.endsWith('labels.json')) return new Response(JSON.stringify(labels));
    if (url.endsWith('vocab.txt')) return new Response(vocab);
    return new Response(new Uint8Array([0]));
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe('model deployment policy', () => {
  it('runs rules-only checks without fetching or invoking a model', async () => {
    const result = await analyzeText('She have a book.', {...options(), mode: 'rules'});
    expect(applySuggestions(result.text, result.suggestions)).toBe('She has a book.');
    expect(fetch).not.toHaveBeenCalled();
    expect(runtime.runs).toBe(0);
  });

  it('suppresses all neural edits when development calibration disables them', async () => {
    manifest.disableModelEdits = true;
    runtime.predict = () => 1;
    const result = await analyzeText('hello.', {...options(), mode: 'model', maxPasses: 3});
    expect(result.suggestions).toEqual([]);
    expect(result.modelRuns).toBe(0);
    expect(runtime.runs).toBe(0);
  });

  it('never relaxes a caller threshold with a family threshold', async () => {
    manifest.confidenceThresholds = {case: .9};
    runtime.predict = (ids, position) => ids[position] === 4 ? 1 : 0;
    const result = await analyzeText('hello.', {...options(), mode: 'model', confidenceThreshold: 1});
    expect(result.suggestions).toEqual([]);
    expect(result.modelRuns).toBe(1);
  });

  it('composes punctuation insertion and stops when the next pass has no edits', async () => {
    runtime.predict = (ids, position) => ids[position] === 4 && !ids.includes(8) ? 4 : 0;
    const result = await analyzeText('😀 hello world.', {...options(), mode: 'model', maxPasses: 3});
    expect(applySuggestions(result.text, result.suggestions)).toBe('😀 hello, world.');
    expect(result.modelRuns).toBe(2);
    expect(() => applySuggestions('stale ' + result.text, result.suggestions)).toThrow(/changed/u);
  });

  it('stops correction cycles without undoing the accepted first pass', async () => {
    runtime.predict = (ids, position) => ids[position] === 4 ? (runtime.runs === 1 ? 1 : 2) : 0;
    const result = await analyzeText('hello.', {...options(), mode: 'model', maxPasses: 3});
    expect(applySuggestions(result.text, result.suggestions)).toBe('Hello.');
    expect(result.modelRuns).toBe(2);
  });

  it('keeps rules authoritative and skips inference in protected windows', async () => {
    runtime.predict = (ids, position) => ids[position] === 10 ? 5 : 0;
    const result = await analyzeText('She have a book.', {...options(), maxPasses: 2});
    expect(applySuggestions(result.text, result.suggestions)).toBe('She has a book.');
    const protectedResult = await analyzeText('`She have a book.`', {...options(), mode: 'model'});
    expect(protectedResult.suggestions).toEqual([]);
    expect(protectedResult.modelRuns).toBe(0);
  });

  it('rejects malformed correction policies through the existing rules fallback', async () => {
    for (const invalid of [
      {editSchema: 3}, {confidenceThreshold: NaN}, {maxSequenceLength: 513},
      {maxPasses: 4}, {disableModelEdits: 'false'}, {confidenceThresholds: {case: .1}},
    ]) {
      manifest = {...manifest, ...invalid};
      const result = await analyzeText('She have a book.', options());
      expect(result.modelError).toMatch(/Invalid model/u);
      expect(applySuggestions(result.text, result.suggestions)).toBe('She has a book.');
      manifest = {confidenceThreshold: .8, maxSequenceLength: 64, editSchema: 2};
    }
  });
});
