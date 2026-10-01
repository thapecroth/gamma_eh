import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';
import reference from './fixtures/onnx-reference.json';
import regression from '../data/regression.json';
import { JaxSession } from '../packages/engine/src/jax-runtime';
import { analyzeText, applySuggestions } from '../packages/engine/src/index';

const modelDirectory = new URL('../models/browser/', import.meta.url);
afterEach(() => vi.unstubAllGlobals());

describe('JAX model migration', () => {
  it('matches FP32 ONNX Runtime logits at short and maximum context lengths on repeated runs', async () => {
    const bytes = new Uint8Array(await readFile(new URL('model.onnx', modelDirectory)));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(reference.modelSha256);
    const session = await JaxSession.create(bytes, false);
    try {
      for (let repeat = 0; repeat < 2; repeat++) {
        for (const row of reference.rows) {
          const output = await session.run(row.ids);
          expect(output.dims).toEqual(row.dims);
          expect(output.data.length).toBe(row.logits.length);
          const error = Math.max(...row.logits.map((value, i) => Math.abs(value - output.data[i])));
          expect(error).toBeLessThan(0.0001);
        }
      }
    } finally { session.dispose(); }
  });

  it('executes the real JAX WASM model for every grammar regression', async () => {
    vi.stubGlobal('fetch', async (url: string) => new Response(await readFile(new URL(url.split('/').at(-1)!, modelDirectory))));
    for (const item of regression.cases) {
      const result = await analyzeText(item.source, {modelBaseUrl: '/jax-regression/', preferWebGPU: false});
      expect(result.modelError).toBeUndefined();
      expect(result.backend).toBe('wasm');
      expect(applySuggestions(item.source, result.suggestions)).toBe(item.target);
      if (item.source === 'The students has a notebook.') expect(result.suggestions.some(edit => edit.source === 'model')).toBe(true);
    }
  });

  it('retries an asset failure and preserves model UTF-16 offsets and stale-edit protection', async () => {
    let unavailable = true;
    vi.stubGlobal('fetch', async (url: string) => unavailable ? new Response('', {status: 404})
      : new Response(await readFile(new URL(url.split('/').at(-1)!, modelDirectory))));
    const options = {modelBaseUrl: '/jax-retry/', preferWebGPU: false};
    const text = '😀 The students has a notebook.';
    const failed = await analyzeText(text, options);
    expect(failed.backend).toBe('rules');
    expect(failed.modelError).toMatch(/404/u);
    unavailable = false;
    const recovered = await analyzeText(text, options);
    expect(recovered.backend).toBe('wasm');
    const edit = recovered.suggestions.find(item => item.source === 'model')!;
    expect(edit).toMatchObject({start: 16, end: 19, original: 'has', replacement: 'have'});
    expect(applySuggestions(text, recovered.suggestions)).toBe('😀 The students have a notebook.');
    expect(() => applySuggestions('😀 The students had a notebook.', recovered.suggestions)).toThrow(/changed/u);
  });

  it('rejects unusable context lengths before model execution', async () => {
    for (const length of [0, 2, 64.5, 513]) {
      vi.stubGlobal('fetch', async (url: string) => url.endsWith('manifest.json')
        ? Response.json({confidenceThreshold: 0.85, maxSequenceLength: length})
        : new Response(await readFile(new URL(url.split('/').at(-1)!, modelDirectory))));
      const result = await analyzeText('She have a book.', {modelBaseUrl: `/jax-invalid-${length}/`, preferWebGPU: false});
      expect(result.backend).toBe('rules');
      expect(result.modelError).toBe('Invalid model manifest');
    }
  });
});
