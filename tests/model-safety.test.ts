import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';
import { applySuggestions } from '../packages/engine/src/edits';
import { analyzeModel } from '../packages/engine/src/model';

const runtime = vi.hoisted(() => ({run: vi.fn()}));
vi.mock('../packages/engine/src/jax-runtime', () => ({
  JaxSession: {create: async () => ({backend: 'wasm', run: runtime.run})},
}));

const labels = ['KEEP', 'REPLACE:am', 'REPLACE:is', 'REPLACE:are', 'REPLACE:was', 'REPLACE:were',
  'REPLACE:has', 'REPLACE:have', 'REPLACE:go', 'REPLACE:goes', 'REPLACE:read', 'REPLACE:reads', 'REPLACE:friend',
  'REPLACE:the', 'REPLACE:because', "REPLACE:they're", 'REPLACE:a', 'REPLACE:an', 'APPEND:a', 'APPEND:an', 'DELETE'];
const vocabulary = '[PAD]\n[UNK]\n[CLS]\n[SEP]\nam\nis\nare\nwas\nwere\nhas\nhave\ngo\ngoes\nread\nreads\nfreind\nbe\ntecnologies\ntheir\nspeling\na\nan\nbook\numbrella\napple\none\n';
const ids = vocabulary.trim().split('\n');
let predictions: Record<string, string>;

afterEach(() => vi.unstubAllGlobals());

beforeEach(() => {
  predictions = {};
  vi.stubGlobal('fetch', async (url: string) => {
    if (url.endsWith('manifest.json')) return new Response(JSON.stringify({confidenceThreshold: .85, maxSequenceLength: 8}));
    if (url.endsWith('labels.json')) return new Response(JSON.stringify(labels));
    if (url.endsWith('vocab.txt')) return new Response(vocabulary);
    return new Response(new Uint8Array([0]));
  });
  // Force the selected edit to essentially 100% confidence, independent of
  // the real weights. The context guard must still reject a harmful change.
  runtime.run.mockImplementation(async (tokenIds: number[]) => {
    const data = new Float32Array(tokenIds.length * labels.length).fill(-20);
    for (const [position, id] of tokenIds.entries()) {
      const prediction = predictions[ids[Number(id)]] ?? 'KEEP';
      data[position * labels.length + labels.indexOf(prediction)] = 20;
    }
    return {data, dims: [1, tokenIds.length, labels.length]};
  });
});

async function correct(text: string): Promise<string> {
  const result = await analyzeModel(text, {preferWebGPU: false});
  expect(result.backend).toBe('wasm');
  return applySuggestions(text, result.suggestions);
}

describe('model verb safety at high confidence', () => {
  it('honors a disabled edit policy even with a caller threshold override', async () => {
    predictions.freind = 'REPLACE:friend';
    vi.stubGlobal('fetch', async (url: string) => {
      if (url.endsWith('manifest.json')) return new Response(JSON.stringify({confidenceThreshold: .85, maxSequenceLength: 8, disableModelEdits: true}));
      if (url.endsWith('labels.json')) return new Response(JSON.stringify(labels));
      if (url.endsWith('vocab.txt')) return new Response(vocabulary);
      return new Response(new Uint8Array([0]));
    });
    const result = await analyzeModel('A freind called.', {modelBaseUrl: '/disabled-policy/', preferWebGPU: false, confidenceThreshold: 0});
    expect(result.suggestions).toEqual([]);
    expect(result.backend).toBe('rules');
  });

  it('rejects an invalid edit policy', async () => {
    vi.stubGlobal('fetch', async (url: string) => {
      if (url.endsWith('manifest.json')) return new Response(JSON.stringify({confidenceThreshold: .85, maxSequenceLength: 8, disableModelEdits: 'false'}));
      if (url.endsWith('labels.json')) return new Response(JSON.stringify(labels));
      return new Response(vocabulary);
    });
    await expect(analyzeModel('A freind called.', {modelBaseUrl: '/invalid-policy/', preferWebGPU: false})).rejects.toThrow('Invalid model correction policy');
  });

  it.each([
    ['my cat is hungry.', 'is', 'are'],
    ['My dog is happy.', 'is', 'are'],
    ['The cat is sleeping.', 'is', 'are'],
    ['My cats are hungry.', 'are', 'is'],
    ['The news is good.', 'is', 'are'],
    ['My cat was hungry.', 'was', 'were'],
    ['My dogs were hungry.', 'were', 'was'],
    ['I am ready.', 'am', 'is'],
    ['I was hungry.', 'was', 'were'],
    ['The children have food.', 'have', 'has'],
  ])('preserves valid agreement in %s', async (text, original, replacement) => {
    predictions[original] = `REPLACE:${replacement}`;
    expect(await correct(text)).toBe(text);
  });

  it.each([
    ['My cat are hungry.', 'are', 'is', 'My cat is hungry.'],
    ['My cats is hungry.', 'is', 'are', 'My cats are hungry.'],
    ['The students has a notebook.', 'has', 'have', 'The students have a notebook.'],
    ['My dog were hungry.', 'were', 'was', 'My dog was hungry.'],
    ['I were hungry.', 'were', 'was', 'I was hungry.'],
    ['I is ready.', 'is', 'am', 'I am ready.'],
    ['My freind have food.', 'have', 'has', 'My freind has food.'],
    ['My friend go every morning.', 'go', 'goes', 'My friend goes every morning.'],
  ])('retains supported corrections in %s', async (text, original, replacement, target) => {
    predictions[original] = `REPLACE:${replacement}`;
    expect(await correct(text)).toBe(target);
  });

  it.each([
    ['Does my cat have food?', 'have', 'has'],
    ['I insist that he have a chance.', 'have', 'has'],
    ['I insist that\nhe have a chance.', 'have', 'has'],
    ['If I were you, I would wait.', 'were', 'was'],
    ['My cat and dog is hungry.', 'is', 'are'],
    ['The axolotl is hungry.', 'is', 'are'],
    ['My friend go every morning?', 'go', 'goes'],
    ['My friend go every morning\n?', 'go', 'goes'],
    ['My friend go home.', 'go', 'goes'],
    ['My friend read every morning.', 'read', 'reads'],
    ['My cat is hungry.', 'is', 'was'],
    ['My cat is hungry.', 'is', 'has'],
    ['My cat has food.', 'has', 'is'],
    ['My cat is hungry.', 'is', 'friend'],
    ['😀. My cat is hungry.', 'is', 'are'],
  ])('abstains in unsupported or contradicted contexts in %s', async (text, original, replacement) => {
    predictions[original] = `REPLACE:${replacement}`;
    expect(await correct(text)).toBe(text);
  });

  it('preserves the full source context across model window boundaries', async () => {
    const text = 'They know I insist that he have a chance.';
    predictions.have = 'REPLACE:has';
    expect(await correct(text)).toBe(text);
    const supported = 'An unknown sentence comes first. The students has a notebook.';
    predictions = {has: 'REPLACE:have'};
    expect(await correct(supported)).toBe('An unknown sentence comes first. The students have a notebook.');
  });

  it('preserves case and exact UTF-16 offsets for a supported model edit', async () => {
    const text = '😀. My cats IS hungry.';
    predictions.is = 'REPLACE:are';
    const {suggestions} = await analyzeModel(text, {preferWebGPU: false});
    expect(suggestions).toHaveLength(1);
    expect(suggestions[0]).toMatchObject({start: 12, end: 14, original: 'IS', replacement: 'ARE', source: 'model'});
    expect(applySuggestions(text, suggestions)).toBe('😀. My cats ARE hungry.');
  });

  it('continues to allow spelling edits', async () => {
    predictions.freind = 'REPLACE:friend';
    expect(await correct('A freind called.')).toBe('A friend called.');
  });

  it.each([
    ['We develop tecnologies.', 'tecnologies', 'the'],
    ['It will be useful.', 'be', 'because'],
    ['They left their coats there.', 'their', "they're"],
    ['Their coats are here.', 'their', "they're"],
    ['She have_value here.', 'have', 'has'],
    ['A freind-like person called.', 'freind', 'friend'],
    ['A freind_name called.', 'freind', 'friend'],
    ['A freindé called.', 'freind', 'friend'],
    ['It is a book.', 'a', 'the'],
    ['I read book reviews.', 'read', 'a'],
    ['We met Freind yesterday.', 'freind', 'friend'],
    ['A fReInD called.', 'freind', 'friend'],
  ])('rejects lexical hallucinations and partial-token edits in %s', async (text, original, replacement) => {
    predictions[original] = original === 'read' && replacement === 'a' ? 'APPEND:a' : `REPLACE:${replacement}`;
    expect(await correct(text)).toBe(text);
  });

  it.each([
    ['She has book.', 'has', 'a', 'She has a book.'],
    ['They have umbrella.', 'have', 'an', 'They have an umbrella.'],
  ])('keeps supported missing articles in %s', async (text, word, article, expected) => {
    predictions[word] = `APPEND:${article}`;
    expect(await correct(text)).toBe(expected);
  });

  it.each(['She has book reviews.', 'They have apple juice.', 'She has one ticket.', 'I have Apple.', 'They have\n\nBook.'])('preserves an already valid object phrase in %s', async text => {
    predictions.has = 'APPEND:a';
    predictions.have = 'APPEND:an';
    expect(await correct(text)).toBe(text);
  });

  it.each(['book book_reviews', 'foo_book book', 'book book-reviews', 'book, book', 'book\nbook'])('rejects partial-token or nonadjacent duplicate evidence in %s', async text => {
    predictions.book = 'DELETE';
    expect(await correct(text)).toBe(text);
  });

  it.each(['A book book.', 'A book book book.'])('keeps the first word when duplicate logits all request deletion in %s', async text => {
    predictions.book = 'DELETE';
    expect(await correct(text)).toBe('A book.');
  });
});
