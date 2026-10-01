import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vite-plus/test';
import { applySuggestion, applySuggestions } from '../packages/engine/src/edits';
import { analyzeRules } from '../packages/engine/src/rules';
import { spellingDistance, suggestSpelling } from '../packages/engine/src/spelling';

describe('local dictionary spelling', () => {
  it('suggests a greeting for standalone halo, without treating it as a misspelled noun', () => {
    for (const [text, replacement] of [['halo', 'hello'], [' Halo! ', 'Hello'], ['HALO', 'HELLO']]) {
      const edits = analyzeRules(text);
      expect(edits).toHaveLength(1);
      expect(edits[0]).toMatchObject({original: text.trim().replace('!', ''), replacement, source: 'dictionary'});
      expect(edits[0].confidence).toBeLessThan(1);
      expect(edits[0].message).toContain('valid word');
    }
    expect(analyzeRules('The moon has a halo. A halo surrounds the moon. Halo is a game.')).toEqual([]);
    expect(suggestSpelling('halo')).toBeUndefined();
  });

  it.each([
    ['helo', 'hello'], ['helllo', 'hello'], ['speling', 'spelling'],
    ['sentnce', 'sentence'], ['correcton', 'correction'], ['wrold', 'world'],
    ['langauge', 'language'], ['enviroment', 'environment'], ['acomodate', 'accommodate'],
    ['hte', 'the'], ['quik', 'quick'],
  ])('corrects %s to %s, including errors outside the old typo table', (word, replacement) => {
    expect(suggestSpelling(word)).toBe(replacement);
    expect(applySuggestions(word, analyzeRules(word))).toBe(replacement);
    // Repeated drafts must return the same result through the bounded cache.
    expect(suggestSpelling(word)).toBe(replacement);
  });

  it('leaves accepted words, ambiguous candidates and unfamiliar terms alone', () => {
    expect(analyzeRules('A clean sentence with colour, color, programme and program. We had had enough.')).toEqual([]);
    for (const word of ['wrod', 'qzxv', 'konkui', 'github', 'prisma', 'webgpu', 'vercel', 'kubectl']) {
      expect(suggestSpelling(word), word).toBeUndefined();
    }
  });

  it('preserves Unicode boundaries, identifiers, contractions, names and code', () => {
    const text = "Speling. My friend Speling uses spelingValue, speling_value, speling123, caféspeling and русскийspeling. Don't edit speling's or speling-like. NASA HTTP `speling` ```speling``` https://example.test/speling speling@example.test /speling speling.ts #speling";
    const edits = analyzeRules(text);
    expect(edits).toHaveLength(1);
    expect(edits[0]).toMatchObject({original: 'Speling', replacement: 'Spelling', start: 0, end: 7});
    expect(analyzeRules('caféteh русскийteh teh_value teh123')).toEqual([]);
    expect(analyzeRules('`speling')).toEqual([]);
    expect(analyzeRules('```\nspeling')).toEqual([]);
  });

  it('preserves casing, UTF-16 offsets and stale-edit rejection alongside existing rules', () => {
    const text = '😀. Speling matters. She have a freind and a sentnce.';
    const edits = analyzeRules(text);
    expect(edits.find(edit => edit.original === 'Speling')).toMatchObject({start: 4, end: 11, replacement: 'Spelling'});
    expect(applySuggestions(text, edits)).toBe('😀. Spelling matters. She has a friend and a sentence.');
    expect(() => applySuggestion(text.replace('Speling', 'Writing'), edits[0])).toThrow(/changed/u);
    expect(analyzeRules('recieve')).toHaveLength(1);
    expect(analyzeRules('recieve')[0].source).toBe('rule');
  });

  it('bounds token length and skips unsupported scripts', () => {
    for (const word of ['', 'ab', 'a'.repeat(10000), 'speling42', 'speling_value', 'café']) {
      expect(suggestSpelling(word)).toBeUndefined();
    }
  });

  it('keeps generated data tied to its pinned source and transformation', () => {
    const provenance = JSON.parse(readFileSync(new URL('../packages/engine/dictionary-provenance.json', import.meta.url), 'utf8'));
    const generated = readFileSync(new URL('../packages/engine/src/dictionary.generated.ts', import.meta.url));
    expect(createHash('sha256').update(generated).digest('hex')).toBe(provenance.generatedSha256);
    expect(provenance.acceptedWords).toBe(82769);
    expect(provenance.candidateWords).toBe(20000);
  });
});

describe('bounded spelling edit distance', () => {
  // Independent full-matrix reference verifies pruning against exact OSA distance.
  function reference(a: string, b: string): number {
    const rows = Array.from({length: a.length + 1}, (_, i) => Array.from({length: b.length + 1}, (_, j) => i ? j ? 0 : i : j));
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) {
      rows[i][j] = Math.min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + Number(a[i - 1] !== b[j - 1]));
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) rows[i][j] = Math.min(rows[i][j], rows[i - 2][j - 2] + 1);
    }
    return rows[a.length][b.length];
  }
  it('matches the reference on insertions, deletions, substitutions, repeats and transpositions', () => {
    const words = ['', 'a', 'b', 'ab', 'ba', 'abc', 'cab', 'aab', 'abab', 'baba', 'hello', 'helo', 'hlllo', 'world', 'wrold'];
    for (const a of words) for (const b of words) for (const limit of [0, 1, 2]) {
      expect(spellingDistance(a, b, limit), `${a}, ${b}, ${limit}`).toBe(Math.min(reference(a, b), limit + 1));
    }
  });
});
