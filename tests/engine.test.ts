import { describe, expect, it } from 'vite-plus/test';
import { applySuggestion, applySuggestions } from '../packages/engine/src/edits';
import { analyzeRules } from '../packages/engine/src/rules';
import { splitWords, WordPieceTokenizer } from '../packages/engine/src/tokenizer';
import type { Suggestion } from '../packages/engine/src/types';
import { validArticleEdit } from '../packages/engine/src/guards';

describe('correction safety', () => {
  it('fixes deterministic errors with exact offsets', () => {
    const text = '😀. She have a freind. We is writting.';
    expect(applySuggestions(text, analyzeRules(text))).toBe('😀. She has a friend. We are writing.');
  });
  it('preserves case and does not edit protected content', () => {
    const text = 'TEH `teh` https://example.test/teh teh@example.test';
    expect(applySuggestions(text, analyzeRules(text))).toBe('THE `teh` https://example.test/teh teh@example.test');
  });
  it('does not flag valid agreement or valid repeated constructions', () => {
    expect(analyzeRules('I am ready. They have books. She is here. She had had enough. I knew that that was fine. Does she have time? Would he have time? I insist that he have a chance.')).toEqual([]);
  });
  it('rejects stale and overlapping edits atomically', () => {
    const edits = analyzeRules('She have a freind.');
    expect(() => applySuggestion('She has a freind.', edits[0])).toThrow(/changed/u);
    expect(() => applySuggestions('She have a freind.', [edits[0], edits[0]])).toThrow(/Overlapping/u);
  });
  it('corrects simple habitual noun-subject agreement with exact UTF-16 offsets', () => {
    const text = '😀. My friend go to school every morning. Our teacher study every day.';
    expect(applySuggestions(text, analyzeRules(text))).toBe('😀. My friend goes to school every morning. Our teacher studies every day.');
  });
  it('leaves questions, plural/compound subjects, ambiguous tense and subjunctives alone', () => {
    expect(analyzeRules('Does my friend go every morning? My friend go every morning? My friends go every day. My friend and I go every day. I insist that my friend go every day. My friend read the paper every morning. My friend go home.')).toEqual([]);
  });
  it('does not edit habitual agreement inside protected code', () => {
    const text = '```\nMy friend go to school every morning.\n```';
    expect(analyzeRules(text)).toEqual([]);
  });
  it('inserts missing words without deleting neighbors', () => {
    const edit: Suggestion = {id: 'insert', start: 6, end: 6, original: '', replacement: ' a',
      message: '', category: 'grammar', confidence: .99, source: 'model'};
    expect(applySuggestion('I have book.', edit)).toBe('I have a book.');
  });
  it('rejects negative, NaN and oversized offsets', () => {
    const edit = analyzeRules('teh')[0];
    for (const start of [-1, NaN, Infinity, 100]) expect(() => applySuggestion('teh', {...edit, start})).toThrow();
  });
  it('rejects contradicted article predictions including misspelled heads and sound exceptions', () => {
    expect(validArticleEdit('an', 'freind')).toBe(false);
    expect(validArticleEdit('a', 'freind')).toBe(true);
    expect(validArticleEdit('an', 'hour')).toBe(true);
    expect(validArticleEdit('an', 'university')).toBe(false);
    expect(validArticleEdit('a', 'university')).toBe(true);
    expect(validArticleEdit('a', 'unknownword')).toBe(false);
  });
});

describe('token offsets and windowing', () => {
  const tokenizer = new WordPieceTokenizer('[PAD]\n[UNK]\n[CLS]\n[SEP]\ni\nhave\na\nbook\n.\nshe\n##s\n');
  it('tracks UTF-16 offsets after emoji and contractions', () => {
    expect(splitWords("😀 You're ready.").find(word => word.text === "You're")).toEqual({text: "You're", start: 3, end: 9});
  });
  it('respects the exact context budget without dropping ordinary words', () => {
    const chunks = tokenizer.chunks('I have a book. I have a book.', 6);
    expect(chunks.every(chunk => chunk.ids.length <= 6)).toBe(true);
    expect(chunks.flatMap(chunk => chunk.positions.map(p => p.word.text))).toEqual(['I', 'have', 'a', 'book', '.', 'I', 'have', 'a', 'book', '.']);
  });
  it('uses greedy wordpieces and unknown tokens', () => {
    expect(tokenizer.encodeWord('shes')).toEqual([9, 10]);
    expect(tokenizer.encodeWord('xyz')).toEqual([1]);
  });
});
