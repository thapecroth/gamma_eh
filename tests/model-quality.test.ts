import { describe, expect, it } from 'vite-plus/test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import parity from '../training/edit-parity.json';
import tokenizerParity from '../training/tokenizer-parity.json';
import { EditHistory } from '../packages/engine/src/edit-history';
import { decodeProposal, decodeWord } from '../packages/engine/src/edit-tags';
import { applySuggestions } from '../packages/engine/src/edits';
import { splitWords, WordPieceTokenizer } from '../packages/engine/src/tokenizer';
import type { Suggestion } from '../packages/engine/src/types';

function edit(text: string, start: number, end: number, replacement: string): Suggestion {
  return {id: 'test', start, end, original: text.slice(start, end), replacement,
    message: 'Test', category: 'grammar', confidence: .99, source: 'model'};
}

function corrected(text: string, index: number, tag: string, confidence = .99, schema = 2): string {
  const proposal = decodeProposal(text, splitWords(text), index, tag, confidence, schema);
  return proposal ? applySuggestions(text, [edit(text, proposal.start, proposal.end, proposal.replacement)]) : text;
}

describe('versioned edit contract', () => {
  it('reproduces the pinned Python tokenizer on multilingual context and contractions', () => {
    const vocabulary = readFileSync(new URL('../models/browser/vocab.txt', import.meta.url), 'utf8');
    expect(createHash('sha256').update(vocabulary).digest('hex')).toBe(tokenizerParity.vocab_sha256);
    const tokenizer = new WordPieceTokenizer(vocabulary);
    for (const {word, ids} of tokenizerParity.cases) expect(tokenizer.encodeWord(word)).toEqual(ids);
  });
  it('matches uncased BERT accent stripping, Chinese isolation, and Unicode character lengths', () => {
    const tokenizer = new WordPieceTokenizer('[PAD]\n[UNK]\n[CLS]\n[SEP]\ncafe\na\nb\n中\n文\n_\n𠮷\n');
    expect(tokenizer.encodeWord('Café')).toEqual([4]);
    expect(tokenizer.encodeWord('a中文b')).toEqual([5, 7, 8, 6]);
    expect(tokenizer.encodeWord('a_b')).toEqual([5, 9, 6]);
    expect(tokenizer.encodeWord('𠮷')).toEqual([10]);
    expect(splitWords('😀a中文b')[1]).toEqual({text: 'a中文b', start: 2, end: 6});
  });
  it('uses the same bounded edit vocabulary as the Python trainer', () => {
    for (const {word, tag, tokens} of parity.cases) expect(decodeWord(word, tag, parity.schema)).toEqual(tokens);
    for (const {word, tag} of parity.invalid) expect(() => decodeWord(word, tag, parity.schema)).toThrow();
    expect(() => decodeWord('hello', 'CASE:TITLE', 1)).toThrow();
  });

  it('matches full-source verb abstention for legacy and richer edits', () => {
    for (const row of parity.guarded) expect(corrected(row.source, row.index, row.tag)).toBe(row.target);
  });

  it('applies exact case, morphology, and multiple words with UTF-16 offsets', () => {
    expect(corrected('😀 hello.', 1, 'CASE:TITLE')).toBe('😀 Hello.');
    expect(corrected('She study every day.', 1, 'SUFFIX:Y_TO_IES')).toBe('She studies every day.');
    expect(corrected('We visited York.', 2, 'REPLACE_EXACT:New York')).toBe('We visited New York.');
    expect(corrected('Book.', 0, 'REPLACE_EXACT:A book')).toBe('A book.');
  });

  it('inserts punctuation and prefixes without consuming source whitespace', () => {
    expect(corrected('Hello world.', 0, 'APPEND_EXACT:,')).toBe('Hello, world.');
    expect(corrected('Hello\tworld.', 0, 'APPEND_EXACT:,')).toBe('Hello,\tworld.');
    expect(corrected('hello.', 0, 'PREPEND_EXACT:Well,')).toBe('Well, hello.');
    expect(corrected('hello.', 0, 'PREPEND_EXACT:(')).toBe('(hello.');
  });

  it('guards deletions, sound classes, numbers, emoji, and unsupported symbols', () => {
    expect(corrected('Hello,world.', 1, 'DELETE')).toBe('Hello world.');
    expect(corrected('Hello, world.', 1, 'DELETE')).toBe('Hello world.');
    expect(corrected('We read books.', 2, 'DELETE')).toBe('We read books.');
    expect(corrected('I have the book.', 2, 'DELETE', .979)).toBe('I have the book.');
    expect(corrected('I have the book.', 2, 'DELETE')).toBe('I have book.');
    expect(corrected('She had had a book.', 1, 'DELETE')).toBe('She had had a book.');
    expect(corrected('I knew that that was fine.', 2, 'DELETE')).toBe('I knew that that was fine.');
    expect(corrected('I have a university.', 2, 'REPLACE_EXACT:an')).toBe('I have a university.');
    expect(corrected('I have a hour.', 2, 'REPLACE_EXACT:an')).toBe('I have an hour.');
    expect(corrected('Book.', 0, 'PREPEND_EXACT:An')).toBe('Book.');
    expect(corrected('Hour.', 0, 'PREPEND_EXACT:An')).toBe('An Hour.');
    for (const text of ['42', '😀', '$']) expect(corrected(text, 0, 'REPLACE_EXACT:word')).toBe(text);
  });
});

describe('composed correction offsets', () => {
  it('composes a correction to inserted text into the original insertion', () => {
    const original = '😀. I have book.';
    const history = new EditHistory(original);
    const anchor = original.indexOf('book');
    history.apply([edit(history.text, anchor, anchor, 'an ')]);
    history.apply([edit(history.text, anchor, anchor + 2, 'a')]);
    expect(history.text).toBe('😀. I have a book.');
    expect(applySuggestions(original, history.suggestions())).toBe(history.text);
    expect(history.suggestions()[0]).toMatchObject({start: anchor, end: anchor, original: '', replacement: 'a '});
    expect(() => applySuggestions('stale ' + original, history.suggestions())).toThrow(/changed/u);
  });

  it('combines replacements, adjacent insertions, deletion, and net undo', () => {
    const original = 'bad cat.';
    const history = new EditHistory(original);
    history.apply([edit(history.text, 0, 3, 'good'), edit(history.text, 4, 7, 'dog')]);
    history.apply([edit(history.text, 4, 4, ' little')]);
    expect(applySuggestions(original, history.suggestions())).toBe('good little dog.');
    history.apply([edit(history.text, 0, 11, 'bad ')]);
    expect(applySuggestions(original, history.suggestions())).toBe(history.text);
    const undo = new EditHistory('book');
    undo.apply([edit(undo.text, 0, 4, 'books')]);
    undo.apply([edit(undo.text, 0, 5, 'book')]);
    expect(undo.suggestions()).toEqual([]);
  });

  it('reconstructs bounded edit sequences against the untouched original', () => {
    let seed = 42;
    const random = (bound: number) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % bound; };
    for (let trial = 0; trial < 100; trial++) {
      const original = '😀ab cd! ef';
      const history = new EditHistory(original);
      for (let stage = 0; stage < 10; stage++) {
        const start = random(history.text.length + 1);
        const end = start + random(history.text.length - start + 1);
        const replacement = ['', 'x', '😀', 'a b'][random(4)];
        history.apply([edit(history.text, start, end, replacement)]);
        expect(applySuggestions(original, history.suggestions())).toBe(history.text);
      }
    }
  });
});
