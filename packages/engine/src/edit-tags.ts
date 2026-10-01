import { preserveCase } from './edits';
import { validArticleEdit, validVerbEdit } from './guards';
import { splitWords, type WordToken } from './tokenizer';

const wordPattern = /^[A-Za-z]+(?:['’][A-Za-z]+)*$/u;
const punctuationPattern = /^[.,!?;:()[\]{}'"-]$/u;
const functionWords = new Set('a an the to of in on at for from with is are am was were be been being has have do does did'.split(' '));

export function renderTokens(words: string[]): string {
  return words.join(' ').replace(/\s+([.,!?;:)\]}])/gu, '$1').replace(/([([{])\s+/gu, '$1');
}

function payloadWords(value: string): string[] {
  const words = splitWords(value).map(word => word.text);
  if (!words.length || words.length > 4 || renderTokens(words) !== value) throw new Error('Invalid exact edit payload');
  return words;
}

function suffixTransform(word: string, operation: string): string {
  if (!/^[A-Za-z]+$/u.test(word)) throw new Error('Suffix edits require an English word');
  const lower = word.toLowerCase();
  let value: string;
  if (operation === 'ADD_S') value = lower + 's';
  else if (operation === 'ADD_ES') value = lower + 'es';
  else if (operation === 'REMOVE_S' && lower.endsWith('s') && lower.length > 1) value = lower.slice(0, -1);
  else if (operation === 'REMOVE_ES' && lower.endsWith('es') && lower.length > 2) value = lower.slice(0, -2);
  else if (operation === 'Y_TO_IES' && lower.endsWith('y') && lower.length > 1) value = lower.slice(0, -1) + 'ies';
  else if (operation === 'IES_TO_Y' && lower.endsWith('ies') && lower.length > 3) value = lower.slice(0, -3) + 'y';
  else throw new Error('Invalid suffix edit');
  return preserveCase(word, value);
}

export function decodeWord(word: string, tag: string, schema: number): string[] {
  if (tag === 'KEEP') return [word];
  if (tag === 'DELETE') return [];
  if (tag.startsWith('REPLACE:')) return [preserveCase(word, tag.slice(8))];
  if (tag.startsWith('APPEND:')) return [word, tag.slice(7)];
  if (schema !== 2) throw new Error('Unsupported edit tag');
  if (tag.startsWith('REPLACE_EXACT:')) return payloadWords(tag.slice(14));
  if (tag.startsWith('APPEND_EXACT:')) return [word, ...payloadWords(tag.slice(13))];
  if (tag.startsWith('PREPEND_EXACT:')) return [...payloadWords(tag.slice(14)), word];
  if (tag === 'CASE:LOWER') return [word.toLowerCase()];
  if (tag === 'CASE:UPPER') return [word.toUpperCase()];
  if (tag === 'CASE:TITLE') return [word.slice(0, 1).toUpperCase() + word.slice(1).toLowerCase()];
  if (tag.startsWith('SUFFIX:')) return [suffixTransform(word, tag.slice(7))];
  throw new Error('Unsupported edit tag');
}

export function tagCategory(tag: string): string {
  if (tag === 'KEEP') return 'keep';
  if (tag === 'DELETE') return 'delete';
  if (tag.startsWith('CASE:')) return 'case';
  if (tag.startsWith('SUFFIX:')) return 'suffix';
  const payload = tag.slice(tag.indexOf(':') + 1);
  if (payload && !/[\p{L}\p{N}_]/u.test(payload)) return 'punctuation';
  if (tag.startsWith('APPEND:') || tag.startsWith('APPEND_EXACT:')) return 'append';
  if (tag.startsWith('PREPEND_EXACT:')) return 'prepend';
  return 'replace';
}

export interface DecodedEdit { start: number; end: number; replacement: string }

function validPayloadArticles(tokens: string[], nextWord: string): boolean {
  return tokens.every((token, index) => !['a', 'an'].includes(token.toLowerCase()) ||
    validArticleEdit(token.toLowerCase(), tokens[index + 1] ?? nextWord));
}

export function decodeProposal(text: string, words: WordToken[], index: number, tag: string, confidence: number, schema: number): DecodedEdit | undefined {
  const word = words[index];
  if (!word || (!wordPattern.test(word.text) && !(schema === 2 && punctuationPattern.test(word.text)))) return;
  if (tag === 'KEEP') return;
  let {start, end} = word;
  let replacement: string;
  const nextWord = words[index + 1]?.text ?? '';
  if (tag === 'DELETE') {
    const lower = word.text.toLowerCase();
    const duplicate = wordPattern.test(word.text) && !['had', 'that'].includes(lower) &&
      (words[index - 1]?.text.toLowerCase() === lower || nextWord.toLowerCase() === lower);
    const punctuation = punctuationPattern.test(word.text);
    if (!duplicate && !(schema === 2 && confidence >= 0.98 && (punctuation || functionWords.has(lower)))) return;
    if (punctuation) {
      replacement = /[\p{L}\p{N}]/u.test(text[start - 1] ?? '') && /[\p{L}\p{N}]/u.test(text[end] ?? '') ? ' ' : '';
    } else {
      if (/[ \t]/u.test(text[end] ?? '')) { while (/[ \t]/u.test(text[end] ?? '')) end++; }
      else { while (start > 0 && /[ \t]/u.test(text[start - 1])) start--; }
      replacement = '';
    }
  } else {
    let decoded: string[];
    try { decoded = decodeWord(word.text, tag, schema); } catch { return; }
    if (tag.startsWith('APPEND:')) {
      const payload = tag.slice(7);
      if (['a', 'an'].includes(payload) && !validArticleEdit(payload, nextWord)) return;
      start = end;
      replacement = ' ' + payload;
    } else if (tag.startsWith('APPEND_EXACT:')) {
      const payload = tag.slice(13);
      if (!validPayloadArticles(decoded.slice(1), nextWord)) return;
      start = end;
      const prefix = /^[.,!?;:)\]}]/u.test(payload) ? '' : ' ';
      const suffix = text[end] && !/[\s.,!?;:)\]}]/u.test(text[end]) ? ' ' : '';
      replacement = prefix + payload + suffix;
    } else if (tag.startsWith('PREPEND_EXACT:')) {
      end = start;
      const payload = tag.slice(14);
      if (!validPayloadArticles(decoded.slice(0, -1), word.text)) return;
      replacement = payload + (/[([{]$/u.test(payload) ? '' : ' ');
    } else {
      replacement = renderTokens(decoded);
      if (word.text.toLowerCase() !== replacement.toLowerCase() && !validVerbEdit(text, word, replacement)) return;
      if (!validPayloadArticles(decoded, nextWord)) return;
      if (punctuationPattern.test(word.text)) {
        // Replacing a punctuation anchor with a word must not join its neighbors.
        if (/^[\p{L}\p{N}]/u.test(replacement) && /[\p{L}\p{N}]$/u.test(text.slice(0, start))) replacement = ' ' + replacement;
        if (/[\p{L}\p{N}]$/u.test(replacement) && /^[\p{L}\p{N}]/u.test(text.slice(end))) replacement += ' ';
      }
    }
  }
  if (text.slice(start, end) === replacement) return;
  return {start, end, replacement};
}
