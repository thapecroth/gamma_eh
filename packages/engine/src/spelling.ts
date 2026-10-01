import { dictionaryCounts, dictionaryWords } from './dictionary.generated';
import { preserveCase } from './edits';
import type { Suggestion } from './types';

// Two deletes of a seven-character prefix produce at most 29 lookup keys.
// Only common words are candidates; the entire vocabulary protects valid words.
const PREFIX_LENGTH = 7;
const MAX_WORD_LENGTH = 32;
const CACHE_SIZE = 2048;
interface Dictionary {
  known: Set<string>;
  words: string[];
  counts: number[];
  deletes?: Map<string, number | number[]>;
}
let dictionary: Dictionary | undefined;
const cache = new Map<string, string | null>();
// Current tools often postdate the frequency corpus. Do not turn their names
// into nearby ordinary words (for example, prisma -> prism).
const technicalWords = new Set('github gitlab webgpu webgl wasm typescript javascript nodejs npm pnpm vite vitest esbuild eslint prisma vercel kubectl chatgpt'.split(' '));

export function spellingWords(text: string): RegExpStringIterator<RegExpExecArray> {
  return text.matchAll(/[\p{L}\p{M}\p{N}\p{Pc}\u200C\u200D]+(?:['’\u2010\u2011-][\p{L}\p{M}\p{N}\p{Pc}\u200C\u200D]+)*/gu);
}

function getDictionary(): Dictionary {
  if (!dictionary) {
    const words = dictionaryWords.split(' ');
    const counts = dictionaryCounts.split(' ').map(Number);
    dictionary = {known: new Set(words), words: words.slice(0, counts.length), counts};
  }
  return dictionary;
}

function deleteKeys(word: string, distance: number): Set<string> {
  const prefix = word.slice(0, PREFIX_LENGTH);
  const keys = new Set([prefix]);
  let level = [prefix];
  for (let depth = 0; depth < distance; depth++) {
    const next: string[] = [];
    for (const value of level) {
      for (let i = 0; i < value.length; i++) {
        const deleted = value.slice(0, i) + value.slice(i + 1);
        if (!keys.has(deleted)) { keys.add(deleted); next.push(deleted); }
      }
    }
    level = next;
  }
  return keys;
}

function getIndex(data: Dictionary): Map<string, number | number[]> {
  if (!data.deletes) {
    const index = new Map<string, number | number[]>();
    for (const [id, word] of data.words.entries()) {
      if (word.length < 3 || word.length > MAX_WORD_LENGTH) continue;
      for (const key of deleteKeys(word, 2)) {
        const previous = index.get(key);
        if (previous === undefined) index.set(key, id);
        else if (typeof previous === 'number') index.set(key, [previous, id]);
        else previous.push(id);
      }
    }
    data.deletes = index;
  }
  return data.deletes;
}

/** Bounded optimal-string-alignment distance: insertion, deletion, substitution, adjacent transposition. */
export function spellingDistance(a: string, b: string, limit: number): number {
  if (Math.abs(a.length - b.length) > limit) return limit + 1;
  let previousPrevious: number[] = [];
  let previous = Array.from({length: b.length + 1}, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const current = Array.from({length: b.length + 1}, () => limit + 1);
    current[0] = i;
    let minimum = current[0];
    for (let j = Math.max(1, i - limit); j <= Math.min(b.length, i + limit); j++) {
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1,
        previous[j - 1] + Number(a[i - 1] !== b[j - 1]));
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        current[j] = Math.min(current[j], previousPrevious[j - 2] + 1);
      }
      minimum = Math.min(minimum, current[j]);
    }
    if (minimum > limit) return limit + 1;
    previousPrevious = previous;
    previous = current;
  }
  return Math.min(previous[b.length], limit + 1);
}

export function suggestSpelling(word: string): string | undefined {
  if (!/^[a-z]{3,32}$/u.test(word)) return;
  const data = getDictionary();
  if (data.known.has(word) || technicalWords.has(word)) return;
  if (cache.has(word)) return cache.get(word) ?? undefined;
  const limit = word.length < 5 ? 1 : 2;
  const index = getIndex(data);
  const candidates = new Set<number>();
  for (const key of deleteKeys(word, limit)) {
    const ids = index.get(key);
    if (typeof ids === 'number') candidates.add(ids);
    else if (ids) for (const id of ids) candidates.add(id);
  }
  let bestDistance = limit + 1;
  let best = -1;
  let runnerUp = -1;
  let bestRepeat = false;
  for (const id of candidates) {
    const distance = spellingDistance(word, data.words[id], Math.min(limit, bestDistance));
    if (distance > limit) continue;
    // Missing a doubled letter is a common typo: prefer helo -> hello over
    // helo -> help. Keep distance primary; apply this only to a single edit.
    const candidate = data.words[id];
    const repeat = distance === 1 && candidate.length === word.length + 1 &&
      [...candidate.matchAll(/(.)\1/gu)].some(match => candidate.slice(0, match.index) + candidate.slice(match.index + 1) === word);
    if (distance < bestDistance || (distance === bestDistance && repeat && !bestRepeat)) {
      bestDistance = distance; best = id; runnerUp = -1; bestRepeat = repeat;
    }
    else if (distance === bestDistance) {
      if (repeat !== bestRepeat) continue;
      if (id < best) { runnerUp = best; best = id; }
      else if (runnerUp === -1 || id < runnerUp) runnerUp = id;
    }
  }
  // Abstain when equally close candidates have similar frequency, especially
  // on short words. Frequency is a ranking heuristic, not grammar confidence.
  const margin = bestDistance === 2 ? 4 : 2;
  const replacement = best >= 0 && (runnerUp < 0 || data.counts[best] >= margin * data.counts[runnerUp])
    ? data.words[best] : null;
  if (cache.size >= CACHE_SIZE) cache.delete(cache.keys().next().value!);
  cache.set(word, replacement);
  return replacement ?? undefined;
}

export function analyzeSpelling(text: string, protectedRanges: Array<{start: number; end: number}>): Suggestion[] {
  const suggestions: Suggestion[] = [];
  // Consume entire Unicode tokens, contractions and compounds before filtering
  // for plain English. This prevents editing pieces of café, identifiers or names.
  for (const match of spellingWords(text)) {
    const original = match[0];
    const start = match.index;
    const end = start + original.length;
    if (!/^[A-Za-z]{3,32}$/u.test(original) ||
        protectedRanges.some(range => start < range.end && range.start < end)) continue;
    const word = original.toLowerCase();
    const greeting = word === 'halo' && /^\s*halo[.!?]*\s*$/iu.test(text);
    // Mixed case, acronyms and capitalized names inside prose are left alone.
    if (!greeting && (original === original.toUpperCase() ||
        !/^[A-Z]?[a-z]+$/u.test(original) ||
        (/^[A-Z]/u.test(original) && !/(?:^|[.!?]\s*)$/u.test(text.slice(0, start).trimEnd())))) continue;
    const replacement = greeting ? 'hello' : suggestSpelling(word);
    if (!replacement) continue;
    suggestions.push({id: `dictionary-${start}-${end}-${replacement}`, start, end, original,
      replacement: preserveCase(original, replacement), category: 'spelling', source: 'dictionary',
      confidence: greeting ? 0.65 : 0.9,
      message: greeting ? 'Did you mean “hello” as a greeting? “Halo” is also a valid word.' : 'Check the spelling of this word.'});
  }
  return suggestions;
}
