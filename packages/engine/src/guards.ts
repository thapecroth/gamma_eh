import { canonicalSpelling } from './rules';
import { isKnownSpellingWord, spellingDistance } from './spelling';
import type { WordToken } from './tokenizer';

// The v1 model knows a narrow article vocabulary. Reject neural article edits
// outside known sound classes; first-letter heuristics fail for hour/university.
const articleHeads: Record<string, 'a' | 'an'> = {
  apple: 'an', orange: 'an', egg: 'an', umbrella: 'an', envelope: 'an', idea: 'an',
  hour: 'an', honest: 'an', honor: 'an', heir: 'an',
  book: 'a', pencil: 'a', bicycle: 'a', camera: 'a', notebook: 'a', ticket: 'a',
  friend: 'a', message: 'a', pen: 'a', project: 'a', library: 'a',
  university: 'a', user: 'a', unicorn: 'a', european: 'a', one: 'a',
};

export function validArticleEdit(article: string, nextWord: string): boolean {
  return (article === 'a' || article === 'an') && articleHeads[canonicalSpelling(nextWord)] === article;
}

interface VerbFamily { singular: string; plural: string; firstPerson?: string; habitual?: boolean }
const verbFamilies: VerbFamily[] = [
  {singular: 'is', plural: 'are', firstPerson: 'am'},
  {singular: 'was', plural: 'were', firstPerson: 'was'},
  {singular: 'has', plural: 'have'},
  ...[
    ['work', 'works'], ['walk', 'walks'], ['read', 'reads'], ['write', 'writes'],
    ['play', 'plays'], ['learn', 'learns'], ['travel', 'travels'], ['cook', 'cooks'],
    ['wait', 'waits'], ['talk', 'talks'], ['sleep', 'sleeps'], ['run', 'runs'],
    ['study', 'studies'], ['watch', 'watches'], ['go', 'goes'],
  ].map(([plural, singular]) => ({singular, plural, habitual: true})),
];
const agreementVerbs = new Map(verbFamilies.flatMap(family =>
  [family.singular, family.plural, ...(family.firstPerson ? [family.firstPerson] : [])]
    .map(verb => [verb, family] as const)));
// Deliberately bounded: neither a final "s" nor the model's confidence proves
// subject number. Unknown heads, modifiers and compound subjects abstain.
const singularHeads = new Set(['friend', 'teacher', 'neighbor', 'student', 'child', 'colleague', 'manager', 'cat', 'dog', 'news']);
const pluralHeads = new Set(['friends', 'teachers', 'neighbors', 'students', 'children', 'colleagues', 'managers', 'cats', 'dogs', 'people']);

function expectedVerb(text: string, word: WordToken, family: VerbFamily): string | undefined {
  const subject = text.slice(0, word.start).match(
    /(?:^|[.!?]\s+)(?:(i|you|we|they|he|she|it)|(?:my|your|our|his|her|their|the|a|an)[ \t]+([A-Za-z]+))[ \t]+$/iu);
  if (!subject) return;
  const tail = text.slice(word.end);
  if (/^[^.!?]*\?/u.test(tail)) return;
  const pronoun = subject[1]?.toLowerCase();
  const head = canonicalSpelling(subject[2] ?? '');
  let singular: boolean;
  if (pronoun) singular = ['he', 'she', 'it'].includes(pronoun);
  else if (singularHeads.has(head)) singular = true;
  else if (pluralHeads.has(head)) singular = false;
  else return;
  if (family.habitual && (family.plural === 'read' ||
      !/^[^.!?\n]*\bevery[ \t]+(?:morning|day|evening|night|week)\b[^.!?\n]*(?:\.|$)/iu.test(tail))) return;
  return pronoun === 'i' && family.firstPerson
    ? family.firstPerson : singular ? family.singular : family.plural;
}

export function validVerbEdit(text: string, word: WordToken, proposed: string): boolean {
  const originalFamily = agreementVerbs.get(word.text.toLowerCase());
  const proposedFamily = agreementVerbs.get(proposed.toLowerCase());
  if (!originalFamily && !proposedFamily) return true;
  // Agreement must not insert a verb in place of a noun, change its meaning,
  // or switch tense. Check the full source, never a truncated model window.
  if (!originalFamily || originalFamily !== proposedFamily) return false;
  const expected = expectedVerb(text, word, originalFamily);
  return word.text.toLowerCase() !== expected && proposed.toLowerCase() === expected;
}

const countableArticleObjects = new Set('apple orange egg umbrella envelope idea hour heir book pencil bicycle camera notebook ticket friend message pen project library university user unicorn'.split(' '));

export function validAppendEdit(text: string, word: WordToken, proposed: string, nextWord?: WordToken): boolean {
  const family = agreementVerbs.get(word.text.toLowerCase());
  return !!family && !!nextWord && family.singular === 'has' && word.text.toLowerCase() === expectedVerb(text, word, family) &&
    nextWord.text === nextWord.text.toLowerCase() && /^[ \t]+$/u.test(text.slice(word.end, nextWord.start)) &&
    countableArticleObjects.has(canonicalSpelling(nextWord.text)) && validArticleEdit(proposed, nextWord.text) &&
    /^[ \t]*(?:[.!?](?:\s|$)|$)/u.test(text.slice(nextWord.end));
}

export function validReplacementEdit(text: string, word: WordToken, proposed: string, nextWord: string): boolean {
  const original = word.text.toLowerCase();
  const replacement = proposed.toLowerCase();
  if (agreementVerbs.has(original) || agreementVerbs.has(replacement)) return validVerbEdit(text, word, replacement);
  if (['a', 'an'].includes(original) || ['a', 'an'].includes(replacement)) {
    return ['a', 'an'].includes(original) && original !== replacement && validArticleEdit(replacement, nextWord);
  }
  // A finite-label classifier cannot infer arbitrary rewrites from confidence.
  // Known words (including homophones) need linguistic evidence unavailable in
  // this baseline. Lexical replacements are bounded spelling corrections only.
  if (!/^[A-Z]?[a-z]+$/u.test(word.text) ||
      (/^[A-Z]/u.test(word.text) && !/(?:^|[.!?]\s*)$/u.test(text.slice(0, word.start).trimEnd()))) return false;
  const canonical = canonicalSpelling(original);
  if (canonical !== original) return canonical === replacement;
  if (!/^[a-z]{3,32}$/u.test(original) || !/^[a-z]{3,32}$/u.test(replacement) ||
      isKnownSpellingWord(original) || !isKnownSpellingWord(replacement)) return false;
  const limit = original.length < 5 ? 1 : 2;
  return spellingDistance(original, replacement, limit) <= limit;
}
