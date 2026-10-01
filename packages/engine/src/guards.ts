import { canonicalSpelling } from './rules';
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

export function validVerbEdit(text: string, word: WordToken, proposed: string): boolean {
  const originalFamily = agreementVerbs.get(word.text.toLowerCase());
  const proposedFamily = agreementVerbs.get(proposed.toLowerCase());
  if (!originalFamily && !proposedFamily) return true;
  // Agreement must not insert a verb in place of a noun, change its meaning,
  // or switch tense. Check the full source, never a truncated model window.
  if (!originalFamily || originalFamily !== proposedFamily) return false;
  const subject = text.slice(0, word.start).match(
    /(?:^|[.!?]\s+)(?:(i|you|we|they|he|she|it)|(?:my|your|our|his|her|their|the|a|an)[ \t]+([A-Za-z]+))[ \t]+$/iu);
  if (!subject) return false;
  const tail = text.slice(word.end);
  if (/^[^.!?]*\?/u.test(tail)) return false;
  const pronoun = subject[1]?.toLowerCase();
  const head = canonicalSpelling(subject[2] ?? '');
  let singular: boolean;
  if (pronoun) singular = ['he', 'she', 'it'].includes(pronoun);
  else if (singularHeads.has(head)) singular = true;
  else if (pluralHeads.has(head)) singular = false;
  else return false;
  if (originalFamily.habitual && (originalFamily.plural === 'read' ||
      !/^[^.!?\n]*\bevery[ \t]+(?:morning|day|evening|night|week)\b[^.!?\n]*(?:\.|$)/iu.test(tail))) return false;
  const expected = pronoun === 'i' && originalFamily.firstPerson
    ? originalFamily.firstPerson : singular ? originalFamily.singular : originalFamily.plural;
  return word.text.toLowerCase() !== expected && proposed.toLowerCase() === expected;
}
