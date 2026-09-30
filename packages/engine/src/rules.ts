import { preserveCase } from './edits';
import type { Suggestion, SuggestionCategory } from './types';

const spelling: Record<string, string> = {
  recieve: 'receive', recieved: 'received', recieving: 'receiving',
  beleive: 'believe', beleived: 'believed', freinds: 'friends',
  seperate: 'separate', neccessary: 'necessary',
  definately: 'definitely', tommorow: 'tomorrow', becuase: 'because', freind: 'friend',
  writting: 'writing', grammer: 'grammar', thier: 'their', adress: 'address',
  avaliable: 'available', realy: 'really', succesful: 'successful', diffrent: 'different',
  peopel: 'people', buisness: 'business', importent: 'important', teh: 'the',
};

function spellingCorrection(lowercaseWord: string): string | undefined {
  return Object.prototype.hasOwnProperty.call(spelling, lowercaseWord)
    ? spelling[lowercaseWord] : undefined;
}

// A bounded fallback for simple singular subjects in explicitly habitual
// present-tense statements. Exclude ambiguous past forms such as "read" and
// require the time cue so questions, commands, and subjunctives stay untouched.
const habitualVerbs: Record<string, string> = {
  go: 'goes', work: 'works', walk: 'walks', write: 'writes', play: 'plays', learn: 'learns',
  travel: 'travels', cook: 'cooks', wait: 'waits', talk: 'talks', sleep: 'sleeps',
  run: 'runs', study: 'studies', watch: 'watches',
};
const habitualAgreement = new RegExp(
  `(^|[.!?]\\s+)((?:my|your|our|his|her|their|the|a)[ \\t]+(?:friend|teacher|neighbor|student|child|colleague|manager))([ \\t]+)(${Object.keys(habitualVerbs).join('|')})\\b(?=[^.!?\\n]*\\bevery[ \\t]+(?:morning|day|evening|night|week)\\b[^.!?\\n]*(?:\\.|$))`, 'giu');

export function canonicalSpelling(word: string): string {
  const lowercaseWord = word.toLowerCase();
  return spellingCorrection(lowercaseWord) ?? lowercaseWord;
}

// Protect URLs, email addresses, inline/fenced code from deterministic edits.
export function protectedSpans(text: string): Array<{start: number; end: number}> {
  return [...text.matchAll(/```[\s\S]*?```|`[^`\n]+`|https?:\/\/\S+|\b[^\s@]+@[^\s@]+\.[^\s@]+/gu)]
    .map(match => ({ start: match.index, end: match.index + match[0].length }));
}

export function analyzeRules(text: string): Suggestion[] {
  const suggestions: Suggestion[] = [];
  const protectedRanges = protectedSpans(text);
  const add = (start: number, end: number, replacement: string, message: string, category: SuggestionCategory) => {
    if (protectedRanges.some(range => start < range.end && range.start < end)) return;
    if (suggestions.some(edit => start < edit.end && edit.start < end)) return;
    suggestions.push({ id: `rule-${start}-${end}-${replacement}`, start, end,
      original: text.slice(start, end), replacement, message, category, confidence: 1, source: 'rule' });
  };
  for (const match of text.matchAll(/\b[A-Za-z]+\b/gu)) {
    const replacement = spellingCorrection(match[0].toLowerCase());
    if (replacement) add(match.index, match.index + match[0].length,
      preserveCase(match[0], replacement), 'Check the spelling of this word.', 'spelling');
  }
  // Restrict rules to sentence-initial subjects. "Does she have" and "I insist
  // that he have" must not become "has"; model suggestions handle other contexts.
  for (const match of text.matchAll(/(^|[.!?]\s+)(I|he|she|it|we|they|you)([ \t]+)(is|are|am|has|have)\b/giu)) {
    const subject = match[2].toLowerCase();
    const verb = match[4].toLowerCase();
    const singular = ['he', 'she', 'it'].includes(subject);
    let replacement: string | undefined;
    if (['is', 'are', 'am'].includes(verb)) {
      const expected = subject === 'i' ? 'am' : singular ? 'is' : 'are';
      if (expected !== verb) replacement = expected;
    } else {
      const expected = singular ? 'has' : 'have';
      if (expected !== verb) replacement = expected;
    }
    if (replacement) {
      const start = match.index + match[1].length + match[2].length + match[3].length;
      add(start, start + match[4].length, preserveCase(match[4], replacement),
        'Match the verb to its subject.', 'grammar');
    }
  }
  for (const match of text.matchAll(/\b([A-Za-z]+)([ \t]+)\1\b/giu)) {
    if (['had', 'that'].includes(match[1].toLowerCase())) continue;
    const start = match.index + match[1].length;
    add(start, match.index + match[0].length, '', 'This word appears twice in a row.', 'grammar');
  }
  for (const match of text.matchAll(habitualAgreement)) {
    const start = match.index + match[1].length + match[2].length + match[3].length;
    add(start, start + match[4].length, preserveCase(match[4], habitualVerbs[match[4].toLowerCase()]),
      'Match the present-tense verb to its singular subject.', 'grammar');
  }
  return suggestions.sort((a, b) => a.start - b.start);
}
