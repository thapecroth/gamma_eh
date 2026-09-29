import type { Suggestion } from './types';

function isValid(text: string, edit: Suggestion): boolean {
  return Number.isInteger(edit.start) && Number.isInteger(edit.end) &&
    edit.start >= 0 && edit.end >= edit.start && edit.end <= text.length &&
    text.slice(edit.start, edit.end) === edit.original;
}

export function applySuggestion(text: string, suggestion: Suggestion): string {
  if (!isValid(text, suggestion)) throw new Error('The text has changed. Check it again before applying this suggestion.');
  return text.slice(0, suggestion.start) + suggestion.replacement + text.slice(suggestion.end);
}

export function applySuggestions(text: string, suggestions: Suggestion[]): string {
  const sorted = [...suggestions].sort((a, b) => a.start - b.start || a.end - b.end);
  let previous: Suggestion | undefined;
  for (const edit of sorted) {
    if (!isValid(text, edit)) throw new Error('The text has changed. Check it again before applying suggestions.');
    if (previous && (edit.start < previous.end || edit.start === previous.start)) {
      throw new Error('Overlapping suggestions must be applied individually.');
    }
    previous = edit;
  }
  return sorted.reverse().reduce((value, edit) => applySuggestion(value, edit), text);
}

export function preserveCase(original: string, replacement: string): string {
  if (original.length > 1 && original === original.toUpperCase()) return replacement.toUpperCase();
  if (/^[A-Z]/u.test(original)) return replacement[0]?.toUpperCase() + replacement.slice(1);
  return replacement;
}

export function overlaps(a: Suggestion, b: Suggestion): boolean {
  return a.start === b.start || (a.start < b.end && b.start < a.end);
}
