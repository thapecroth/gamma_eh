import { applySuggestions } from './edits';
import type { Suggestion } from './types';

interface SourceSpan {start: number; end: number}
interface ChangedSpan extends SourceSpan {edit: Suggestion}

/** Compose accepted model passes into suggestions against the original UTF-16 text. */
export class EditHistory {
  private units: SourceSpan[];
  private changes: ChangedSpan[] = [];
  text: string;

  constructor(private original: string) {
    this.text = original;
    this.units = Array.from({length: original.length}, (_, i) => ({start: i, end: i + 1}));
  }

  apply(edits: Suggestion[]): void {
    const next = applySuggestions(this.text, edits);
    for (const edit of [...edits].sort((a, b) => b.start - a.start || b.end - a.end)) {
      let start: number;
      let end: number;
      if (edit.start < edit.end) {
        const affected = this.units.slice(edit.start, edit.end);
        start = Math.min(...affected.map(unit => unit.start));
        end = Math.max(...affected.map(unit => unit.end));
      } else {
        const left = this.units[edit.start - 1];
        const right = this.units[edit.start];
        if (left && right && left.start === right.start && left.end === right.end) {
          start = left.start; end = left.end;
        } else {
          start = right?.start ?? left?.end ?? 0; end = start;
        }
      }
      this.units.splice(edit.start, edit.end - edit.start,
        ...Array.from({length: edit.replacement.length}, () => ({start, end})));
      const changed: ChangedSpan = {start, end, edit};
      // Adjacent edits can share an insertion anchor; merge them into one safe patch.
      this.changes.push(changed);
      this.changes.sort((a, b) => a.start - b.start || a.end - b.end);
      const merged: ChangedSpan[] = [];
      for (const change of this.changes) {
        const previous = merged.at(-1);
        if (previous && change.start <= previous.end) {
          previous.end = Math.max(previous.end, change.end);
          previous.edit = {...previous.edit, confidence: Math.min(previous.edit.confidence, change.edit.confidence)};
        } else merged.push({...change});
      }
      this.changes = merged;
    }
    this.text = next;
  }

  suggestions(): Suggestion[] {
    return this.changes.flatMap(({start, end, edit}) => {
      let first = -1;
      let last = -1;
      for (let index = 0; index < this.units.length; index++) {
        const unit = this.units[index];
        const belongs = start === end ? unit.start === start && unit.end === end :
          unit.start === unit.end ? unit.start >= start && unit.start <= end : unit.start < end && start < unit.end;
        if (belongs) { if (first < 0) first = index; last = index + 1; }
      }
      const replacement = first < 0 ? '' : this.text.slice(first, last);
      const original = this.original.slice(start, end);
      if (original === replacement) return [];
      return [{...edit, id: 'model-composed-' + start + '-' + end, start, end, original, replacement, checkedText: this.original}];
    });
  }
}
