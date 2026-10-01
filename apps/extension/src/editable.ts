import type { Suggestion } from '@gamma/engine';

export type Editable = HTMLTextAreaElement | HTMLElement;

export function sensitiveAutocomplete(value: string): boolean {
  return value.toLowerCase().split(/\s+/u).some((token) => token.startsWith('cc-') || ['one-time-code', 'current-password', 'new-password', 'username', 'webauthn'].includes(token));
}

export function isEligible(field: HTMLElement): boolean {
  if (!(field instanceof HTMLTextAreaElement) && !field.isContentEditable) return false;
  if (field instanceof HTMLTextAreaElement && (field.disabled || field.readOnly)) return false;
  if (field.closest('[spellcheck="false"], [data-gamma-ignore], [data-private], [data-sensitive], [aria-hidden="true"], [inert]')) return false;
  if (field.getAttribute('aria-disabled') === 'true' || field.getAttribute('aria-readonly') === 'true') return false;
  if (sensitiveAutocomplete(field.getAttribute('autocomplete') ?? '')) return false;
  return true;
}

export function findEditable(target: EventTarget | null): Editable | null {
  if (!(target instanceof HTMLElement)) return null;
  if (target instanceof HTMLTextAreaElement) return isEligible(target) ? target : null;
  const candidate = target.closest<HTMLElement>('[contenteditable="true"], [contenteditable=""], [contenteditable="plaintext-only"]');
  return candidate && isEligible(candidate) ? candidate : null;
}

export function isPlainEditable(field: Editable): boolean {
  if (field instanceof HTMLTextAreaElement) return true;
  if (field.closest('[data-lexical-editor], .ProseMirror, .ql-editor, .cke_editable, [data-slate-editor]')) return false;
  return Array.from(field.childNodes).every((node) => node.nodeType === Node.TEXT_NODE || (node instanceof HTMLBRElement && !node.childNodes.length));
}

export function readText(field: Editable): string {
  if (field instanceof HTMLTextAreaElement) return field.value;
  return Array.from(field.childNodes).map((node) => node instanceof HTMLBRElement ? '\n' : node.textContent ?? '').join('');
}

export function matchesSuggestion(text: string, suggestion: Suggestion): boolean {
  return Number.isInteger(suggestion.start) && Number.isInteger(suggestion.end) && suggestion.start >= 0
    && suggestion.end >= suggestion.start && suggestion.end <= text.length
    && text.slice(suggestion.start, suggestion.end) === suggestion.original;
}

interface Boundary { node: Node; offset: number }
export function boundaryAt(field: HTMLElement, offset: number): Boundary | null {
  let cursor = 0;
  for (const [index, node] of Array.from(field.childNodes).entries()) {
    if (node.nodeType === Node.TEXT_NODE) {
      const length = node.textContent?.length ?? 0;
      if (offset <= cursor + length) return { node, offset: offset - cursor };
      cursor += length;
    } else if (node instanceof HTMLBRElement) {
      if (offset === cursor) return { node: field, offset: index };
      cursor++;
      if (offset === cursor) return { node: field, offset: index + 1 };
    } else return null;
  }
  return offset === cursor ? { node: field, offset: field.childNodes.length } : null;
}

export function replaceText(field: Editable, originalText: string, suggestion: Suggestion): boolean {
  if (!field.isConnected || !isEligible(field) || !isPlainEditable(field) || readText(field) !== originalText || !matchesSuggestion(originalText, suggestion)) return false;
  const beforeInput = new InputEvent('beforeinput', { bubbles: true, cancelable: true, composed: true, inputType: 'insertReplacementText', data: suggestion.replacement });
  if (!field.dispatchEvent(beforeInput)) return false;
  // Event handlers can update editor state; verify the snapshot again before mutation.
  if (!field.isConnected || !isEligible(field) || !isPlainEditable(field) || readText(field) !== originalText) return false;
  if (field instanceof HTMLTextAreaElement) {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
    if (!setter) return false;
    setter.call(field, originalText.slice(0, suggestion.start) + suggestion.replacement + originalText.slice(suggestion.end));
    field.focus({ preventScroll: true });
    const caret = suggestion.start + suggestion.replacement.length;
    field.setSelectionRange(caret, caret);
  } else {
    const start = boundaryAt(field, suggestion.start);
    const end = boundaryAt(field, suggestion.end);
    if (!start || !end) return false;
    const range = document.createRange();
    range.setStart(start.node, start.offset);
    range.setEnd(end.node, end.offset);
    range.deleteContents();
    const replacement = document.createTextNode(suggestion.replacement);
    range.insertNode(replacement);
    range.setStartAfter(replacement);
    range.collapse(true);
    field.focus({ preventScroll: true });
    const selection = window.getSelection();
    selection?.removeAllRanges();
    selection?.addRange(range);
  }
  field.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, inputType: 'insertReplacementText', data: suggestion.replacement }));
  return true;
}
