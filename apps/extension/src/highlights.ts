import type { Suggestion } from '@gamma/engine';
import { boundaryAt, matchesSuggestion, type Editable } from './editable';

export interface HighlightRect {
  left: number;
  top: number;
  width: number;
  height: number;
  suggestion: Suggestion;
}

export interface FieldBounds { left: number; top: number; right: number; bottom: number }

export function visibleFieldBounds(field: Editable): FieldBounds | null {
  const bounds = field.getBoundingClientRect();
  const scaleX = bounds.width / (field.offsetWidth || bounds.width);
  const scaleY = bounds.height / (field.offsetHeight || bounds.height);
  let left = Math.max(0, bounds.left + field.clientLeft * scaleX);
  let top = Math.max(0, bounds.top + field.clientTop * scaleY);
  let right = Math.min(innerWidth, bounds.left + (field.clientLeft + field.clientWidth) * scaleX);
  let bottom = Math.min(innerHeight, bounds.top + (field.clientTop + field.clientHeight) * scaleY);
  // A nested scrolling container can clip the editor before the viewport does.
  for (let parent = field.parentElement; parent; parent = parent.parentElement) {
    // The document scroller is already clipped by the viewport above. Its
    // document-relative bounding box would subtract the page scroll twice.
    if (parent === document.scrollingElement) continue;
    const style = getComputedStyle(parent);
    const rect = parent.getBoundingClientRect();
    const parentScaleX = rect.width / (parent.offsetWidth || rect.width);
    const parentScaleY = rect.height / (parent.offsetHeight || rect.height);
    if (/(auto|scroll|hidden|clip)/u.test(style.overflowX)) {
      left = Math.max(left, rect.left + parent.clientLeft * parentScaleX);
      right = Math.min(right, rect.left + (parent.clientLeft + parent.clientWidth) * parentScaleX);
    }
    if (/(auto|scroll|hidden|clip)/u.test(style.overflowY)) {
      top = Math.max(top, rect.top + parent.clientTop * parentScaleY);
      bottom = Math.min(bottom, rect.top + (parent.clientTop + parent.clientHeight) * parentScaleY);
    }
  }
  if (right <= left || bottom <= top || !field.getClientRects().length || getComputedStyle(field).visibility !== 'visible') return null;
  return {left, top, right, bottom};
}

// Measure text without inserting markup into the site's editor or changing its
// selection. Textareas need a typography-matched mirror; plain editors use Range.
export function measureHighlights(field: Editable, text: string, suggestions: Suggestion[], container: ShadowRoot, clip = visibleFieldBounds(field)): HighlightRect[] {
  if (!clip) return [];
  const {left, top, right, bottom} = clip;
  const bounds = field.getBoundingClientRect();
  const scaleX = bounds.width / (field.offsetWidth || bounds.width);
  const scaleY = bounds.height / (field.offsetHeight || bounds.height);

  let mirror: HTMLDivElement | undefined;
  let mirrorBounds: DOMRect | undefined;
  let textNode: Text | undefined;
  if (field instanceof HTMLTextAreaElement) {
    mirror = document.createElement('div');
    const style = getComputedStyle(field);
    for (const property of ['font-family', 'font-size', 'font-weight', 'font-style', 'font-stretch', 'font-variant', 'font-kerning', 'font-feature-settings', 'font-variation-settings', 'line-height', 'letter-spacing', 'word-spacing', 'text-align', 'text-indent', 'text-transform', 'direction', 'tab-size', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left', 'word-break', 'overflow-wrap', 'white-space']) {
      mirror.style.setProperty(property, style.getPropertyValue(property));
    }
    Object.assign(mirror.style, {position: 'fixed', left: '-100000px', top: '0', width: `${field.clientWidth}px`, boxSizing: 'border-box', visibility: 'hidden'});
    textNode = document.createTextNode(text);
    mirror.append(textNode, document.createTextNode('\u200b'));
    container.append(mirror);
    mirrorBounds = mirror.getBoundingClientRect();
  }
  const highlights: HighlightRect[] = [];
  try {
    for (const suggestion of suggestions) {
      if (!matchesSuggestion(text, suggestion)) continue;
      const range = document.createRange();
      if (textNode) {
        range.setStart(textNode, suggestion.start);
        range.setEnd(textNode, suggestion.end);
      } else {
        const start = boundaryAt(field, suggestion.start);
        const end = boundaryAt(field, suggestion.end);
        if (!start || !end) continue;
        range.setStart(start.node, start.offset);
        range.setEnd(end.node, end.offset);
      }
      for (const rect of Array.from(range.getClientRects())) {
        const x = mirrorBounds ? bounds.left + (field.clientLeft + rect.left - mirrorBounds.left - field.scrollLeft) * scaleX : rect.left;
        const y = mirrorBounds ? bounds.top + (field.clientTop + rect.top - mirrorBounds.top - field.scrollTop) * scaleY : rect.top;
        const width = Math.max(suggestion.start === suggestion.end ? 6 : 0, rect.width * (mirrorBounds ? scaleX : 1));
        const height = rect.height * (mirrorBounds ? scaleY : 1);
        // Do not draw a mark for a line whose baseline is outside the field.
        if (y + height > bottom || y + height < top || x + width <= left || x >= right) continue;
        const clippedLeft = Math.max(x, left);
        highlights.push({left: clippedLeft, top: Math.max(y, top), width: Math.min(x + width, right) - clippedLeft, height: y + height - Math.max(y, top), suggestion});
      }
    }
  } finally { mirror?.remove(); }
  return highlights;
}
