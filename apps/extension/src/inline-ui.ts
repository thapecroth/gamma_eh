import type { AnalysisResult, Suggestion } from '@gamma/engine';
import { isEligible, isPlainEditable, readText, type Editable } from './editable';
import { measureHighlights, visibleFieldBounds, type HighlightRect } from './highlights';

interface Actions {
  accept: (suggestion: Suggestion) => void;
  dismiss: (suggestion: Suggestion) => void;
  pause: () => void;
}

const styles = `
:host{all:initial;position:fixed;inset:0;pointer-events:none;z-index:2147483647;font:14px/1.5 Arial,Helvetica,sans-serif;color:#20252b}
*{box-sizing:border-box}button{font:inherit;cursor:pointer}button:focus-visible{outline:3px solid #8b75c7;outline-offset:3px}[hidden]{display:none!important}
.underline{position:fixed;pointer-events:none;border-bottom:2px solid #e54453;border-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='4'%3E%3Cpath d='M0 3Q2 0 4 3T8 3' fill='none' stroke='%23e54453' stroke-width='1.5'/%3E%3C/svg%3E") 0 0 4 repeat;background:transparent}
.underline.active{background:#e5445314}
.badge{position:fixed;pointer-events:auto;display:flex;align-items:center;gap:6px;height:30px;min-width:30px;padding:0 9px;border:1px solid #ddd9e7;border-radius:16px;background:#fff;color:#756090;box-shadow:0 2px 8px #00000018;font-size:12px;font-weight:700}
.badge.has-issues{color:#bd293b;border-color:#f4c6cc}.badge:hover{background:#f7f4fb}.gamma{font:italic 21px/1 Georgia,serif}
.panel{position:fixed;pointer-events:auto;width:320px;max-width:calc(100vw - 24px);max-height:calc(100vh - 24px);overflow:auto;border:1px solid #e3e1e7;border-top:3px solid #e54453;border-radius:12px;background:#fff;box-shadow:0 8px 32px #20252b26}
header{display:flex;align-items:center;gap:8px;padding:12px 16px 0}.category{font-size:12px;font-weight:700;text-transform:capitalize;color:#b52d40}.close{margin-left:auto;border:0;background:transparent;color:#66707b;font-size:21px;line-height:1;padding:3px 5px;border-radius:4px}.close:hover{background:#f1eff5}
.body{padding:8px 16px 16px}.correction{display:flex;align-items:center;gap:10px;margin:4px 0 10px;flex-wrap:wrap;font-size:18px;line-height:1.5;overflow-wrap:anywhere}.old{color:#ab3c49;text-decoration:line-through}.arrow{color:#818894}.new{color:#216d50;font-weight:700}.message{margin:0 0 16px;font-size:13px;line-height:1.6;color:#59616c;overflow-wrap:anywhere}
.actions{display:flex;gap:8px}.accept{flex:1;border:0;border-radius:7px;padding:9px 12px;background:#756090;color:#fff;font-size:13px;font-weight:700}.accept:hover{background:#60497e}.dismiss{border:1px solid #e1dfe6;border-radius:7px;padding:9px 12px;background:#fff;color:#59616c;font-size:13px}.dismiss:hover{background:#f6f5f8}
.navigation{display:flex;align-items:center;justify-content:space-between;padding:9px 16px;border-top:1px solid #eeedf1;font-size:11px;color:#59616c}.navigation button{border:0;background:none;color:#756090;border-radius:4px;padding:4px 8px;font-size:16px}.navigation button:disabled{color:#b9b5c2;cursor:default}
footer{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 16px;border-top:1px solid #eeedf1;font-size:10px;color:#68716c}.pause{background:none;border:0;padding:2px;color:#68716c;font-size:10px;text-decoration:underline}.notice{font-size:11px;line-height:1.5;color:#75633e;margin:12px 0 0}.checker-status{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
`;

export class InlineSuggestions {
  private host = document.createElement('div');
  private shadow = this.host.attachShadow({mode: 'closed'});
  private marks = document.createElement('div');
  private badge = document.createElement('button');
  private panel = document.createElement('section');
  private status = document.createElement('p');
  private field: Editable | null = null;
  private snapshot: string | null = null;
  private suggestions: Suggestion[] = [];
  private result: AnalysisResult | null = null;
  private useAI = false;
  private rects: HighlightRect[] = [];
  private selected: Suggestion | null = null;
  private pinned = false;
  private hideTimer: ReturnType<typeof setTimeout> | undefined;
  private frame = 0;
  private layoutFrame = 0;
  private lastBounds: DOMRect | null = null;
  private resize = new ResizeObserver(() => this.refresh());
  private mutation = new MutationObserver(() => this.refresh());

  constructor(private actions: Actions) {
    this.host.setAttribute('data-gamma-ignore', '');
    this.host.setAttribute('contenteditable', 'false');
    const style = document.createElement('style'); style.textContent = styles;
    this.marks.setAttribute('aria-hidden', 'true');
    this.badge.className = 'badge'; this.badge.type = 'button'; this.badge.setAttribute('aria-haspopup', 'dialog'); this.badge.setAttribute('aria-expanded', 'false');
    this.badge.onclick = () => { this.pinned = true; this.open(this.suggestions[0] ?? null, true); };
    this.panel.className = 'panel'; this.panel.hidden = true; this.panel.id = 'gamma-suggestion';
    this.panel.setAttribute('role', 'dialog'); this.panel.setAttribute('aria-label', 'Gamma EH writing suggestion');
    this.badge.setAttribute('aria-controls', this.panel.id);
    this.status.className = 'checker-status'; this.status.setAttribute('role', 'status'); this.status.setAttribute('aria-live', 'polite');
    this.shadow.append(style, this.marks, this.badge, this.panel, this.status);
    this.panel.addEventListener('pointerenter', () => this.cancelClose());
    this.panel.addEventListener('pointerleave', () => this.deferClose());
    document.addEventListener('pointermove', (event) => this.pointerMove(event), true);
    document.addEventListener('click', (event) => {
      if (event.target === this.host) return;
      const hit = this.hit(event.clientX, event.clientY, event.target);
      if (hit) { this.pinned = true; this.open(hit.suggestion); }
      else this.close();
    }, true);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !this.panel.hidden) {
        event.preventDefault(); event.stopPropagation(); this.close(true);
      }
      // Accessible alternative to hovering; keep focus inside the correction card.
      if (event.altKey && event.key === 'F8' && this.field && !this.badge.hidden) {
        event.preventDefault(); this.pinned = true; this.open(this.suggestions[0] ?? null, true);
      }
    }, true);
    document.addEventListener('scroll', () => this.refresh(), true);
    window.addEventListener('resize', () => this.refresh());
    window.visualViewport?.addEventListener('resize', () => this.refresh());
    document.fonts.addEventListener('loadingdone', () => this.refresh());
  }

  owns(target: EventTarget | null) { return target === this.host; }

  clear() {
    this.close(); this.resize.disconnect(); this.mutation.disconnect();
    cancelAnimationFrame(this.frame); this.frame = 0;
    cancelAnimationFrame(this.layoutFrame); this.layoutFrame = 0; this.lastBounds = null;
    this.field = null; this.snapshot = null; this.result = null; this.useAI = false; this.rects = []; this.suggestions = [];
    this.marks.replaceChildren(); this.host.remove();
  }

  showStatus(field: Editable, message: string) {
    this.clear(); this.mount(field); this.status.textContent = message;
    this.badge.textContent = 'γ'; this.badge.className = 'badge';
    this.badge.setAttribute('aria-label', `Gamma EH: ${message}`); this.badge.title = message;
    this.position();
  }

  show(field: Editable, result: AnalysisResult, suggestions: Suggestion[], useAI: boolean) {
    this.clear(); this.mount(field); this.result = result; this.snapshot = result.text;
    this.suggestions = suggestions; this.useAI = useAI;
    const backend = result.backend === 'webgpu' ? 'Local AI · WebGPU' : result.backend === 'wasm' ? 'Local AI · CPU' : 'Local rules';
    const count = suggestions.length;
    this.status.textContent = count ? `${count} suggestion${count === 1 ? '' : 's'} · ${backend}` : `No suggestions from this checker · ${backend}`;
    const gamma = document.createElement('span'); gamma.className = 'gamma'; gamma.textContent = 'γ'; gamma.setAttribute('aria-hidden', 'true');
    const number = document.createElement('span'); number.textContent = count ? String(count) : '✓'; number.setAttribute('aria-hidden', 'true');
    this.badge.replaceChildren(gamma, number); this.badge.className = `badge${count ? ' has-issues' : ''}`;
    this.badge.setAttribute('aria-label', `Gamma EH: ${this.status.textContent}. Open suggestions`);
    this.badge.title = count ? 'Hover an underlined word, or open suggestions (Alt+F8)' : this.status.textContent;
    this.position();
  }

  private mount(field: Editable) {
    this.field = field; document.documentElement.append(this.host);
    this.resize.observe(field);
    this.mutation.observe(field, {attributes: true, characterData: true, childList: true, subtree: true});
    for (let parent = field.parentElement; parent; parent = parent.parentElement) {
      this.resize.observe(parent);
      this.mutation.observe(parent, {attributes: true});
    }
    this.lastBounds = field.getBoundingClientRect();
    this.layoutFrame = requestAnimationFrame(() => this.watchLayout());
  }

  private watchLayout() {
    if (!this.field?.isConnected) { this.clear(); return; }
    // Transforms and layout shifts can move a field without resizing it. Read
    // one bounding box per frame; remeasure text only when that box changes.
    const bounds = this.field.getBoundingClientRect();
    const previous = this.lastBounds;
    if (!previous || bounds.left !== previous.left || bounds.top !== previous.top || bounds.width !== previous.width || bounds.height !== previous.height) this.refresh();
    this.lastBounds = bounds;
    this.layoutFrame = requestAnimationFrame(() => this.watchLayout());
  }

  private valid() {
    return this.field?.isConnected && isEligible(this.field) && (this.snapshot === null || isPlainEditable(this.field) && readText(this.field) === this.snapshot);
  }

  private refresh() {
    if (this.frame || !this.field) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.position(); });
  }

  private position() {
    if (!this.valid() || !this.field) { this.clear(); return; }
    const field = this.field;
    const clip = visibleFieldBounds(field);
    this.rects = this.snapshot === null || !clip ? [] : measureHighlights(field, this.snapshot, this.suggestions, this.shadow, clip);
    this.paint();
    const rect = field.getBoundingClientRect();
    const onScreen = clip !== null;
    this.badge.hidden = !onScreen;
    this.badge.style.left = `${Math.max(12, Math.min(innerWidth - this.badge.offsetWidth - 12, rect.right - this.badge.offsetWidth - 8))}px`;
    this.badge.style.top = `${Math.max(12, Math.min(innerHeight - 42, rect.bottom + 6))}px`;
    if (!onScreen) this.close();
    else if (!this.panel.hidden) this.positionPanel();
  }

  private paint() {
    this.marks.replaceChildren(...this.rects.map((rect) => {
      const mark = document.createElement('span'); mark.className = `underline${rect.suggestion.id === this.selected?.id ? ' active' : ''}`;
      Object.assign(mark.style, {left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height + 2}px`});
      return mark;
    }));
  }

  private hit(x: number, y: number, target: EventTarget | null) {
    if (!this.field || !(target instanceof Node) || target !== this.field && !this.field.contains(target)) return undefined;
    return this.rects.find(rect => x >= rect.left && x <= rect.left + rect.width && y >= rect.top && y <= rect.top + rect.height + 4);
  }

  private pointerMove(event: PointerEvent) {
    if (!this.field) return;
    if (!this.valid()) { this.clear(); return; }
    if (event.target === this.host) { this.cancelClose(); return; }
    const hit = this.hit(event.clientX, event.clientY, event.target);
    if (hit) { this.cancelClose(); if (!this.pinned && (this.panel.hidden || this.selected?.id !== hit.suggestion.id)) this.open(hit.suggestion); }
    else this.deferClose();
  }

  private deferClose() {
    // Allow the pointer to cross the gap between the word and its correction.
    if (this.pinned || this.panel.hidden || this.hideTimer) return;
    this.hideTimer = setTimeout(() => { this.hideTimer = undefined; this.close(); }, 250);
  }

  private cancelClose() {
    clearTimeout(this.hideTimer); this.hideTimer = undefined;
  }

  private close(restoreFocus = false) {
    this.cancelClose();
    const focused = this.shadow.activeElement && this.panel.contains(this.shadow.activeElement);
    this.panel.hidden = true; this.panel.replaceChildren(); this.selected = null; this.pinned = false;
    this.badge.setAttribute('aria-expanded', 'false'); this.paint();
    if (restoreFocus && focused) this.field?.focus({preventScroll: true});
  }

  private open(suggestion: Suggestion | null, focus = false) {
    if (!this.valid()) { this.clear(); return; }
    this.cancelClose(); this.selected = suggestion;
    const header = document.createElement('header');
    const category = document.createElement('span'); category.className = 'category'; category.textContent = suggestion?.category ?? 'Gamma EH';
    this.panel.setAttribute('aria-label', suggestion ? `Gamma EH ${suggestion.category} suggestion: ${suggestion.original || 'insert'} to ${suggestion.replacement || 'remove'}` : 'Gamma EH checker status');
    const close = document.createElement('button'); close.className = 'close'; close.type = 'button'; close.textContent = '×'; close.setAttribute('aria-label', 'Close suggestion'); close.onclick = () => this.close(true);
    header.append(category, close);
    const body = document.createElement('div'); body.className = 'body';
    if (suggestion) {
      const correction = document.createElement('p'); correction.className = 'correction';
      const original = document.createElement('span'); original.className = 'old'; original.textContent = suggestion.original || '(insert)';
      const arrow = document.createElement('span'); arrow.className = 'arrow'; arrow.textContent = '→'; arrow.setAttribute('aria-hidden', 'true');
      const replacement = document.createElement('span'); replacement.className = 'new'; replacement.textContent = suggestion.replacement || '(remove)';
      correction.append(original, arrow, replacement);
      const message = document.createElement('p'); message.className = 'message'; message.textContent = suggestion.message;
      const actions = document.createElement('div'); actions.className = 'actions';
      const accept = document.createElement('button'); accept.type = 'button'; accept.className = 'accept'; accept.textContent = 'Accept suggestion'; accept.onclick = () => this.actions.accept(suggestion);
      const dismiss = document.createElement('button'); dismiss.type = 'button'; dismiss.className = 'dismiss'; dismiss.textContent = 'Dismiss'; dismiss.onclick = () => { this.field?.focus({preventScroll: true}); this.actions.dismiss(suggestion); };
      actions.append(accept, dismiss); body.append(correction, message, actions);
    } else {
      const message = document.createElement('p'); message.className = 'message'; message.textContent = this.status.textContent; body.append(message);
    }
    if (suggestion?.source === 'model' || this.useAI) {
      const note = document.createElement('p'); note.className = 'notice'; note.textContent = 'Experimental model trained on synthetic examples. Review suggestions.'; body.append(note);
    }
    if (this.result?.modelError) {
      const note = document.createElement('p'); note.className = 'notice'; note.textContent = 'The model is unavailable; local rule suggestions still work.'; body.append(note);
    }
    this.panel.replaceChildren(header, body);
    if (suggestion && this.suggestions.length > 1) {
      const index = this.suggestions.indexOf(suggestion);
      const navigation = document.createElement('div'); navigation.className = 'navigation';
      const previous = document.createElement('button'); previous.type = 'button'; previous.textContent = '←'; previous.setAttribute('aria-label', 'Previous suggestion'); previous.disabled = index === 0;
      const next = document.createElement('button'); next.type = 'button'; next.textContent = '→'; next.setAttribute('aria-label', 'Next suggestion'); next.disabled = index === this.suggestions.length - 1;
      previous.onclick = () => this.open(this.suggestions[index - 1]!, true); next.onclick = () => this.open(this.suggestions[index + 1]!, true);
      const count = document.createElement('span'); count.textContent = `${index + 1} of ${this.suggestions.length}`;
      navigation.append(previous, count, next); this.panel.append(navigation);
    }
    const footer = document.createElement('footer');
    const privacy = document.createElement('span'); privacy.textContent = 'Gamma EH · On your device';
    const pause = document.createElement('button'); pause.type = 'button'; pause.className = 'pause'; pause.textContent = 'Pause this field'; pause.onclick = () => { this.field?.focus({preventScroll: true}); this.actions.pause(); };
    footer.append(privacy, pause); this.panel.append(footer);
    this.panel.hidden = false; this.badge.setAttribute('aria-expanded', 'true'); this.paint(); this.positionPanel();
    if (focus) (this.panel.querySelector<HTMLButtonElement>('.accept') ?? close).focus({preventScroll: true});
  }

  private positionPanel() {
    const anchor = this.selected ? this.rects.find(rect => rect.suggestion.id === this.selected?.id) : undefined;
    // Off-screen words remain reviewable through the keyboard/count button.
    const rect = anchor ?? this.badge.getBoundingClientRect();
    const width = this.panel.offsetWidth;
    const height = this.panel.offsetHeight;
    const below = rect.top + rect.height + 8;
    const top = below + height <= innerHeight - 12 ? below : rect.top - height - 8;
    this.panel.style.left = `${Math.max(12, Math.min(rect.left, innerWidth - width - 12))}px`;
    this.panel.style.top = `${Math.max(12, Math.min(top, innerHeight - height - 12))}px`;
  }
}
