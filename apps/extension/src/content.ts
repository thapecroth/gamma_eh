import type { AnalysisResult, Suggestion } from '@gamma/engine';
import { findEditable, isEligible, isPlainEditable, readText, replaceText, type Editable } from './editable';
import { getSettings, isSiteEnabled, MAX_FIELD_LENGTH, type CheckResponse, type Settings } from './protocol';

// Legacy registered scripts can meet the default content script on the same page.
const singletonKey = '__gammaEhContentAttached';
const globalScope = globalThis as typeof globalThis & { [singletonKey]?: boolean };
if (window === window.top && !globalScope[singletonKey]) {
  globalScope[singletonKey] = true;
  start();
}

function start() {
  let settings: Settings = getSettings({ enabled: false });
  let activeField: Editable | null = null;
  let result: AnalysisResult | null = null;
  let generation = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let composing = false;
  let stopped = false;
  const ignored = new WeakSet<HTMLElement>();
  const dismissed = new Set<string>();
  // Content scripts also run on HTTP sites, where randomUUID is unavailable.
  const sessionId = crypto.getRandomValues(new Uint32Array(4)).join('-');
  const host = document.createElement('div');
  host.setAttribute('data-gamma-ignore', '');
  host.setAttribute('contenteditable', 'false');
  const shadow = host.attachShadow({ mode: 'closed' });
  const style = document.createElement('style');
  style.textContent = `:host{all:initial;position:fixed;right:20px;bottom:20px;z-index:2147483647;max-width:calc(100vw - 32px);font-family:Arial,sans-serif;color:#34382f}*{box-sizing:border-box}.panel{width:308px;max-width:calc(100vw - 32px);border:1px solid #deded3;border-radius:11px;background:#fbfaf6;box-shadow:0 6px 32px #0000001f;overflow:hidden}header{display:flex;align-items:center;gap:8px;padding:12px 14px;border-bottom:1px solid #e6e4db}.mark{background:#7b6693;color:white;border-radius:6px;height:22px;width:21px;font:italic 22px/20px Georgia,serif;text-align:center}.title{font-size:12px;font-weight:600}.tag{font-size:7px;letter-spacing:.7px;color:#919a80;margin-left:auto}button{font:11px Arial,sans-serif;cursor:pointer;border:0;border-radius:4px}button:focus-visible{outline:3px solid #bda5d4;outline-offset:2px}.close{background:none;color:#8e967f;font-size:18px;padding:1px 4px;margin-left:5px}.body{padding:12px 14px;max-height:340px;overflow:auto}.status{font-size:10px;line-height:1.6;color:#919884;margin:0}.card{padding:10px;border:1px solid #e4e4d9;border-radius:6px;background:#fffefb;margin-top:10px}.category{font-size:8px;text-transform:capitalize;color:#939b84}.correction{display:flex;gap:9px;align-items:center;font:14px/1.6 Georgia,serif;overflow-wrap:anywhere;margin:7px 0}.old{color:#b5796f;text-decoration:line-through}.new{color:#68885e}.arrow{color:#abb39d}.message{font-size:10px;line-height:1.6;color:#909986;margin:0 0 9px}.actions{display:flex;gap:8px}.accept{padding:6px 9px;background:#eee8f5;color:#7f6597}.dismiss{padding:6px 8px;background:none;color:#9aa18d}footer{font-size:8px;color:#a0a58f;border-top:1px solid #e6e4db;padding:9px 14px}.notice{margin-top:8px;color:#977953;font-size:9px;line-height:1.5}@media(max-width:480px){:host{right:12px;bottom:12px}.panel{width:282px}}`;
  const panel = document.createElement('section');
  panel.className = 'panel';
  panel.setAttribute('aria-label', 'Gamma EH writing suggestions');
  const header = document.createElement('header');
  const mark = document.createElement('span'); mark.className = 'mark'; mark.textContent = 'γ'; mark.setAttribute('aria-hidden', 'true');
  const title = document.createElement('span'); title.className = 'title'; title.textContent = 'Gamma EH';
  const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = 'ON YOUR DEVICE';
  const close = document.createElement('button'); close.className = 'close'; close.textContent = '×'; close.type = 'button'; close.setAttribute('aria-label', 'Pause suggestions for this field');
  close.onclick = () => { if (activeField) ignored.add(activeField); invalidate(); host.remove(); };
  header.append(mark, title, tag, close);
  const body = document.createElement('div'); body.className = 'body'; body.setAttribute('aria-live', 'polite');
  const footer = document.createElement('footer'); footer.textContent = 'Private by design · You choose what to change';
  panel.append(header, body, footer); shadow.append(style, panel);

  function invalidate() { generation++; clearTimeout(timer); result = null; }
  function attach() { if (!host.isConnected) document.documentElement.append(host); }
  function showStatus(message: string) { attach(); body.replaceChildren(); const p = document.createElement('p'); p.className = 'status'; p.textContent = message; body.append(p); }
  function hide() { invalidate(); host.remove(); }

  function render() {
    if (!activeField || !result) return;
    attach(); body.replaceChildren();
    const suggestions = result.suggestions.filter((suggestion) => !dismissed.has(suggestion.id));
    const status = document.createElement('p'); status.className = 'status';
    const backend = result.backend === 'webgpu' ? 'Local AI · WebGPU' : result.backend === 'wasm' ? 'Local AI · CPU' : 'Local rules';
    status.textContent = suggestions.length ? `${suggestions.length} suggestion${suggestions.length === 1 ? '' : 's'} · ${backend}` : `No suggestions from this checker · ${backend}`;
    body.append(status);
    for (const suggestion of suggestions.slice(0, 6)) {
      const card = document.createElement('article'); card.className = 'card';
      const category = document.createElement('div'); category.className = 'category'; category.textContent = `${suggestion.category} · ${suggestion.source === 'model' ? 'local model' : 'rule'}`;
      const correction = document.createElement('p'); correction.className = 'correction';
      const original = document.createElement('span'); original.className = 'old'; original.textContent = suggestion.original || '(insert)';
      const arrow = document.createElement('span'); arrow.className = 'arrow'; arrow.textContent = '→'; arrow.setAttribute('aria-hidden', 'true');
      const replacement = document.createElement('span'); replacement.className = 'new'; replacement.textContent = suggestion.replacement || '(remove)';
      correction.append(original, arrow, replacement);
      const message = document.createElement('p'); message.className = 'message'; message.textContent = suggestion.message;
      const actions = document.createElement('div'); actions.className = 'actions';
      const accept = document.createElement('button'); accept.type = 'button'; accept.className = 'accept'; accept.textContent = 'Accept'; accept.onclick = () => acceptSuggestion(suggestion);
      const dismiss = document.createElement('button'); dismiss.type = 'button'; dismiss.className = 'dismiss'; dismiss.textContent = 'Dismiss'; dismiss.setAttribute('aria-label', `Dismiss: ${suggestion.message}`); dismiss.onclick = () => { dismissed.add(suggestion.id); render(); };
      actions.append(accept, dismiss); card.append(category, correction, message, actions); body.append(card);
    }
    if (suggestions.length > 6) { const note = document.createElement('p'); note.className = 'notice'; note.textContent = 'More suggestions will appear as you accept or dismiss these.'; body.append(note); }
    if (settings.useAI) { const note = document.createElement('p'); note.className = 'notice'; note.textContent = 'Experimental model trained on synthetic examples. Review suggestions.'; body.append(note); }
    if (result.modelError) { const note = document.createElement('p'); note.className = 'notice'; note.textContent = 'The model is unavailable; local rule suggestions still work.'; body.append(note); }
  }

  function acceptSuggestion(suggestion: Suggestion) {
    const field = activeField;
    const snapshot = result?.text;
    if (!field || snapshot === undefined) return;
    if (!replaceText(field, snapshot, suggestion)) { showStatus('Your text changed or this editor blocked the change. Checking again…'); schedule(); return; }
    schedule();
  }

  function schedule() {
    invalidate();
    if (stopped || !isSiteEnabled(location.href, settings) || composing || !activeField || ignored.has(activeField) || !isEligible(activeField)) { host.remove(); return; }
    if (!isPlainEditable(activeField)) { showStatus('This rich editor is not supported yet. Use a plain-text field to check your writing safely.'); return; }
    const field = activeField;
    const text = readText(field);
    if (!text.trim()) { host.remove(); return; }
    if (text.length > MAX_FIELD_LENGTH) { showStatus('This field is too long for the extension checker. Check a passage under 6,000 characters.'); return; }
    const checkGeneration = generation;
    showStatus('Checking locally…');
    timer = setTimeout(async () => {
      const requestId = `${sessionId}:${checkGeneration}`;
      try {
        const response: CheckResponse = await chrome.runtime.sendMessage({ target: 'background', action: 'analyze', requestId, text, useAI: settings.useAI });
        if (stopped || checkGeneration !== generation || activeField !== field || !field.isConnected || readText(field) !== text) return;
        if (response?.siteDisabled) { hide(); return; }
        if (response?.requestId !== requestId) throw new Error('The local checker could not return a result.');
        if (response.result && response.result.text === text) { result = response.result; render(); }
        else showStatus(response.error ?? 'The local checker could not return a result.');
      } catch { if (checkGeneration === generation) showStatus('The local checker is unavailable. Reload the extension and this page.'); }
    }, 700);
  }

  document.addEventListener('focusin', (event) => {
    if (event.target === host) return;
    const field = findEditable(event.target);
    if (field === activeField) return;
    activeField = field; dismissed.clear();
    if (field) schedule(); else hide();
  }, true);
  document.addEventListener('input', (event) => {
    const field = findEditable(event.target);
    if (field && field === activeField) { dismissed.clear(); schedule(); }
  }, true);
  document.addEventListener('compositionstart', (event) => { if (findEditable(event.target) === activeField) { composing = true; hide(); } }, true);
  document.addEventListener('compositionend', (event) => { if (findEditable(event.target) === activeField) { composing = false; schedule(); } }, true);
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local' || (!changes.enabled && !changes.useAI && !changes.disabledSites)) return;
    settings = getSettings({ enabled: changes.enabled ? changes.enabled.newValue : settings.enabled,
      useAI: changes.useAI ? changes.useAI.newValue : settings.useAI,
      disabledSites: changes.disabledSites ? changes.disabledSites.newValue : settings.disabledSites });
    schedule();
  });
  chrome.runtime.onMessage.addListener((message: unknown) => {
    if (message && typeof message === 'object' && 'target' in message && 'action' in message && message.target === 'content' && message.action === 'site-disabled') { stopped = true; activeField = null; hide(); }
    if (message && typeof message === 'object' && 'target' in message && 'action' in message && message.target === 'content' && message.action === 'site-enabled') {
      stopped = false;
      void chrome.storage.local.get(['enabled', 'useAI', 'disabledSites']).then((stored) => {
        settings = getSettings(stored);
        activeField = findEditable(document.activeElement);
        schedule();
      });
    }
  });
  void chrome.storage.local.get(['enabled', 'useAI', 'disabledSites']).then((stored) => {
    settings = getSettings(stored); activeField = findEditable(document.activeElement); schedule();
  }).catch(() => { stopped = true; hide(); });
}
