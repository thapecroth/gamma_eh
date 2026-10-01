import type { AnalysisResult, Suggestion } from '@gamma/engine';
import { findEditable, isEligible, isPlainEditable, readText, replaceText, type Editable } from './editable';
import { InlineSuggestions } from './inline-ui';
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
  const ui = new InlineSuggestions({
    accept: acceptSuggestion,
    dismiss: (suggestion) => { dismissed.add(suggestion.id); render(); },
    pause: () => { if (activeField) ignored.add(activeField); invalidate(); },
  });

  function invalidate() { generation++; clearTimeout(timer); result = null; ui.clear(); }
  function showStatus(message: string) { if (activeField) ui.showStatus(activeField, message); }
  function hide() { invalidate(); }

  function render() {
    if (!activeField || !result) return;
    ui.show(activeField, result, result.suggestions.filter((suggestion) => !dismissed.has(suggestion.id)), settings.useAI);
  }

  function acceptSuggestion(suggestion: Suggestion) {
    const field = activeField;
    const snapshot = result?.text;
    if (!field || snapshot === undefined) return;
    if (!replaceText(field, snapshot, suggestion)) { schedule(); return; }
    schedule();
  }

  function schedule() {
    invalidate();
    if (stopped || !isSiteEnabled(location.href, settings) || composing || !activeField || ignored.has(activeField) || !isEligible(activeField)) return;
    if (!isPlainEditable(activeField)) { showStatus('This rich editor is not supported yet. Use a plain-text field to check your writing safely.'); return; }
    const field = activeField;
    const text = readText(field);
    if (!text.trim()) return;
    if (text.length > MAX_FIELD_LENGTH) { showStatus('This field is too long for the extension checker. Check a passage under 6,000 characters.'); return; }
    const checkGeneration = generation;
    showStatus('Checking locally…');
    timer = setTimeout(async () => {
      const requestId = `${sessionId}:${checkGeneration}`;
      try {
        const response: CheckResponse = await chrome.runtime.sendMessage({ target: 'background', action: 'analyze', requestId, text, useAI: settings.useAI });
        if (stopped || checkGeneration !== generation || activeField !== field || !field.isConnected || readText(field) !== text || !isEligible(field) || !isPlainEditable(field)) return;
        if (response?.siteDisabled) { hide(); return; }
        if (response?.requestId !== requestId) throw new Error('The local checker could not return a result.');
        if (response.result && response.result.text === text) { result = response.result; render(); }
        else showStatus(response.error ?? 'The local checker could not return a result.');
      } catch { if (checkGeneration === generation) showStatus('The local checker is unavailable. Reload the extension and this page.'); }
    }, 700);
  }

  document.addEventListener('focusin', (event) => {
    // Controls live in a closed shadow root, so focus is retargeted to its host.
    if (ui.owns(event.target)) return;
    const field = findEditable(event.target);
    if (field === activeField) return;
    activeField = field; dismissed.clear(); composing = false;
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
