import { getSettings, isCheckMessage, isSiteEnabled, sitePattern, type CheckResponse } from './protocol';

let offscreenCreation: Promise<void> | null = null;

async function ensureOffscreen(): Promise<void> {
  if (offscreenCreation) return offscreenCreation;
  offscreenCreation = (async () => {
    const documentUrl = chrome.runtime.getURL('offscreen.html');
    const contexts = await chrome.runtime.getContexts({ contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT], documentUrls: [documentUrl] });
    if (!contexts.length) await chrome.offscreen.createDocument({ url: 'offscreen.html', reasons: [chrome.offscreen.Reason.WORKERS], justification: 'Run the bundled local grammar model in a worker without transmitting typed text.' });
  })();
  try { await offscreenCreation; } finally { offscreenCreation = null; }
}

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse: (response: CheckResponse) => void) => {
  if (!isCheckMessage(message) || message.target !== 'background') return;
  if (sender.id !== chrome.runtime.id || !sender.tab || sender.frameId !== 0) return;
  (async () => {
    try {
      const pattern = sitePattern(sender.url ?? '');
      const allowed = pattern && await chrome.permissions.contains({ origins: [pattern] });
      const settings = getSettings(await chrome.storage.local.get(['enabled', 'useAI', 'disabledSites']));
      if (!allowed || !isSiteEnabled(sender.url ?? '', settings)) {
        sendResponse({ requestId: message.requestId, siteDisabled: true });
        return;
      }
      await ensureOffscreen();
      const response: CheckResponse = await chrome.runtime.sendMessage({ ...message, target: 'offscreen' });
      if (!response || response.requestId !== message.requestId) throw new Error('The local checker returned an invalid response.');
      sendResponse(response);
    } catch (error) {
      sendResponse({ requestId: message.requestId, error: error instanceof Error ? error.message : 'The local checker could not start.' });
    }
  })();
  return true;
});

chrome.runtime.onInstalled.addListener(async () => {
  // Older versions persisted one script registration for each enabled site.
  const scripts = await chrome.scripting.getRegisteredContentScripts();
  const ids = scripts.filter((script) => script.id.startsWith('gamma-site-')).map((script) => script.id);
  if (ids.length) await chrome.scripting.unregisterContentScripts({ ids });
});

chrome.permissions.onRemoved.addListener(async (removed) => {
  if (removed.origins?.length) {
    // URL visibility can disappear with the permission, so contact all open tabs;
    // only matching Gamma EH content scripts receive this extension message.
    const tabs = await chrome.tabs.query({});
    for (const tab of tabs) {
      if (!tab.id) continue;
      const pattern = sitePattern(tab.url ?? '');
      if (pattern && await chrome.permissions.contains({ origins: [pattern] })) continue;
      await chrome.tabs.sendMessage(tab.id, { target: 'content', action: 'site-disabled' }).catch(() => undefined);
    }
  }
});
