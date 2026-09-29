import { getSettings, scriptId, sitePattern } from './protocol';

const enabled = document.querySelector<HTMLInputElement>('#enabled')!;
const useAI = document.querySelector<HTMLInputElement>('#use-ai')!;
const enableSite = document.querySelector<HTMLButtonElement>('#enable-site')!;
const disableSite = document.querySelector<HTMLButtonElement>('#disable-site')!;
const siteName = document.querySelector<HTMLElement>('#site-name')!;
const siteDetail = document.querySelector<HTMLElement>('#site-detail')!;
const status = document.querySelector<HTMLElement>('#status')!;
let tab: chrome.tabs.Tab | undefined;
let pattern: string | null = null;

function report(error: unknown) { status.textContent = error instanceof Error ? error.message : 'This action could not be completed.'; }

async function refresh() {
  const settings = getSettings(await chrome.storage.local.get(['enabled', 'useAI']));
  enabled.checked = settings.enabled;
  useAI.checked = settings.useAI;
  [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  pattern = sitePattern(tab?.url ?? '');
  if (!pattern) {
    siteName.textContent = 'This page is unavailable';
    siteDetail.textContent = 'Open a regular website to enable writing suggestions.';
    enableSite.disabled = true;
    return;
  }
  siteName.textContent = new URL(tab!.url!).hostname;
  const hasPermission = await chrome.permissions.contains({ origins: [pattern] });
  const [registered] = await chrome.scripting.getRegisteredContentScripts({ ids: [scriptId(pattern)] });
  const active = hasPermission && Boolean(registered);
  siteDetail.textContent = active ? 'Enabled on this site. Checks happen locally.' : hasPermission ? 'Permission granted. Activate suggestions for this site.' : 'Access is requested only for this site.';
  enableSite.textContent = active ? 'Enabled on this site' : 'Enable on this site';
  enableSite.disabled = active;
  disableSite.hidden = !hasPermission;
}

enabled.addEventListener('change', () => { void chrome.storage.local.set({ enabled: enabled.checked }).catch(report); });
useAI.addEventListener('change', () => { void chrome.storage.local.set({ useAI: useAI.checked }).catch(report); });
enableSite.addEventListener('click', async () => {
  if (!pattern || !tab?.id) return;
  const currentPattern = pattern;
  const currentTabId = tab.id;
  try {
    // Permission request stays directly in the user gesture before any await.
    const granted = await chrome.permissions.request({ origins: [currentPattern] });
    if (!granted) { status.textContent = 'Site access was not granted.'; return; }
    const id = scriptId(currentPattern);
    const [existing] = await chrome.scripting.getRegisteredContentScripts({ ids: [id] });
    if (!existing) await chrome.scripting.registerContentScripts([{ id, matches: [currentPattern], js: ['content.js'], runAt: 'document_idle', allFrames: false, persistAcrossSessions: true }]);
    await chrome.scripting.executeScript({ target: { tabId: currentTabId, allFrames: false }, files: ['content.js'] });
    await chrome.tabs.sendMessage(currentTabId, {target: 'content', action: 'site-enabled'});
    status.textContent = 'Ready. Focus an ordinary text field to start writing.';
    await refresh();
  } catch (error) { report(error); }
});

disableSite.addEventListener('click', async () => {
  if (!pattern) return;
  try {
    const openTabs = await chrome.tabs.query({ url: pattern });
    const [existing] = await chrome.scripting.getRegisteredContentScripts({ ids: [scriptId(pattern)] });
    if (existing) await chrome.scripting.unregisterContentScripts({ ids: [existing.id] });
    await chrome.permissions.remove({ origins: [pattern] });
    await Promise.all(openTabs.filter((openTab) => openTab.id !== undefined).map((openTab) => chrome.tabs.sendMessage(openTab.id!, { target: 'content', action: 'site-disabled' }).catch(() => undefined)));
    status.textContent = 'Site permission removed. Suggestions are stopped on this site.';
    await refresh();
  } catch (error) { report(error); }
});

void refresh().catch(report);
