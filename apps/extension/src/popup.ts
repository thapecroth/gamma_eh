import { getSettings, sitePattern } from './protocol';

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
  const settings = getSettings(await chrome.storage.local.get(['enabled', 'useAI', 'disabledSites']));
  enabled.checked = settings.enabled;
  useAI.checked = settings.useAI;
  [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  pattern = sitePattern(tab?.url ?? '');
  if (!pattern) {
    siteName.textContent = 'This page is unavailable';
    siteDetail.textContent = 'Writing suggestions run automatically on regular websites.';
    enableSite.disabled = true;
    enableSite.hidden = true;
    disableSite.hidden = true;
    return;
  }
  siteName.textContent = new URL(tab!.url!).hostname;
  const hasPermission = await chrome.permissions.contains({ origins: [pattern] });
  const paused = settings.disabledSites.includes(pattern);
  siteDetail.textContent = !hasPermission ? "Allow access in Chrome's extension settings, then reload this page."
    : paused ? 'Paused on this site.' : !settings.enabled ? 'Writing suggestions are paused everywhere.'
    : 'Enabled automatically. Checks happen locally.';
  enableSite.hidden = !paused;
  enableSite.disabled = !hasPermission;
  disableSite.hidden = paused || !hasPermission;
}

enabled.addEventListener('change', () => { void chrome.storage.local.set({ enabled: enabled.checked }).then(refresh).catch(report); });
useAI.addEventListener('change', () => { void chrome.storage.local.set({ useAI: useAI.checked }).catch(report); });

async function setSitePaused(paused: boolean) {
  if (!pattern) return;
  const currentPattern = pattern;
  try {
    const settings = getSettings(await chrome.storage.local.get('disabledSites'));
    const disabledSites = settings.disabledSites.filter((site) => site !== currentPattern);
    if (paused) disabledSites.push(currentPattern);
    await chrome.storage.local.set({ disabledSites });
    status.textContent = paused ? 'Suggestions are paused on this site.' : 'Suggestions are enabled on this site.';
    await refresh();
  } catch (error) { report(error); }
}

enableSite.addEventListener('click', () => { void setSitePaused(false); });
disableSite.addEventListener('click', () => { void setSitePaused(true); });

void refresh().catch(report);
