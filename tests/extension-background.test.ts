import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';
import type { CheckResponse } from '../apps/extension/src/protocol';

let onMessage: (message: unknown, sender: chrome.runtime.MessageSender, respond: (response: CheckResponse) => void) => boolean | void;
let onInstalled: () => Promise<void>;
const request = { target: 'background', action: 'analyze', requestId: 'request-1', text: 'A freind.', useAI: false };
const sender: chrome.runtime.MessageSender = { id: 'gamma-test', tab: { id: 1, index: 0, pinned: false, highlighted: false, windowId: 1, active: true, incognito: false,
  frozen: false, selected: true, discarded: false, autoDiscardable: true, groupId: -1 }, frameId: 0, url: 'https://example.org/editor' };

function browser() {
  return {
    runtime: {
      id: 'gamma-test',
      onMessage: { addListener: (listener: typeof onMessage) => { onMessage = listener; } },
      onInstalled: { addListener: (listener: typeof onInstalled) => { onInstalled = listener; } },
      getURL: (path: string) => `chrome-extension://gamma-test/${path}`,
      ContextType: { OFFSCREEN_DOCUMENT: 'OFFSCREEN_DOCUMENT' },
      getContexts: vi.fn(async () => [{}]),
      sendMessage: vi.fn(async () => ({ requestId: request.requestId })),
    },
    permissions: { contains: vi.fn(async () => true), onRemoved: { addListener: vi.fn() } },
    storage: { local: { get: vi.fn(async (): Promise<Record<string, unknown>> => ({})) } },
    scripting: { getRegisteredContentScripts: vi.fn(async () => [{ id: 'gamma-site-legacy' }, { id: 'unrelated-script' }]), unregisterContentScripts: vi.fn(async () => undefined) },
  };
}

let api: ReturnType<typeof browser>;

beforeEach(async () => {
  vi.resetModules();
  api = browser();
  vi.stubGlobal('chrome', api);
  await import('../apps/extension/src/background');
});
afterEach(() => { vi.unstubAllGlobals(); });

async function analyze() {
  const respond = vi.fn<(response: CheckResponse) => void>();
  expect(onMessage(request, sender, respond)).toBe(true);
  await vi.waitFor(() => expect(respond).toHaveBeenCalledOnce());
  return respond.mock.calls[0][0];
}

describe('extension service worker site access', () => {
  it('checks a newly visited website without a dynamic registration', async () => {
    expect(await analyze()).toEqual({ requestId: request.requestId });
    expect(api.runtime.sendMessage).toHaveBeenCalledWith({ ...request, target: 'offscreen' });
    expect(api.scripting.getRegisteredContentScripts).not.toHaveBeenCalled();
  });

  it.each([{ enabled: false }, { disabledSites: ['https://example.org/*'] }])('stops paused requests before inference with settings %j', async (settings) => {
    api.storage.local.get.mockResolvedValue(settings);
    expect(await analyze()).toEqual({ requestId: request.requestId, siteDisabled: true });
    expect(api.runtime.sendMessage).not.toHaveBeenCalled();
    expect(api.runtime.getContexts).not.toHaveBeenCalled();
  });

  it('respects Chrome host access even when suggestions are enabled', async () => {
    api.permissions.contains.mockResolvedValue(false);
    expect(await analyze()).toEqual({ requestId: request.requestId, siteDisabled: true });
    expect(api.runtime.sendMessage).not.toHaveBeenCalled();
  });

  it('rejects foreign senders and child frames before reading settings or text', () => {
    const respond = vi.fn();
    expect(onMessage(request, { ...sender, id: 'another-extension' }, respond)).toBeUndefined();
    expect(onMessage(request, { ...sender, frameId: 1 }, respond)).toBeUndefined();
    expect(respond).not.toHaveBeenCalled();
    expect(api.storage.local.get).not.toHaveBeenCalled();
    expect(api.runtime.sendMessage).not.toHaveBeenCalled();
  });

  it('removes only obsolete site registrations on installation or update', async () => {
    await onInstalled();
    expect(api.scripting.unregisterContentScripts).toHaveBeenCalledWith({ ids: ['gamma-site-legacy'] });
    expect(api.storage.local.get).not.toHaveBeenCalled();
  });
});
