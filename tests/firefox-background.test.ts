import { afterEach, expect, it, vi } from 'vite-plus/test';
import type { CheckResponse } from '../apps/extension/src/protocol';

afterEach(() => vi.unstubAllGlobals());

it('checks permitted Firefox requests in a local worker and rejects foreign senders', async () => {
  vi.resetModules();
  vi.stubGlobal('GAMMA_FIREFOX', true);
  let listener: (message: unknown, sender: chrome.runtime.MessageSender, respond: (response: CheckResponse) => void) => unknown;
  const posted = vi.fn();
  class LocalWorker {
    onmessage?: (event: {data: CheckResponse}) => void;
    postMessage(message: {requestId: string}) {
      posted(message);
      queueMicrotask(() => this.onmessage?.({data: {requestId: message.requestId}}));
    }
  }
  const api = {
    runtime: {id: 'gamma', getURL: (path: string) => `moz-extension://gamma/${path}`,
      onMessage: {addListener: (value: typeof listener) => {listener = value;}},
      sendMessage: vi.fn(), getContexts: vi.fn()},
    permissions: {contains: vi.fn(async () => true), onRemoved: {addListener: vi.fn()}},
    scripting: {getRegisteredContentScripts: vi.fn(async () => [{id: 'gamma-site-test'}])},
  };
  vi.stubGlobal('chrome', api);
  vi.stubGlobal('Worker', LocalWorker);
  await import('../apps/extension/src/background');
  const request = {target: 'background', action: 'analyze', requestId: 'one', text: 'A freind.', useAI: false};
  const sender = {id: 'gamma', tab: {id: 1}, frameId: 0, url: 'https://example.com/'} as chrome.runtime.MessageSender;
  const respond = vi.fn();
  expect(listener!(request, {...sender, id: 'foreign'}, respond)).toBeUndefined();
  expect(listener!(request, {...sender, frameId: 1}, respond)).toBeUndefined();
  expect(posted).not.toHaveBeenCalled();
  expect(listener!(request, sender, respond)).toBe(true);
  await vi.waitFor(() => expect(respond).toHaveBeenCalledWith({requestId: 'one'}));
  expect(posted).toHaveBeenCalledWith(expect.objectContaining({text: request.text, modelBaseUrl: 'moz-extension://gamma/models/'}));
  expect(api.runtime.sendMessage).not.toHaveBeenCalled();
  expect(api.runtime.getContexts).not.toHaveBeenCalled();
  api.permissions.contains.mockResolvedValue(false);
  respond.mockClear();
  listener!(request, sender, respond);
  await vi.waitFor(() => expect(respond).toHaveBeenCalledWith({requestId: 'one', siteDisabled: true}));
  expect(posted).toHaveBeenCalledOnce();
});
