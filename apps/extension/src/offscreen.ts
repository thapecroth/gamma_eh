import { isCheckMessage, type CheckResponse } from './protocol';

const worker = new Worker(chrome.runtime.getURL('inference-worker.mjs'), { type: 'module' });
const pending = new Map<string, { resolve: (response: CheckResponse) => void; timeout: ReturnType<typeof setTimeout> }>();

worker.onmessage = (event: MessageEvent<CheckResponse>) => {
  const request = pending.get(event.data.requestId);
  if (!request) return;
  clearTimeout(request.timeout);
  pending.delete(event.data.requestId);
  request.resolve(event.data);
};
worker.onerror = () => {
  for (const [requestId, request] of pending) {
    clearTimeout(request.timeout);
    request.resolve({ requestId, error: 'The local model worker stopped. Reload the extension to restart it.' });
  }
  pending.clear();
};

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse: (response: CheckResponse) => void) => {
  if (!isCheckMessage(message) || message.target !== 'offscreen' || sender.id !== chrome.runtime.id || sender.tab) return;
  if (pending.has(message.requestId)) {
    sendResponse({ requestId: message.requestId, error: 'A check with this identifier is already running.' });
    return;
  }
  const timeout = setTimeout(() => {
    pending.delete(message.requestId);
    sendResponse({ requestId: message.requestId, error: 'Local analysis timed out. Try a shorter passage.' });
  }, 90_000);
  pending.set(message.requestId, { resolve: sendResponse, timeout });
  worker.postMessage({ ...message, modelBaseUrl: chrome.runtime.getURL('models/') });
  return true;
});
