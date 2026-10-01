import type { CheckMessage, CheckResponse } from './protocol';

let worker: Worker | undefined;
const pending = new Map<string, { resolve: (response: CheckResponse) => void; timeout: ReturnType<typeof setTimeout> }>();

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(chrome.runtime.getURL('inference-worker.mjs'), { type: 'module' });

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
    worker?.terminate();
    worker = undefined;
  };

  return worker;
}

export async function checkLocally(message: CheckMessage): Promise<CheckResponse> {
  const activeWorker = ensureWorker();
  if (pending.has(message.requestId)) return Promise.resolve({requestId: message.requestId, error: 'A check with this identifier is already running.'});
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      pending.delete(message.requestId);
      resolve({requestId: message.requestId, error: 'Local analysis timed out. Try a shorter passage.'});
    }, 90_000);
    pending.set(message.requestId, {resolve, timeout});
    try {
      activeWorker.postMessage({...message, modelBaseUrl: chrome.runtime.getURL('models/')});
    } catch (error) {
      clearTimeout(timeout);
      pending.delete(message.requestId);
      resolve({requestId: message.requestId, error: error instanceof Error ? error.message : 'Analysis failed.'});
    }
  });
}
