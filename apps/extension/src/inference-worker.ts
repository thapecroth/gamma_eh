import { analyzeRules, analyzeText, type AnalysisResult } from '@gamma/engine';
import type { CheckMessage } from './protocol';

type Request = CheckMessage & { modelBaseUrl: string; wasmBaseUrl: string };
const pending = new Map<string, Request>();
let busy = false;

async function drain() {
  if (busy) return;
  busy = true;
  try {
    while (pending.size) {
      const [key, request] = pending.entries().next().value!;
      pending.delete(key);
      try {
        const started = performance.now();
        const result: AnalysisResult = request.useAI
          ? await analyzeText(request.text, { modelBaseUrl: request.modelBaseUrl, wasmBaseUrl: request.wasmBaseUrl })
          : { text: request.text, suggestions: analyzeRules(request.text), backend: 'rules', elapsedMs: performance.now() - started };
        self.postMessage({ requestId: request.requestId, result });
      } catch (error) {
        self.postMessage({ requestId: request.requestId, error: error instanceof Error ? error.message : 'Analysis failed.' });
      }
    }
  } finally { busy = false; }
}

self.addEventListener('message', (event: MessageEvent<Request>) => {
  const request = event.data;
  const key = request.requestId.slice(0, request.requestId.lastIndexOf(':'));
  const previous = pending.get(key);
  if (previous) self.postMessage({ requestId: previous.requestId, error: 'Superseded by a newer draft.' });
  if (!previous && pending.size >= 32) {
    self.postMessage({ requestId: request.requestId, error: 'The local checker is busy. Try again shortly.' });
    return;
  }
  pending.set(key, request);
  void drain();
});
