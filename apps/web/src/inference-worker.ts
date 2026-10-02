import { analyzeRules, analyzeText, type AnalysisResult } from '@gamma/engine';

interface CheckRequest { requestId: number; text: string; useAI: boolean }
let pending: CheckRequest | null = null;
let busy = false;

async function drain() {
  if (busy) return;
  busy = true;
  try {
    while (pending) {
      const { requestId, text, useAI } = pending;
      pending = null;
      try {
        const started = performance.now();
        const result: AnalysisResult = useAI
          ? await analyzeText(text, { modelBaseUrl: `${import.meta.env.BASE_URL}models/` })
          : { text, suggestions: analyzeRules(text), backend: 'rules', elapsedMs: performance.now() - started };
        self.postMessage({ requestId, result });
      } catch (error) {
        self.postMessage({ requestId, error: error instanceof Error ? error.message : 'Analysis failed.' });
      }
    }
  } finally { busy = false; }
}

self.addEventListener('message', (event: MessageEvent<CheckRequest>) => {
  pending = event.data; // Only the latest draft waits behind the active check.
  void drain();
});
