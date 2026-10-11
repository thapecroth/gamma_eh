import { analyzeRules } from '../../../packages/engine/src/rules';
import type { AnalysisResult } from '../../../packages/engine/src/types';
import type { CheckRequest } from './checker';

let pending: CheckRequest | null = null;
let busy = false;

async function drain() {
  if (busy) return;
  busy = true;
  try {
    while (pending) {
      const { requestId, text, useAI, preferWebGPU } = pending;
      pending = null;
      try {
        const started = performance.now();
        const result: AnalysisResult = useAI
          ? await (await import('@gamma/engine')).analyzeText(text, {
            modelBaseUrl: `${import.meta.env.BASE_URL}models/`, preferWebGPU,
            onModelLoadProgress: progress => self.postMessage({requestId, progress}),
          })
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
