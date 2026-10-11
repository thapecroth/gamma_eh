import type { AnalysisResult, ModelLoadProgress } from '../../../packages/engine/src/types';

export interface CheckRequest { requestId: number; text: string; useAI: boolean; preferWebGPU?: boolean }
export interface WorkerResponse { requestId: number; result?: AnalysisResult; error?: string; progress?: ModelLoadProgress }
export interface CheckUpdate extends WorkerResponse { pendingAI?: boolean; retryingCPU?: boolean }

// Bound work from outside the inference worker: GPU compilation can block its
// event loop as well as leave an asynchronous adapter/readback promise pending.
const AI_TIMEOUT_MS = 30_000;

export class WebChecker {
  private readonly rules: Worker;
  private ai?: Worker;
  private timer?: ReturnType<typeof setTimeout>;
  private request?: CheckRequest;
  private rulesResult?: AnalysisResult;
  private aiResult?: AnalysisResult;
  private failure?: string;
  private preferWebGPU = true;
  private progress?: ModelLoadProgress;

  constructor(private readonly onUpdate: (update: CheckUpdate) => void) {
    this.rules = this.createWorker();
    this.rules.onmessage = (event: MessageEvent<WorkerResponse>) => {
      if (!this.matches(event.data)) return;
      if (event.data.error) this.onUpdate(event.data);
      else {
        this.rulesResult = event.data.result;
        this.publish();
      }
    };
    this.rules.onerror = () => {
      if (this.request) this.onUpdate({requestId: this.request.requestId, error: 'Local rules could not start.'});
    };
  }

  private createWorker(): Worker {
    return new Worker(new URL('./inference-worker.ts', import.meta.url), {type: 'module'});
  }

  check(request: CheckRequest): void {
    this.request = request;
    this.rulesResult = undefined;
    this.aiResult = undefined;
    this.rules.postMessage({...request, useAI: false});
    if (!request.useAI) {
      this.cancelAI();
    } else if (!this.failure) this.checkAI();
  }

  private matches(message: WorkerResponse): boolean {
    return message.requestId === this.request?.requestId &&
      (!message.result || message.result.text === this.request.text);
  }

  private checkAI(): void {
    try {
      if (!this.ai) {
        const worker = this.createWorker();
        this.ai = worker;
        worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
          if (this.ai !== worker) return; // Ignore callbacks queued before termination.
          clearTimeout(this.timer);
          this.timer = undefined;
          if (event.data.progress) {
            // Loading belongs to the active worker, even if its original draft
            // has been superseded. Bytes arriving keep a slow download alive.
            this.progress = event.data.progress;
            this.armTimeout();
            this.publish();
            return;
          }
          if (!this.matches(event.data)) {
            this.armTimeout(); // Completed old work; the latest draft is queued.
            return;
          }
          if (event.data.error || event.data.result?.modelError) {
            this.fail(event.data.error ?? event.data.result!.modelError!);
          } else {
            this.aiResult = event.data.result;
            this.progress = undefined;
            this.publish();
          }
        };
        worker.onerror = () => { if (this.ai === worker) this.retryCPU(); };
      }
      this.ai.postMessage({...this.request, preferWebGPU: this.preferWebGPU});
      this.armTimeout();
    } catch { this.fail('Local AI could not start.'); }
  }

  private armTimeout(): void {
    // Typing must not postpone recovery from a hung first check indefinitely.
    this.timer ??= setTimeout(() => {
      if (this.progress?.stage === 'downloading') this.fail('The model download stopped responding.');
      else this.retryCPU();
    }, AI_TIMEOUT_MS);
  }

  private retryCPU(): void {
    this.stopAI();
    if (!this.preferWebGPU) {
      this.fail('Local AI took too long to respond.');
      return;
    }
    this.preferWebGPU = false;
    this.publish();
    this.checkAI();
  }

  private fail(message: string): void {
    this.stopAI();
    this.failure = message;
    this.publish();
  }

  private publish(): void {
    if (!this.request) return;
    const result = this.aiResult ?? this.rulesResult;
    if (!result) return;
    this.onUpdate({requestId: this.request.requestId,
      result: this.failure ? {...result, modelError: this.failure} : result,
      pendingAI: this.request.useAI && !this.failure && !this.aiResult,
      retryingCPU: !this.preferWebGPU, progress: this.progress});
  }

  private stopAI(): void {
    clearTimeout(this.timer);
    this.timer = undefined;
    this.ai?.terminate();
    this.ai = undefined;
    this.progress = undefined;
  }

  cancelAI(): void {
    this.stopAI();
    this.failure = undefined;
    this.preferWebGPU = true;
  }

  dispose(): void {
    this.stopAI();
    this.rules.terminate();
    this.request = undefined;
  }
}
