import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';
import { WebChecker, type CheckRequest, type WorkerResponse } from '../apps/web/src/checker';
import type { AnalysisResult } from '../packages/engine/src/types';

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage?: (event: {data: WorkerResponse}) => void;
  onerror?: () => void;
  postMessage = vi.fn<(request: CheckRequest) => void>();
  terminate = vi.fn();
  constructor() { FakeWorker.instances.push(this); }
  reply(request: CheckRequest, backend: AnalysisResult['backend'] = 'rules', modelError?: string) {
    this.onmessage?.({data: {requestId: request.requestId,
      result: {text: request.text, backend, suggestions: [], elapsedMs: 1, ...(modelError ? {modelError} : {})}}});
  }
}

const first: CheckRequest = {requestId: 1, text: 'She have a freind.', useAI: true};
const update = vi.fn();
let checker: WebChecker;
beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('Worker', FakeWorker);
  FakeWorker.instances = [];
  update.mockReset();
  checker = new WebChecker(update);
});
afterEach(() => { checker.dispose(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('web checker startup and recovery', () => {
  it('publishes rules before AI and does not downgrade a completed AI result', () => {
    checker.check(first);
    const [rules, ai] = FakeWorker.instances;
    rules.reply(first);
    expect(update).toHaveBeenLastCalledWith(expect.objectContaining({pendingAI: true, result: expect.objectContaining({backend: 'rules'})}));
    ai.reply(first, 'webgpu');
    rules.reply(first);
    expect(update).toHaveBeenLastCalledWith(expect.objectContaining({pendingAI: false, result: expect.objectContaining({backend: 'webgpu'})}));
    vi.advanceTimersByTime(60_000);
    expect(ai.terminate).not.toHaveBeenCalled();
  });

  it('turns off stalled AI immediately and ignores its late result and error', () => {
    checker.check(first);
    const [rules, ai] = FakeWorker.instances;
    const off = {...first, requestId: 2, useAI: false};
    checker.check(off);
    expect(ai.terminate).toHaveBeenCalledOnce();
    rules.reply(off);
    ai.reply(first, 'webgpu');
    ai.onerror?.();
    vi.advanceTimersByTime(60_000);
    expect(update).toHaveBeenCalledTimes(1);
    expect(update.mock.lastCall?.[0]).toMatchObject({requestId: 2, pendingAI: false, result: {backend: 'rules'}});
    expect(FakeWorker.instances).toHaveLength(2);
  });

  it('restarts a stalled worker on CPU for the latest draft without extending the deadline on typing', () => {
    checker.check(first);
    const [rules, ai] = FakeWorker.instances;
    vi.advanceTimersByTime(29_000);
    const latest = {...first, requestId: 2, text: 'The students has a notebook.'};
    checker.check(latest);
    rules.reply(latest);
    vi.advanceTimersByTime(1000);
    const cpu = FakeWorker.instances[2];
    expect(ai.terminate).toHaveBeenCalledOnce();
    expect(cpu.postMessage).toHaveBeenCalledWith({...latest, preferWebGPU: false});
    expect(update.mock.lastCall?.[0]).toMatchObject({pendingAI: true, retryingCPU: true});
    ai.reply(first, 'webgpu');
    cpu.reply(latest, 'wasm');
    expect(update.mock.lastCall?.[0]).toMatchObject({requestId: 2, pendingAI: false, result: {text: latest.text, backend: 'wasm'}});
    vi.advanceTimersByTime(60_000);
    expect(cpu.terminate).not.toHaveBeenCalled();
  });

  it('bounds the CPU attempt, retains rules on later drafts, and allows an explicit retry', () => {
    checker.check(first);
    const rules = FakeWorker.instances[0];
    rules.reply(first);
    vi.advanceTimersByTime(60_000);
    expect(FakeWorker.instances).toHaveLength(3);
    expect(FakeWorker.instances[2].terminate).toHaveBeenCalledOnce();
    expect(update.mock.lastCall?.[0]).toMatchObject({pendingAI: false, result: {backend: 'rules', modelError: expect.stringContaining('too long')}});
    const next = {...first, requestId: 2};
    checker.check(next);
    rules.reply(next);
    expect(update.mock.lastCall?.[0].result.modelError).toBeTruthy();
    expect(FakeWorker.instances).toHaveLength(3);
    checker.check({...first, requestId: 3, useAI: false});
    checker.check({...first, requestId: 4});
    expect(FakeWorker.instances).toHaveLength(4);
    expect(FakeWorker.instances[3].postMessage).toHaveBeenCalledWith(expect.objectContaining({preferWebGPU: true}));
  });

  it('handles model failure before rules return and suppresses repeated failed downloads', () => {
    checker.check(first);
    const [rules, ai] = FakeWorker.instances;
    ai.reply(first, 'rules', 'Could not load model.onnx (503)');
    expect(update).not.toHaveBeenCalled();
    rules.reply(first);
    expect(update.mock.lastCall?.[0]).toMatchObject({pendingAI: false, result: {modelError: expect.stringContaining('503')}});
    checker.check({...first, requestId: 2});
    expect(FakeWorker.instances).toHaveLength(2);
  });

  it('does not reset the deadline with an old terminated worker or apply a stale same-text reply', () => {
    checker.check(first);
    const [rules, ai] = FakeWorker.instances;
    checker.check({...first, requestId: 2});
    rules.reply(first);
    ai.reply(first, 'webgpu');
    expect(update).not.toHaveBeenCalled();
    vi.advanceTimersByTime(30_000);
    vi.advanceTimersByTime(29_000);
    ai.reply(first, 'webgpu');
    vi.advanceTimersByTime(1000);
    expect(FakeWorker.instances[2].terminate).toHaveBeenCalledOnce();
  });

  it('recovers a worker startup error on CPU, then stops if the CPU worker also fails', () => {
    checker.check(first);
    FakeWorker.instances[0].reply(first);
    FakeWorker.instances[1].onerror?.();
    expect(FakeWorker.instances[2].postMessage).toHaveBeenCalledWith(expect.objectContaining({preferWebGPU: false}));
    FakeWorker.instances[2].onerror?.();
    expect(update.mock.lastCall?.[0]).toMatchObject({pendingAI: false, result: {backend: 'rules', modelError: expect.any(String)}});
  });

  it('releases both workers and deadlines when the page unmounts', () => {
    checker.check(first);
    checker.dispose();
    vi.advanceTimersByTime(60_000);
    expect(FakeWorker.instances).toHaveLength(2);
    for (const worker of FakeWorker.instances) {
      expect(worker.terminate).toHaveBeenCalled();
      worker.reply(first);
    }
    expect(update).not.toHaveBeenCalled();
  });

  it('allows a progressing download beyond 30 seconds, then bounds a stalled download without fetching it twice', () => {
    checker.check(first);
    const [rules, ai] = FakeWorker.instances;
    rules.reply(first);
    for (let chunk = 1; chunk <= 4; chunk++) {
      vi.advanceTimersByTime(20_000);
      ai.onmessage?.({data: {requestId: first.requestId, progress: {stage: 'downloading', loadedBytes: chunk * 1000}}});
      expect(ai.terminate).not.toHaveBeenCalled();
    }
    vi.advanceTimersByTime(30_000);
    expect(ai.terminate).toHaveBeenCalledOnce();
    expect(FakeWorker.instances).toHaveLength(2);
    expect(update.mock.lastCall?.[0]).toMatchObject({pendingAI: false, result: {modelError: expect.stringContaining('download')}});
  });

  it('can reset failure immediately without waiting for a debounced rules-only check', () => {
    checker.check(first);
    FakeWorker.instances[1].reply(first, 'rules', 'Unavailable');
    checker.cancelAI();
    checker.check({...first, requestId: 3});
    expect(FakeWorker.instances).toHaveLength(3);
    expect(FakeWorker.instances[2].postMessage).toHaveBeenCalledWith(expect.objectContaining({preferWebGPU: true}));
  });
});
