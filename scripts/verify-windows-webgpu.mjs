import assert from 'node:assert/strict';

// Simulate Windows descriptors while executing the real browser model. This
// proves the adapter options without claiming physical Windows GPU coverage.
export async function verifyWindowsWebGPU(context, origin, gpuRequired, scopes) {
  const evidence = [];
  const checks = scopes.flatMap(scope => ['available', 'unavailable'].map(mode => ({scope, mode})));
  for (const {scope, mode} of checks) {
    const page = await context.newPage();
    const warnings = [];
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
      if (/powerPreference.*ignored|VerifyEachNodeIsAssignedToAnEp/u.test(message.text())) warnings.push(message.text());
    });
    try {
      await page.addInitScript(mode => {
        Object.defineProperty(navigator, 'userAgentData', {value: {platform: 'Windows'}});
        const requestAdapter = navigator.gpu?.requestAdapter.bind(navigator.gpu);
        globalThis.__gammaAdapterRequests = [];
        const record = async options => {
          globalThis.__gammaAdapterRequests.push({hasPowerPreference: options != null && 'powerPreference' in options, powerPreference: options?.powerPreference});
          return mode === 'unavailable' ? null : await requestAdapter?.(options) ?? null;
        };
        if (navigator.gpu) navigator.gpu.requestAdapter = record;
        else Object.defineProperty(navigator, 'gpu', {value: {requestAdapter: record}});
      }, mode);
      await page.goto(origin + '/fixture.html');
      const result = await page.evaluate(async ({origin, scope, mode}) => {
        const engine = await import(origin + '/test-engine.mjs');
        let requests = globalThis.__gammaAdapterRequests;
        const text = 'The students has a notebook.';
        const result = scope.url ? await new Promise((resolve, reject) => {
          requests = [];
          const bootstrap = new URL('/test-windows-worker.mjs', origin);
          bootstrap.searchParams.set('module', new URL(scope.url, origin).href);
          bootstrap.searchParams.set('mode', mode);
          const worker = new Worker(bootstrap, {type: 'module'});
          const timer = setTimeout(() => { worker.terminate(); reject(new Error('Windows worker check timed out')); }, 90_000);
          const finish = (error, result) => {
            clearTimeout(timer);
            worker.terminate();
            if (error) reject(new Error(error));
            else resolve(result);
          };
          worker.onerror = event => finish(event.message);
          worker.onmessageerror = () => finish('Windows worker response could not be decoded');
          worker.onmessage = ({data}) => {
            if (data.adapterRequest) requests.push(data.adapterRequest);
            else if (data.ready) worker.postMessage({requestId: 'windows-check:1', text, useAI: true, modelBaseUrl: origin + '/models/'});
            else finish(data.error, data.result);
          };
        }) : await engine.analyzeText(text, {
          modelBaseUrl: origin + '/models/', wasmBaseUrl: origin + '/runtime/', preferWebGPU: true,
        });
        return {backend: result.backend, modelError: result.modelError,
          actual: engine.applySuggestions(result.text, result.suggestions),
          modelEdit: result.suggestions.some(edit => edit.source === 'model'),
          requests};
      }, {origin, scope, mode});
      assert.equal(result.modelError, undefined);
      assert.equal(result.actual, 'The students have a notebook.');
      assert.equal(result.modelEdit, true);
      assert(result.requests.length > 0, 'The model must attempt GPU adapter initialization');
      assert(result.requests.every(options => !options.hasPowerPreference), 'Windows adapter requests must omit the ignored power preference');
      if (mode === 'unavailable') assert.equal(result.backend, 'wasm');
      else if (gpuRequired) assert.equal(result.backend, 'webgpu');
      assert.deepEqual(warnings, [], 'Windows startup must not emit ignored-preference or expected placement warnings');
      assert.deepEqual(errors, [], 'Windows startup must not throw uncaught exceptions');
      evidence.push({scope: scope.name, mode, ...result, warnings});
    } finally { await page.close(); }
  }
  return evidence;
}
