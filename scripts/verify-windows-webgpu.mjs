import assert from 'node:assert/strict';

// Simulate Windows descriptors while executing the real browser model. This
// proves the adapter options without claiming physical Windows GPU coverage.
export async function verifyWindowsWebGPU(context, origin, gpuRequired) {
  const evidence = [];
  for (const mode of ['available', 'unavailable']) {
    const page = await context.newPage();
    const warnings = [];
    page.on('console', message => {
      if (/powerPreference.*ignored|VerifyEachNodeIsAssignedToAnEp/u.test(message.text())) warnings.push(message.text());
    });
    try {
      await page.addInitScript(mode => {
        Object.defineProperty(navigator, 'userAgentData', {value: {platform: 'Windows'}});
        const requestAdapter = navigator.gpu?.requestAdapter.bind(navigator.gpu);
        globalThis.__gammaAdapterRequests = [];
        const record = async options => {
          globalThis.__gammaAdapterRequests.push({powerPreference: options?.powerPreference});
          return mode === 'unavailable' ? null : await requestAdapter?.(options) ?? null;
        };
        if (navigator.gpu) navigator.gpu.requestAdapter = record;
        else Object.defineProperty(navigator, 'gpu', {value: {requestAdapter: record}});
      }, mode);
      await page.goto(origin + '/fixture.html');
      const result = await page.evaluate(async origin => {
        const engine = await import(origin + '/test-engine.mjs');
        const result = await engine.analyzeText('The students has a notebook.', {
          modelBaseUrl: origin + '/models/', wasmBaseUrl: origin + '/runtime/', preferWebGPU: true,
        });
        return {backend: result.backend, modelError: result.modelError,
          actual: engine.applySuggestions(result.text, result.suggestions),
          modelEdit: result.suggestions.some(edit => edit.source === 'model'),
          requests: globalThis.__gammaAdapterRequests};
      }, origin);
      assert.equal(result.modelError, undefined);
      assert.equal(result.actual, 'The students have a notebook.');
      assert.equal(result.modelEdit, true);
      assert(result.requests.length > 0, 'The model must attempt GPU adapter initialization');
      assert(result.requests.every(options => options.powerPreference === undefined), 'Windows adapter requests must omit the ignored power preference');
      if (mode === 'unavailable') assert.equal(result.backend, 'wasm');
      else if (gpuRequired) assert.equal(result.backend, 'webgpu');
      assert.deepEqual(warnings, [], 'Windows startup must not emit ignored-preference or expected placement warnings');
      evidence.push({mode, ...result, warnings});
    } finally { await page.close(); }
  }
  return evidence;
}
