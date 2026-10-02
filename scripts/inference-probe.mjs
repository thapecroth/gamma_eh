// Served only by the analysis harness. No instrumentation is shipped in products.
export const workloadVersion = 1;
export const workloadNames = ['unchanged', 'edited', 'fresh', 'nearby'];
export function paragraph(seed = 0) {
  return Array.from({length: 24}, (_, index) => `The students has ${seed + index + 1} notebooks in the classroom.`).join(' ');
}

export async function initializeProbe(engineUrl, modelBaseUrl, backend, coldOnly = false) {
  const adapters = [];
  if (backend === 'webgpu') {
    const gpu = navigator.gpu;
    if (!gpu) throw new Error('WEBGPU_UNAVAILABLE');
    const requestAdapter = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = async function observedAdapter(...args) {
      const adapter = await requestAdapter(...args);
      if (adapter) {
        const info = adapter.info;
        adapters.push({vendor: info.vendor, architecture: info.architecture, description: info.description,
          isFallbackAdapter: info.isFallbackAdapter,
          software: /swiftshader|llvmpipe|software/iu.test(`${info.vendor} ${info.description}`) || info.isFallbackAdapter === true,
          attribution: 'engine-initialization',
          effectiveRequest: Object.fromEntries(['powerPreference', 'forceFallbackAdapter', 'featureLevel']
            .filter(key => args[0]?.[key] !== undefined).map(key => [key, args[0][key]]))});
      }
      return adapter;
    };
  }
  function usedAdapter() {
    if (backend !== 'webgpu') return null;
    if (adapters.length !== 1) throw new Error('WEBGPU_ADAPTER_ATTRIBUTION_AMBIGUOUS');
    return adapters[0];
  }
  const engine = await import(engineUrl);
  const base = paragraph(0);
  const options = {modelBaseUrl, preferWebGPU: backend === 'webgpu', mode: 'combined', maxPasses: 1};
  const counters = {runtimeCalls: 0, modelRuns: 0, cacheHits: 0, cacheMisses: 0, tokenizerCalls: 0, tokenizerMs: 0, runtimeWallMs: 0};
  const traceEvents = [];
  let capture = false;
  const originalRun = engine.JaxSession.prototype.run;
  engine.JaxSession.prototype.run = async function observedInference(ids) {
    const started = performance.now();
    const result = await originalRun.call(this, ids);
    const elapsed = performance.now() - started;
    counters.runtimeCalls++; counters.runtimeWallMs += elapsed;
    if (capture && traceEvents.length < 10000) traceEvents.push({name: `JaxSession.run (${ids.length} tokens): elapsed await`, cat: 'runtime-wall', ph: 'X',
      ts: started * 1000, dur: elapsed * 1000, pid: 1, tid: 1, args: {backend, tokens: ids.length, meaning: 'Wall duration includes async waits; not device CPU time'}});
    return result;
  };
  const originalChunks = engine.WordPieceTokenizer.prototype.chunks;
  engine.WordPieceTokenizer.prototype.chunks = function observedTokenizer(...args) {
    const started = performance.now();
    const result = originalChunks.apply(this, args);
    counters.tokenizerCalls++; counters.tokenizerMs += performance.now() - started;
    return result;
  };
  if (engine.InferenceCache) {
    const originalGet = engine.InferenceCache.prototype.get;
    engine.InferenceCache.prototype.get = function observedCache(key) {
      const value = originalGet.call(this, key);
      counters[value ? 'cacheHits' : 'cacheMisses']++;
      return value;
    };
  }
  async function check(text, overrides = {}) {
    const started = performance.now();
    const result = await engine.analyzeText(text, {...options, ...overrides});
    const elapsedMs = performance.now() - started;
    if (result.modelError || result.backend !== backend) throw new Error('MODEL_FAILURE_OR_BACKEND_FALLBACK');
    if (!Number.isInteger(result.modelRuns) || result.modelRuns < 0) throw new Error('MISSING_MODEL_RUN_COUNTER');
    counters.modelRuns += result.modelRuns;
    if (capture && traceEvents.length < 10000) traceEvents.push({name: 'analyzeText: elapsed', cat: 'analysis-wall', ph: 'X',
      ts: started * 1000, dur: elapsedMs * 1000, pid: 1, tid: 2, args: {backend}});
    return {elapsedMs, suggestions: result.suggestions, modelRuns: result.modelRuns};
  }
  function reset(detail = false) {
    for (const key of Object.keys(counters)) counters[key] = 0;
    traceEvents.length = 0; capture = detail;
  }
  const manifest = await (await fetch(modelBaseUrl + 'manifest.json')).json();
  const vocabulary = await (await fetch(modelBaseUrl + 'vocab.txt')).text();
  const tokenizer = new engine.WordPieceTokenizer(vocabulary);
  function keys(text) { return tokenizer.chunks(text, manifest.maxSequenceLength).map(chunk => chunk.ids.join(',')); }
  function validateWorkload(seed) {
    const ids = keys(paragraph(seed));
    if (ids.length !== 24 || new Set(ids).size !== 24) throw new Error('WORKLOAD_TOKEN_WINDOWS_NOT_DISTINCT');
    return ids;
  }
  const originalKeys = validateWorkload(0);
  const editKeys = keys(base.replace('24 notebooks', '30000 notebooks'));
  if (editKeys.length !== 24 || editKeys.filter((value, index) => value !== originalKeys[index]).length !== 1) throw new Error('EDIT_WORKLOAD_NOT_ONE_WINDOW');
  if (coldOnly) {
    globalThis.gammaProbe = {async cold() {
      reset(true);
      const result = await check(base);
      if (result.modelRuns < 1) throw new Error('MODEL_WAS_NOT_EXERCISED');
      return {elapsedMs: result.elapsedMs, iterations: 1, traceEvents, counters: {...counters}, traceTruncated: false, adapter: usedAdapter()};
    }};
    return {coldReady: true};
  }
  reset();
  const cold = await check(base);
  if (cold.modelRuns < 1) throw new Error('MODEL_WAS_NOT_EXERCISED');
  reset();
  const warm = await check(base);
  const warmCounters = {...counters};
  reset();
  const edited = await check(base.replace('24 notebooks', '30000 notebooks'));
  const editedCounters = {...counters};
  if (engine.InferenceCache && (warm.modelRuns !== 0 || edited.modelRuns !== 1 || warmCounters.runtimeCalls !== 0 || editedCounters.runtimeCalls !== 1)) throw new Error('INCONSISTENT_CACHE_WORKLOAD_COUNTERS');
  let rawSession;
  async function raw() {
    if (!rawSession) {
      const bytes = new Uint8Array(await (await fetch(modelBaseUrl + 'model.onnx')).arrayBuffer());
      rawSession = await engine.JaxSession.create(bytes, backend === 'webgpu');
      if (rawSession.backend !== backend) throw new Error('RAW_SESSION_BACKEND_FALLBACK');
      for (let length = 8; length <= 16; length++) await rawSession.run([101, ...Array(length - 2).fill(1996), 102]);
    }
    return rawSession;
  }
  const adapter = usedAdapter();
  function inputPlan(name, count, ordinal) {
    const plan = Array.from({length: count}, (_, index) => name === 'fresh' ? paragraph(100000 + (ordinal + index) * 24) :
      name === 'edited' ? base.replace('24 notebooks', `${30000 + ordinal + index} notebooks`) :
        name === 'nearby' ? [101, ...Array(6 + index % 9).fill(1996), 102] : base);
    if (name === 'edited' || name === 'fresh') {
      const seen = new Set();
      for (const text of plan) {
        const ids = keys(text);
        if (ids.length !== 24 || new Set(ids).size !== 24) throw new Error('WORKLOAD_TOKEN_WINDOWS_NOT_DISTINCT');
        if (name === 'edited' && ids.filter((value, index) => value !== originalKeys[index]).length !== 1) throw new Error('EDIT_WORKLOAD_NOT_ONE_WINDOW');
        for (const key of name === 'edited' ? ids.slice(-1) : ids) {
          if (seen.has(key)) throw new Error('WORKLOAD_VARIANTS_NOT_DISTINCT'); seen.add(key);
        }
      }
    }
    return plan;
  }
  let profilePlan = [];
  async function runOne(name, input) {
    if (name === 'nearby') {
      const started = performance.now();
      await (await raw()).run(input);
      return {elapsedMs: performance.now() - started};
    }
    return check(input);
  }
  globalThis.gammaProbe = {
    evidence: {coldMs: cold.elapsedMs, coldModelRuns: cold.modelRuns, warmCounters, editedCounters, adapter,
      fixture: {version: workloadVersion, chunks: 24, uniqueTokenWindows: 24, oneEditChangedWindows: 1, base}},
    async timing(name, repetitions, ordinal) {
      const plan = inputPlan(name, repetitions, ordinal);
      if (name === 'nearby') await raw();
      await check(base); reset();
      const elapsed = [], outputs = [];
      for (let index = 0; index < repetitions; index++) {
        const result = await runOne(name, plan[index]); elapsed.push(result.elapsedMs);
        if (name !== 'nearby') outputs.push({id: `${name}:${ordinal + index}`, corrected: engine.applySuggestions(plan[index], result.suggestions), suggestions: result.suggestions});
      }
      if (name === 'fresh' && (counters.cacheHits !== 0 || counters.runtimeCalls !== repetitions * 24)) throw new Error('FRESH_WORKLOAD_IS_NOT_ALL_MISSES');
      if (counters.runtimeCalls !== counters.modelRuns && name !== 'nearby') throw new Error('MODEL_RUN_COUNTER_MISMATCH');
      return {elapsed, counters: {...counters}, outputs};
    },
    async prepareProfile(name) {
      profilePlan = inputPlan(name, {edited: 512, fresh: 64, nearby: 9, unchanged: 1}[name], 1000000);
      if (name === 'nearby') await raw(); await check(base); reset(true);
    },
    async profile(name, durationMs) {
      const started = performance.now(); let iterations = 0;
      do { await runOne(name, profilePlan[iterations++ % profilePlan.length]); } while (performance.now() - started < durationMs && iterations < 100000);
      capture = false;
      if (name === 'fresh' && (counters.cacheHits !== 0 || counters.runtimeCalls !== iterations * 24)) throw new Error('PROFILE_FRESH_WORKLOAD_IS_NOT_ALL_MISSES');
      if (name === 'edited' && engine.InferenceCache && (counters.runtimeCalls !== iterations || counters.cacheMisses !== iterations)) throw new Error('PROFILE_EDIT_WORKLOAD_IS_NOT_ONE_MISS');
      if (name === 'unchanged' && engine.InferenceCache && counters.runtimeCalls !== 0) throw new Error('PROFILE_UNCHANGED_WORKLOAD_IS_NOT_ALL_HITS');
      return {elapsedMs: performance.now() - started, iterations, counters: {...counters}, traceEvents,
        traceTruncated: traceEvents.length >= 10000};
    },
    profileInputs(iterations) { return profilePlan.filter(row => typeof row === 'string').slice(0, Math.min(iterations, profilePlan.length)); },
    async quality(rows) {
      const outputs = [];
      for (const row of rows) for (const mode of ['model', 'combined']) for (const maxPasses of [1, 2]) {
        const result = await check(row.source, {mode, maxPasses});
        outputs.push({id: `${row.id}:${mode}:${maxPasses}`, corrected: engine.applySuggestions(row.source, result.suggestions), suggestions: result.suggestions});
      }
      return outputs;
    },
    async smoke(cases) {
      const outputs = [];
      for (let index = 0; index < cases.length; index++) {
        const row = cases[index], result = await check(row.source);
        const corrected = engine.applySuggestions(row.source, result.suggestions);
        if (row.target !== undefined && corrected !== row.target) throw new Error('BUNDLED_REGRESSION_FAILED');
        outputs.push({id: row.id ?? `regression-${index}`, corrected, suggestions: result.suggestions});
      }
      return outputs;
    },
    async logits() {
      const outputs = [];
      for (const length of [3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64]) {
        const output = await (await raw()).run([101, ...Array(length - 2).fill(1996), 102]);
        outputs.push({length, dims: output.dims, values: Array.from(output.data)});
      }
      return outputs;
    },
    dispose() { rawSession?.dispose(); },
  };
  return globalThis.gammaProbe.evidence;
}
