import { runInNewContext } from 'node:vm';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build } from 'esbuild';
import { describe, expect, it } from 'vite-plus/test';
import { patchWebGPUAdapterRequest, webGPUAdapterEsbuildPlugin, webGPUAdapterVitePlugin } from '../scripts/webgpu-adapter-options.mjs';

function request(source, ...descriptors) {
  return recordRequest(patchWebGPUAdapterRequest(source), ...descriptors);
}

function recordRequest(source, platform, options, userAgentData, userAgent) {
  let received;
  const navigator = {platform, userAgentData, userAgent, gpu: {requestAdapter(value) { received = value; return Promise.resolve(null); }}};
  runInNewContext(source, {navigator, options});
  return received;
}

describe('Windows WebGPU adapter options', () => {
  const jax = 'navigator.gpu.requestAdapter({ powerPreference: "high-performance" });';
  const native = 'navigator.gpu.requestAdapter(options).then;';

  it.each(['Win32', 'Windows', 'windows'])('omits the ignored preference on %s', platform => {
    expect(request(jax, platform)).toEqual({});
  });

  it.each(['MacIntel', 'Linux x86_64', 'Android'])('preserves GPU selection on %s', platform => {
    expect(request(jax, platform)).toEqual({powerPreference: 'high-performance'});
  });

  it('uses client hints when the legacy platform is unavailable or reduced', () => {
    expect(request(jax, '', undefined, {platform: 'Windows'})).toEqual({});
    expect(request(jax, 'Win32', undefined, {platform: 'macOS'})).toEqual({powerPreference: 'high-performance'});
  });

  it('falls back past empty descriptors to the Windows user agent', () => {
    expect(request(jax, '', undefined, {platform: ''}, 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')).toEqual({});
  });

  it('preserves feature level and fallback options without mutating the original object', () => {
    const options = {powerPreference: 'high-performance', forceFallbackAdapter: false, featureLevel: 'core'};
    // Avoid requiring a promise from the recorder; only the request options matter.
    const received = request(native, 'Win32', options);
    expect(received).toEqual({forceFallbackAdapter: false, featureLevel: 'core'});
    expect(options.powerPreference).toBe('high-performance');
    expect(request(native, 'Linux x86_64', options)).toBe(options);
  });

  it('fails visibly when the upstream request is missing or duplicated', () => {
    expect(() => patchWebGPUAdapterRequest('navigator.gpu.requestAdapter();')).toThrow(/found 0/u);
    expect(() => patchWebGPUAdapterRequest(jax + jax)).toThrow(/found 2/u);
    expect(() => patchWebGPUAdapterRequest(jax + 'navigator.gpu.requestAdapter(getOptions());')).toThrow(/found 2/u);
    expect(() => patchWebGPUAdapterRequest(jax + 'navigator.gpu.requestAdapter();')).toThrow(/found 2/u);
  });

  it('transforms renamed JAX modules, supports worker ids, and leaves other packages alone', () => {
    const plugin = webGPUAdapterVitePlugin();
    expect(plugin.transform(jax, '/project/apps/web/src/app.ts')).toBeUndefined();
    expect(plugin.transform(jax, '/project/node_modules/other/dist/backend-example.js')).toBeUndefined();
    expect(plugin.transform('export const version = 1;', '/project/node_modules/@jax-js/jax/dist/index.js')).toBeUndefined();
    const id = '/project/node_modules/@jax-js/jax/lib/runtime-renamed.mjs';
    expect(plugin.transform(jax, id + '?worker').code).toContain('delete supported.powerPreference');
    expect(plugin.transform(jax, id).code).toContain('delete supported.powerPreference');
    expect(() => plugin.buildEnd()).not.toThrow();
  });

  it('fails Vite builds when adapter requests disappear or split across modules', () => {
    const plugin = webGPUAdapterVitePlugin();
    plugin.transform('export const version = 1;', '/project/node_modules/@jax-js/jax/dist/index.js');
    expect(() => plugin.buildEnd()).toThrow(/found 0/u);
    plugin.transform(jax, '/project/node_modules/@jax-js/jax/dist/first.js');
    plugin.transform(jax, '/project/node_modules/@jax-js/jax/dist/second.js');
    expect(() => plugin.buildEnd()).toThrow(/found 2/u);
    plugin.buildStart();
    expect(() => plugin.buildEnd()).not.toThrow();
  });

  it('fails visibly when the runtime starts using an unsupported adapter expression', () => {
    const plugin = webGPUAdapterVitePlugin();
    expect(() => plugin.transform('gpu.requestAdapter(options);', '/project/node_modules/@jax-js/jax/dist/runtime.js')).toThrow(/found 0/u);
  });
});

describe('esbuild adapter integration', () => {
  async function bundle(modules, entry = "import './node_modules/@jax-js/jax/dist/index.js';", subdirectory = 'dist') {
    const directory = await mkdtemp(join(tmpdir(), 'gamma-adapter-test-'));
    try {
      const distribution = join(directory, 'node_modules/@jax-js/jax', subdirectory);
      await mkdir(distribution, {recursive: true});
      await writeFile(join(directory, 'entry.js'), entry);
      for (const [name, source] of Object.entries(modules)) await writeFile(join(distribution, name), source);
      return await build({entryPoints: [join(directory, 'entry.js')], bundle: true, write: false,
        format: 'esm', platform: 'browser', logLevel: 'silent', plugins: [webGPUAdapterEsbuildPlugin()]});
    } finally { await rm(directory, {recursive: true, force: true}); }
  }

  it('patches the adapter after an upstream backend filename change', async () => {
    const result = await bundle({'index.js': "import './runtime-next.js';", 'runtime-next.js': 'navigator.gpu.requestAdapter({powerPreference: "high-performance"});'});
    expect(recordRequest(result.outputFiles[0].text, 'Win32')).toEqual({});
  });

  it('patches the adapter after an upstream distribution directory change', async () => {
    const result = await bundle({'runtime.js': 'navigator.gpu.requestAdapter({powerPreference: "high-performance"});'},
      "import './node_modules/@jax-js/jax/lib/runtime.js';", 'lib');
    expect(recordRequest(result.outputFiles[0].text, 'Win32')).toEqual({});
  });

  it('rejects a JAX bundle with missing or multiple adapter modules', async () => {
    await expect(bundle({'index.js': 'console.log("no adapter");'})).rejects.toThrow(/found 0/u);
    await expect(bundle({'index.js': "import './first.js'; import './second.js';",
      'first.js': 'navigator.gpu.requestAdapter({powerPreference: "high-performance"});',
      'second.js': 'navigator.gpu.requestAdapter({powerPreference: "low-power"});'})).rejects.toThrow(/found 2/u);
  });

  it('permits extension entrypoints that do not load JAX', async () => {
    const result = await bundle({}, 'console.log("rules only");');
    expect(result.outputFiles[0].text).toContain('rules only');
  });
});
