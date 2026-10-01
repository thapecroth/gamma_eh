import { runInNewContext } from 'node:vm';
import { describe, expect, it } from 'vite-plus/test';
import { patchWebGPUAdapterRequest, webGPUAdapterVitePlugin } from '../scripts/webgpu-adapter-options.mjs';

function request(source, platform, options, userAgentData) {
  let received;
  const navigator = {platform, userAgentData, gpu: {requestAdapter(value) { received = value; return Promise.resolve(null); }}};
  runInNewContext(patchWebGPUAdapterRequest(source), {navigator, options});
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

  it('only transforms the JAX backend and supports worker build module ids', () => {
    const plugin = webGPUAdapterVitePlugin();
    expect(plugin.transform(jax, '/project/apps/web/src/app.ts')).toBeUndefined();
    expect(plugin.transform(jax, '/project/node_modules/@jax-js/jax/dist/backend-example.js?worker').code).toContain('delete supported.powerPreference');
  });
});
