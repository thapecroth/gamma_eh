import { describe, expect, it } from 'vite-plus/test';
import { webBase } from '../scripts/web-base.mjs';

describe('same-origin web deployment paths', () => {
  it('supports root builds and repository Pages paths', () => {
    expect(webBase({})).toBe('/');
    expect(webBase({ GAMMA_WEB_BASE: '/gamma_eh/' })).toBe('/gamma_eh/');
    expect(webBase({ GAMMA_WEB_BASE: '/nested/demo/' })).toBe('/nested/demo/');
  });
  it.each(['https://cdn.example/', '//cdn.example/', 'gamma_eh/', '/gamma_eh', '/../', '/nested/../', '/demo/?text=private', '/demo/#draft', '/%2e%2e/', ''])('rejects an unsafe deployment base: %s', base => {
    expect(() => webBase({ GAMMA_WEB_BASE: base })).toThrow(/directory path/u);
  });
});
