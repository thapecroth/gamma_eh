import { describe, expect, it } from 'vite-plus/test';
import { firefoxManifest } from '../scripts/firefox-manifest.mjs';

describe('Firefox manifest', () => {
  it('preserves shared permissions and privacy while replacing Chrome-only infrastructure', () => {
    const source = {manifest_version: 3, version: '0.1.0', minimum_chrome_version: '116',
      permissions: ['storage', 'offscreen', 'scripting'], background: {service_worker: 'background.mjs'},
      host_permissions: ['https://*/*'], content_scripts: [{js: ['content.js']}],
      content_security_policy: {extension_pages: "script-src 'self' 'wasm-unsafe-eval'"}};
    const manifest = firefoxManifest(source);
    expect(manifest.background).toEqual({scripts: ['background.js']});
    expect(manifest.permissions).toEqual(['storage', 'scripting']);
    expect(manifest.minimum_chrome_version).toBeUndefined();
    expect(manifest.browser_specific_settings.gecko.data_collection_permissions.required).toEqual(['none']);
    expect(manifest.host_permissions).toEqual(source.host_permissions);
    expect(manifest.content_security_policy).toEqual(source.content_security_policy);
    expect(source.permissions).toContain('offscreen');
  });
});
