import { describe, expect, it } from 'vite-plus/test';
import { browserArguments, capabilityFailure, checkEnvironment, summarizeCapabilities, teacherCatalogUrl } from '../scripts/browser-environment.mjs';

describe('browser/pipeline prerequisites', () => {
  it('does not confuse browser readiness with teacher availability', () => {
    const checks = {localHttp: {status: 'passed'}, chromium: {status: 'passed'}, teacher: {status: 'not-requested'}};
    expect(summarizeCapabilities(checks, false)).toMatchObject({passed: true, scope: 'browser-prerequisites', blocked: []});
    expect(summarizeCapabilities(checks, true)).toMatchObject({passed: false, scope: 'live-pipeline-prerequisites', blocked: ['teacher']});
    expect(summarizeCapabilities({}, false).passed).toBe(false);
  });
  it('collects every failed capability sequentially without storing sensitive errors', async () => {
    const calls = [];
    const error = Object.assign(new Error('private-provider-body secret-value'), {code: 'EPERM'});
    const report = await checkEnvironment({GAMMA_REQUIRE_TEACHER: '1'}, {
      localHttp: async () => { calls.push('localHttp'); throw error; },
      chromium: async () => { calls.push('chromium'); throw new Error('setsockopt: Operation not permitted'); },
      teacher: async () => { calls.push('teacher'); throw error; },
    });
    expect(calls).toEqual(['localHttp', 'chromium', 'teacher']);
    expect(report.blocked).toEqual(calls);
    expect(JSON.stringify(report)).not.toMatch(/private-provider-body|secret-value/u);
    expect(report.checks.localHttp).toEqual({status: 'failed', code: 'EPERM', permissionDenied: true});
    expect(capabilityFailure({message: 'fetch failed', cause: {code: 'EACCES'}}).permissionDenied).toBe(true);
    expect(capabilityFailure({code: 'secret-value', message: 'secret-value'}).code).toBe('CAPABILITY_FAILED');
  });
  it('makes the teacher optional for app-only E2E and required when configured', async () => {
    const passed = async () => ({status: 'passed'});
    const browser = await checkEnvironment({}, {localHttp: passed, chromium: passed});
    expect(browser.passed).toBe(true);
    expect(browser.checks.teacher.status).toBe('not-requested');
    const complete = await checkEnvironment({TEACHER_BASE_URL: 'http://localhost:8317/v1'}, {
      localHttp: passed, chromium: passed, teacher: passed,
    });
    expect(complete.scope).toBe('live-pipeline-prerequisites');
    expect(complete.passed).toBe(true);
  });
  it('uses a supported Chromium launch and only opts into WebGPU when requested', () => {
    expect(browserArguments({})).toEqual(['--no-sandbox']);
    expect(browserArguments({GAMMA_TEST_WEBGPU: '1'})).toContain('--enable-unsafe-webgpu');
  });
  it('rejects secret-bearing and remote plaintext endpoints', () => {
    expect(teacherCatalogUrl('http://127.0.0.1:8317/v1/')).toBe('http://127.0.0.1:8317/v1/models');
    expect(teacherCatalogUrl('https://teacher.example.test/v1')).toBe('https://teacher.example.test/v1/models');
    for (const endpoint of ['https://user:secret@example.test/v1', 'https://example.test/v1?token=secret',
      'https://example.test/v1#secret', 'http://remote.example.test/v1', 'file:///tmp/provider']) {
      expect(() => teacherCatalogUrl(endpoint)).toThrow();
    }
  });
});
