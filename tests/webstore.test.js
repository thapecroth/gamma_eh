import { describe, expect, it } from 'vite-plus/test';
import { storeConfig, submitToStore } from '../scripts/webstore-submit.mjs';

const environment = {CWS_PUBLISHER_ID: 'fictional-publisher', CWS_EXTENSION_ID: 'a'.repeat(32),
  CWS_CLIENT_ID: 'fictional-client', CWS_CLIENT_SECRET: 'fictional-secret', CWS_REFRESH_TOKEN: 'fictional-refresh'};
const config = storeConfig(environment);
const archive = Buffer.from('fictional ZIP body');
const revision = (version, state = 'PUBLISHED') => ({state, distributionChannels: [{crxVersion: version, deployPercentage: 100}]});
const status = overrides => ({name: config.name, publishedItemRevisionStatus: revision('0.1.0'), ...overrides});
const upload = overrides => ({name: config.name, crxVersion: '0.1.1', uploadState: 'SUCCEEDED', ...overrides});

function mock(responses) {
  const calls = [];
  const request = async (url, options) => {
    calls.push({url, options});
    const response = responses.shift();
    if (response instanceof Error) throw response;
    if (!response) throw new Error('Unexpected request');
    return {ok: response.http === undefined || response.http === 200, status: response.http ?? 200, json: async () => response};
  };
  return {calls, request, wait: async () => {}};
}
const token = {access_token: 'fictional-access'};

describe('Chrome Web Store submission', () => {
  it('rejects missing credentials and malformed IDs without disclosing values', () => {
    expect(() => storeConfig({...environment, CWS_REFRESH_TOKEN: ''})).toThrow(/Missing CWS_REFRESH_TOKEN/);
    expect(() => storeConfig({...environment, CWS_PUBLISHER_ID: '../foreign'})).toThrow(/Invalid/);
    expect(() => storeConfig({...environment, CWS_EXTENSION_ID: 'z'.repeat(32)})).toThrow(/Invalid/);
  });

  it('waits for asynchronous upload success, then submits with review and warning checks', async () => {
    const transport = mock([token, status(), upload({uploadState: 'IN_PROGRESS', crxVersion: undefined}),
      status({lastAsyncUploadState: 'IN_PROGRESS'}), status({lastAsyncUploadState: 'SUCCEEDED'}), {name: config.name, state: 'PENDING_REVIEW'}]);
    expect(await submitToStore({archive, version: '0.1.1', config}, transport)).toEqual({version: '0.1.1', state: 'PENDING_REVIEW', alreadySubmitted: false});
    const sent = transport.calls.find(call => call.url.endsWith(':upload'));
    expect(sent.options.body).toBe(archive);
    expect(sent.url).toBe(`https://chromewebstore.googleapis.com/upload/v2/${config.name}:upload`);
    const publish = transport.calls.at(-1);
    expect(JSON.parse(publish.options.body)).toEqual({publishType: 'DEFAULT_PUBLISH', skipReview: false, blockOnWarnings: true});
    expect(transport.calls.every(call => call.options.redirect === 'error')).toBe(true);
  });

  it('never publishes failed, missing, or indefinitely processing uploads', async () => {
    for (const state of ['FAILED', 'NOT_FOUND', undefined, 'IN_PROGRESS']) {
      const transport = mock([token, status(), upload({uploadState: state}), ...Array.from({length: 24}, () => status({lastAsyncUploadState: 'IN_PROGRESS'}))]);
      await expect(submitToStore({archive, version: '0.1.1', config}, transport)).rejects.toThrow(/did not complete/);
      expect(transport.calls.some(call => call.url.endsWith(':publish'))).toBe(false);
    }
  });

  it('refuses an incorrect item or version and a store policy warning', async () => {
    for (const response of [status({name: 'foreign'}), status({warned: true}), status({takenDown: true})]) {
      const transport = mock([token, response]);
      await expect(submitToStore({archive, version: '0.1.1', config}, transport)).rejects.toThrow(/dashboard attention/);
      expect(transport.calls).toHaveLength(2);
    }
    const transport = mock([token, status(), upload({crxVersion: '0.2.0'})]);
    await expect(submitToStore({archive, version: '0.1.1', config}, transport)).rejects.toThrow(/different item or version/);
    expect(transport.calls).toHaveLength(3);
  });

  it('requires first dashboard publication and preserves a different pending submission', async () => {
    for (const response of [status({publishedItemRevisionStatus: undefined}), status({submittedItemRevisionStatus: revision('0.2.0', 'PENDING_REVIEW')})]) {
      const transport = mock([token, response]);
      await expect(submitToStore({archive, version: '0.1.1', config}, transport)).rejects.toThrow(/dashboard/);
      expect(transport.calls).toHaveLength(2);
    }
  });

  it('recognizes the same version already published or pending without uploading twice', async () => {
    for (const state of ['PUBLISHED', 'PENDING_REVIEW', 'STAGED']) {
      const response = state === 'PUBLISHED' ? status({publishedItemRevisionStatus: revision('0.1.1')})
        : status({submittedItemRevisionStatus: revision('0.1.1', state)});
      const transport = mock([token, response]);
      expect(await submitToStore({archive, version: '0.1.1', config}, transport)).toEqual({version: '0.1.1', state, alreadySubmitted: true});
      expect(transport.calls).toHaveLength(2);
    }
  });

  it('does not expose credential-bearing errors or Google response bodies', async () => {
    for (const response of [new Error('private-token=secret'), {http: 401, error: 'private-token=secret'}]) {
      const transport = mock([response]);
      const error = await submitToStore({archive, version: '0.1.1', config}, transport).catch(failure => failure);
      expect(error).toBeInstanceOf(Error);
      expect(error.message).toMatch(/OAuth refresh.*(request failed|HTTP 401)/);
      expect(error.message).not.toMatch(/private-token|secret/);
      expect(transport.calls).toHaveLength(1);
    }
  });
});
