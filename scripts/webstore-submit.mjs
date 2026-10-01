import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { releaseVersion } from './package-release.mjs';

const run = promisify(execFile);
const api = 'https://chromewebstore.googleapis.com';

export function storeConfig(environment) {
  const keys = ['CWS_PUBLISHER_ID', 'CWS_EXTENSION_ID', 'CWS_CLIENT_ID', 'CWS_CLIENT_SECRET', 'CWS_REFRESH_TOKEN'];
  for (const key of keys) if (!environment[key]?.trim()) throw new Error(`Missing ${key}. Configure the chrome-web-store environment.`);
  if (!/^[a-zA-Z0-9_-]+$/u.test(environment.CWS_PUBLISHER_ID) || !/^[a-p]{32}$/u.test(environment.CWS_EXTENSION_ID)) throw new Error('Invalid Chrome Web Store publisher or extension ID.');
  return {name: `publishers/${environment.CWS_PUBLISHER_ID}/items/${environment.CWS_EXTENSION_ID}`,
    clientId: environment.CWS_CLIENT_ID, clientSecret: environment.CWS_CLIENT_SECRET, refreshToken: environment.CWS_REFRESH_TOKEN};
}

export async function verifiedArchive(directory, version) {
  releaseVersion(version, version);
  const filename = `gamma-eh-chrome-v${version}.zip`;
  const archive = await readFile(join(directory, filename));
  const lines = (await readFile(join(directory, 'SHA256SUMS.txt'), 'utf8')).trim().split('\n');
  const matches = lines.filter(line => line.endsWith(`  ${filename}`));
  if (matches.length !== 1 || matches[0] !== `${createHash('sha256').update(archive).digest('hex')}  ${filename}`) throw new Error('Chrome release checksum mismatch.');
  const {stdout} = await run('unzip', ['-p', join(directory, filename), 'manifest.json'], {maxBuffer: 64 * 1024});
  const manifest = JSON.parse(stdout);
  if (manifest.manifest_version !== 3 || manifest.version !== version || !manifest.icons?.['128']) throw new Error('Chrome release manifest does not match the requested store version.');
  return archive;
}

export async function submitToStore({archive, version, config}, {request = fetch, wait = delay} = {}) {
  releaseVersion(version, version);
  // Never include Google response bodies or transport exceptions in logs: they
  // can contain OAuth credentials or private account information.
  async function json(url, options, label) {
    let response;
    try { response = await request(url, {...options, redirect: 'error', signal: AbortSignal.timeout(60_000)}); }
    catch { throw new Error(`${label} request failed. Check the developer dashboard before retrying.`); }
    if (!response.ok) throw new Error(`${label} failed (HTTP ${response.status}). Check the developer dashboard.`);
    try { return await response.json(); } catch { throw new Error(`${label} returned invalid JSON.`); }
  }
  const token = await json('https://oauth2.googleapis.com/token', {method: 'POST', body: new URLSearchParams({
    client_id: config.clientId, client_secret: config.clientSecret, refresh_token: config.refreshToken, grant_type: 'refresh_token',
  })}, 'OAuth refresh');
  if (typeof token.access_token !== 'string' || !token.access_token || /[\r\n]/u.test(token.access_token)) throw new Error('OAuth refresh did not return a usable access token.');
  const headers = {Authorization: `Bearer ${token.access_token}`};
  const statusUrl = `${api}/v2/${config.name}:fetchStatus`;
  const status = await json(statusUrl, {headers}, 'Store status');
  if (status.name !== config.name || status.takenDown || status.warned) throw new Error('Store item identity or policy status requires dashboard attention.');
  const published = status.publishedItemRevisionStatus;
  if (published?.state !== 'PUBLISHED') throw new Error('Complete the first unlisted publication in the developer dashboard before enabling automated updates.');
  const hasVersion = revision => revision?.distributionChannels?.some(channel => channel.crxVersion === version);
  if (hasVersion(published)) return {version, state: 'PUBLISHED', alreadySubmitted: true};
  const submitted = status.submittedItemRevisionStatus;
  if (['PENDING_REVIEW', 'STAGED'].includes(submitted?.state)) {
    if (hasVersion(submitted)) return {version, state: submitted.state, alreadySubmitted: true};
    throw new Error('Another version is under review or staged. Resolve it in the developer dashboard first.');
  }
  const uploaded = await json(`${api}/upload/v2/${config.name}:upload`, {method: 'POST', headers: {...headers, 'Content-Type': 'application/zip'}, body: archive}, 'Store upload');
  if (uploaded.name !== config.name || uploaded.crxVersion && uploaded.crxVersion !== version) throw new Error('Store upload returned a different item or version.');
  let state = uploaded.uploadState;
  for (let attempt = 0; state === 'IN_PROGRESS' && attempt < 24; attempt++) {
    await wait(5_000);
    const polled = await json(statusUrl, {headers}, 'Upload status');
    if (polled.name !== config.name) throw new Error('Upload status returned a different item.');
    state = polled.lastAsyncUploadState;
  }
  if (state !== 'SUCCEEDED') throw new Error('Store upload did not complete successfully. Check the dashboard before retrying.');
  const result = await json(`${api}/v2/${config.name}:publish`, {method: 'POST', headers: {...headers, 'Content-Type': 'application/json'},
    body: JSON.stringify({publishType: 'DEFAULT_PUBLISH', skipReview: false, blockOnWarnings: true})}, 'Store submission');
  if (result.name !== config.name || !['PENDING_REVIEW', 'PUBLISHED'].includes(result.state)) throw new Error('Store submission was not accepted for review or publication. Check the dashboard.');
  return {version, state: result.state, alreadySubmitted: false};
}

async function main() {
  const [directory, version] = process.argv.slice(2);
  if (!directory || !version) throw new Error('Usage: webstore-submit.mjs RELEASE_DIRECTORY X.Y.Z');
  const config = storeConfig(process.env);
  const archive = await verifiedArchive(resolve(directory), version);
  console.log(JSON.stringify(await submitToStore({archive, version, config})));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
