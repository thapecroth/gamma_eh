import { access, mkdtemp, readdir, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { homedir, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { chromium } from '@playwright/test';

export async function findChromium(environment = process.env) {
  if (environment.GAMMA_CHROME_PATH) {
    await access(environment.GAMMA_CHROME_PATH);
    return environment.GAMMA_CHROME_PATH;
  }
  try { await access(chromium.executablePath()); return chromium.executablePath(); } catch { /* existing cache below */ }
  const cache = join(homedir(), '.cache/ms-playwright');
  let entries;
  try { entries = await readdir(cache); }
  catch (error) { if (error.code !== 'ENOENT') throw error; entries = []; }
  for (const entry of entries.filter(name => /^chromium-\d+$/u.test(name)).sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]))) {
    for (const relative of ['chrome-linux64/chrome', 'chrome-linux/chrome']) {
      const executable = join(cache, entry, relative);
      try { await access(executable); return executable; } catch { /* next installed candidate */ }
    }
  }
  throw new Error('Install Chromium with npx playwright install chromium, or set GAMMA_CHROME_PATH.');
}

export function browserArguments(environment = process.env) {
  return ['--no-sandbox', ...(environment.GAMMA_TEST_WEBGPU === '1' ? ['--enable-unsafe-webgpu'] : [])];
}

export function capabilityFailure(error) {
  // Do not persist exception bodies, URLs, headers, or provider content.
  const code = error?.code ?? error?.cause?.code;
  const permissionDenied = /EPERM|EACCES|Operation not permitted|Permission denied/u.test(`${code ?? ''} ${error?.message ?? ''} ${error?.cause?.message ?? ''}`);
  return {status: 'failed', code: typeof code === 'string' && /^[A-Z0-9_]+$/u.test(code) ? code : 'CAPABILITY_FAILED', permissionDenied};
}

export function summarizeCapabilities(checks, teacherRequired) {
  const required = ['localHttp', 'chromium', ...(teacherRequired ? ['teacher'] : [])];
  return {scope: teacherRequired ? 'live-pipeline-prerequisites' : 'browser-prerequisites', checks,
    passed: required.every(name => checks[name]?.status === 'passed'),
    blocked: required.filter(name => checks[name]?.status !== 'passed')};
}

export async function withLocalHttp(operation, serverFactory = createServer) {
  const server = serverFactory((_request, response) => response.end('gamma-environment-ok'));
  // Chromium may preconnect without sending a request. server.close() alone
  // waits for those sockets, while the browser is only closed after this helper.
  const sockets = new Set();
  server.on('connection', socket => {
    sockets.add(socket);
    socket.once('close', () => sockets.delete(socket));
  });
  try {
    await new Promise((resolveReady, rejectListen) => {
      server.once('error', rejectListen);
      server.listen(0, '127.0.0.1', resolveReady);
    });
    return await operation(`http://127.0.0.1:${server.address().port}/`);
  } finally {
    if (server.listening) await new Promise(resolveClosed => {
      server.close(resolveClosed);
      for (const socket of sockets) socket.destroy();
    });
  }
}

export async function probeLocalHttp() {
  return withLocalHttp(async origin => {
    const response = await fetch(origin, {signal: AbortSignal.timeout(5000)});
    if (!response.ok || await response.text() !== 'gamma-environment-ok') throw new Error('Local HTTP roundtrip failed');
    return {status: 'passed'};
  });
}

export async function probeChromium(environment = process.env) {
  const prefix = join(tmpdir(), 'gamma-environment-');
  const directory = await mkdtemp(prefix);
  let context;
  try {
    context = await chromium.launchPersistentContext(directory, {
      channel: 'chromium', executablePath: await findChromium(environment), headless: true,
      timeout: 15000, args: browserArguments(environment),
    });
    const page = context.pages()[0] ?? await context.newPage();
    const result = {status: 'passed', version: context.browser()?.version() ?? 'unknown'};
    if (environment.GAMMA_TEST_WEBGPU === '1') {
      const adapter = await withLocalHttp(async origin => {
        await page.goto(origin, {timeout: 5000});
        return page.evaluate(async () => {
          let timer;
          try {
            const result = await Promise.race([
              Promise.resolve(navigator.gpu?.requestAdapter()).then(adapter => adapter ? {
                vendor: adapter.info.vendor, architecture: adapter.info.architecture,
                isFallbackAdapter: adapter.info.isFallbackAdapter,
              } : null),
              new Promise(resolve => { timer = setTimeout(() => resolve({timedOut: true}), 10000); }),
            ]);
            return result;
          } finally { clearTimeout(timer); }
        });
      });
      if (adapter?.timedOut) throw Object.assign(new Error('WebGPU adapter probe timed out'), {code: 'WEBGPU_TIMEOUT'});
      if (!adapter) throw Object.assign(new Error('Required WebGPU adapter unavailable'), {code: 'WEBGPU_UNAVAILABLE'});
      result.adapter = adapter;
    }
    return result;
  } finally {
    try { if (context) await context.close(); }
    finally {
      if (dirname(directory) === tmpdir() && directory.startsWith(prefix)) await rm(directory, {recursive: true, force: true});
    }
  }
}

export function teacherCatalogUrl(base) {
  const parsed = new URL(base);
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error('Teacher endpoint must be HTTP(S), with no embedded credentials, query, or fragment');
  }
  if (parsed.protocol === 'http:' && !['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname)) {
    throw new Error('Remote teacher endpoints must use HTTPS');
  }
  return base.replace(/\/$/u, '') + '/models';
}

export async function probeTeacher(environment = process.env) {
  if (!environment.TEACHER_BASE_URL || !environment.TEACHER_API_KEY) throw new Error('Teacher base URL and API key are required');
  const response = await fetch(teacherCatalogUrl(environment.TEACHER_BASE_URL), {
    headers: {Authorization: `Bearer ${environment.TEACHER_API_KEY}`}, signal: AbortSignal.timeout(10000), redirect: 'error',
  });
  if (!response.ok) return {status: 'failed', code: `HTTP_${response.status}`, permissionDenied: false};
  const payload = await response.json();
  const available = Array.isArray(payload?.data) && payload.data.some(row => row?.id === (environment.TEACHER_MODEL ?? 'glm-5.3-flash'));
  return available ? {status: 'passed'} : {status: 'failed', code: 'MODEL_UNAVAILABLE', permissionDenied: false};
}

export async function checkEnvironment(environment = process.env, probes = {}) {
  const teacherRequired = environment.GAMMA_REQUIRE_TEACHER === '1' || Boolean(environment.TEACHER_BASE_URL);
  const checks = {};
  for (const [name, probe] of [['localHttp', probes.localHttp ?? probeLocalHttp], ['chromium', probes.chromium ?? (() => probeChromium(environment))]]) {
    try { checks[name] = await probe(); } catch (error) { checks[name] = capabilityFailure(error); }
  }
  if (teacherRequired) {
    try { checks.teacher = await (probes.teacher ?? (() => probeTeacher(environment)))(); }
    catch (error) { checks.teacher = capabilityFailure(error); }
  } else checks.teacher = {status: 'not-requested'};
  return summarizeCapabilities(checks, teacherRequired);
}
