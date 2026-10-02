import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdir, readFile, realpath, writeFile} from 'node:fs/promises';
import {platform, release, tmpdir} from 'node:os';
import {dirname, isAbsolute, join, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {build, version as esbuildVersion} from 'esbuild';
import {chromium} from '@playwright/test';
import {acquireLock, changedProductPaths, git, jsonFile, runId, sourceStamp} from './agent-runtime.mjs';
import {browserArguments, findChromium, withLocalHttp} from './browser-environment.mjs';
import {buildPaths} from './paths.mjs';
import {runtimeManifest} from './inference-runtime.mjs';
import {webGPUAdapterEsbuildPlugin} from './webgpu-adapter-options.mjs';
import {compareLogits, compareOutputs, digest, escapeHtml, flamegraphSvg, frozenQuality, hostSample, percentile, policy,
  profileTree, sourceResolver, swapAdvanced, timingGate, validateAssetSet, validateFrozenQuality} from './inference-analysis.mjs';
import {paragraph, workloadNames} from './inference-probe.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const guardedSource = ['tokenizer.ts', 'edit-tags.ts', 'guards.ts', 'rules.ts', 'spelling.ts', 'dictionary.generated.ts', 'edits.ts', 'edit-history.ts'];
const protectedHarness = ['scripts/analyze-inference.mjs', 'scripts/inference-analysis.mjs', 'scripts/inference-probe.mjs',
  'scripts/browser-environment.mjs', 'scripts/inference-runtime.mjs', 'scripts/webgpu-adapter-options.mjs',
  'scripts/paths.mjs', 'scripts/agent-runtime.mjs', 'data/regression.json', 'training/import_jfleg.py'];

function options() {
  const {values} = parseArgs({options: {baseline: {type: 'string'}, candidate: {type: 'string', default: root},
    anchor: {type: 'string'}, 'quality-dir': {type: 'string'}, output: {type: 'string'}, webgpu: {type: 'boolean'},
    'include-holdout': {type: 'boolean'}, trials: {type: 'string', default: '5'},
    iterations: {type: 'string', default: '25'}, 'profile-ms': {type: 'string', default: '750'}, help: {type: 'boolean'}}});
  if (values.help) return null;
  const [nodeMajor, nodeMinor] = process.versions.node.split('.').map(Number);
  assert(nodeMajor === 24 && nodeMinor >= 11, 'Analysis requires Node 24.11 or newer in the 24.x line');
  assert(!process.env.GAMMA_MODEL_DIR && !process.env.GAMMA_BUILD_PROFILE, 'Analysis requires bundled models/browser; clear alternate model/profile variables');
  for (const [name, low, high] of [['trials', 5, 20], ['iterations', 5, 100], ['profile-ms', 500, 5000]]) {
    assert(/^\d+$/u.test(values[name]) && Number(values[name]) >= low && Number(values[name]) <= high, `Invalid ${name}: require ${low}..${high}`);
    values[name] = Number(values[name]);
  }
  assert(!values.anchor || values.baseline, 'Anchor requires baseline');
  assert(!values['include-holdout'] || values['quality-dir'], 'Holdout requires frozen quality directory');
  return values;
}

async function snapshot(tree) {
  const stamp = await sourceStamp(tree);
  const guarded = {};
  for (const name of guardedSource) guarded[name] = digest(await readFile(join(tree, 'packages/engine/src', name)));
  return {...stamp, dirtyPaths: (await changedProductPaths(tree)).sort(), guarded,
    lockSha256: digest(await readFile(join(tree, 'package-lock.json')))};
}

async function compile(name, tree, directory, assets) {
  const files = {};
  // Each source tree supplies its own exact model assets; candidate assets never
  // stand in for a baseline. Alternate build profiles are deliberately rejected.
  const modelDir = buildPaths(tree, {}).modelDirectory;
  for (const file of ['manifest.json', 'labels.json', 'vocab.txt', 'model.onnx']) {
    files[file] = await readFile(join(modelDir, file)); assets.set(`/${name}/models/${file}`, files[file]);
  }
  const modelAssets = validateAssetSet(files);
  const outfile = join(directory, `${name}.mjs`);
  let cacheExport = 'export const InferenceCache = undefined;';
  try { await readFile(join(tree, 'packages/engine/src/inference-cache.ts')); cacheExport = "export {InferenceCache} from './packages/engine/src/inference-cache.ts';"; }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const entry = `export * from './packages/engine/src/index.ts';
export {JaxSession} from './packages/engine/src/jax-runtime.ts';
export {WordPieceTokenizer} from './packages/engine/src/tokenizer.ts';
${cacheExport}`;
  const result = await build({stdin: {contents: entry, resolveDir: tree}, absWorkingDir: tree, outfile,
    bundle: true, format: 'esm', platform: 'browser', target: 'chrome116', sourcemap: 'external', metafile: true,
    plugins: [webGPUAdapterEsbuildPlugin()]});
  const bytes = await readFile(outfile); assets.set(`/${name}.mjs`, bytes);
  const sourceMap = JSON.parse(await readFile(outfile + '.map', 'utf8'));
  const inputs = {};
  for (const input of Object.keys(result.metafile.inputs).sort()) if (input !== '<stdin>') {
    inputs[input] = digest(await readFile(isAbsolute(input) ? input : join(tree, input)));
  }
  const dependencyInputs = Object.fromEntries(Object.entries(inputs).filter(([path]) => path.includes('node_modules/')).map(([path, hash]) => [path.replace(/^.*node_modules\//u, ''), hash]).sort(([a], [b]) => a.localeCompare(b)));
  return {modelAssets, bundleSha256: digest(bytes), sourceInputsSha256: digest(JSON.stringify(inputs)), inputs,
    dependencyInputsSha256: digest(JSON.stringify(dependencyInputs)),
    sourceMapSha256: digest(await readFile(outfile + '.map')), sourceMap};
}

function timingSummary(result) {
  assert(result.elapsed.every(value => Number.isFinite(value) && value > 0), 'Invalid unprofiled timing');
  return {medianMs: percentile(result.elapsed, 0.5), p95Ms: percentile(result.elapsed, 0.95), ...result};
}

function wallSvg(events) {
  if (!events.length) return '<p>No wall spans captured.</p>';
  const start = Math.min(...events.map(row => row.ts)), end = Math.max(...events.map(row => row.ts + row.dur));
  // Render a bounded preview. Complete spans remain available in the trace file.
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 115" role="img" aria-label="Elapsed analysis and runtime timeline"><text x="4" y="16">Elapsed wall spans: ${(end - start) / 1000} ms; async waits included</text>${events.slice(0, 2000).map(row => {
    const x = 1280 * (row.ts - start) / Math.max(1, end - start), width = Math.max(0.2, 1280 * row.dur / Math.max(1, end - start));
    return `<g><title>${escapeHtml(row.name)}: ${(row.dur / 1000).toFixed(3)} ms</title><rect x="${x}" y="${row.tid === 1 ? 40 : 75}" width="${width}" height="24" fill="${row.tid === 1 ? '#76a7da' : '#eaaa73'}"/></g>`;
  }).join('')}<text x="4" y="37">JaxSession.run elapsed await (not GPU execution time)</text><text x="4" y="72">analyzeText elapsed</text></svg>`;
}

function htmlReport(report, sections) {
  const rows = Object.entries(report.trials).flatMap(([name, trials]) => !trials.length ? [] : ['cold', ...workloadNames].map(workload => {
    const median = percentile(trials.map(row => workload === 'cold' ? row.coldMs : row[workload].medianMs), 0.5);
    const p95 = workload === 'cold' ? null : percentile(trials.map(row => row[workload].p95Ms), 0.5);
    const calls = workload === 'cold' ? percentile(trials.map(row => row.coldRuntimeCalls), 0.5) : percentile(trials.map(row => row[workload].counters.runtimeCalls), 0.5);
    const hits = workload === 'cold' ? 0 : percentile(trials.map(row => row[workload].counters.cacheHits), 0.5);
    return `<tr><td>${escapeHtml(name)}</td><td>${workload}</td><td>${median.toFixed(3)}</td><td>${p95?.toFixed(3) ?? '—'}</td><td>${calls ?? '—'}</td><td>${hits}</td></tr>`;
  })).join('');
  const adapter = report.trees.candidate?.probe?.adapter;
  const qualityCount = report.parity?.baseline?.quality?.comparisons ?? 0;
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Gamma EH inference analysis</title><style>body{font:16px system-ui;margin:24px;max-width:1400px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:16px}svg{max-width:100%;height:auto}details{margin:16px 0}a{color:#14559c}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style><h1>Inference analysis</h1><p>Decision: <strong>${escapeHtml(report.decision.status)}</strong> (${escapeHtml(report.decision.reason)}). Timing decisions use unprofiled paired trials only.</p><p>Backend: ${escapeHtml(report.backend ?? 'not executed')}${adapter ? `; ${escapeHtml(adapter.description || adapter.vendor)}; software adapter: ${adapter.software}` : ''}. Frozen quality comparisons: ${qualityCount}. ${report.decision.speedup95 ? `Latency reduction 95% interval: ${report.decision.speedup95.map(value => (100 * value).toFixed(1) + '%').join(' to ')}.` : ''}</p><table><thead><tr><th>Variant</th><th>Workload</th><th>Median ms</th><th>Median trial p95 ms</th><th>Runtime calls/trial</th><th>Cache hits/trial</th></tr></thead><tbody>${rows}</tbody></table><p>Cold latency is one observation per trial; other rows aggregate trial summaries. CPU flame width is sampled renderer/main-thread CPU, including garbage collection. Async runtime wall spans include waits. WebGPU device execution, compositor and other worker/process CPU are not measured by this graph. Raw profiles retain idle, program and GC samples.</p><a href="report.json">Machine report</a><details><summary>Gate evidence and raw timing</summary><pre>${escapeHtml(JSON.stringify({decision: report.decision, performance: report.performance, policy: report.policy, quality: report.quality, trials: report.trials}, null, 2))}</pre></details>${sections.join('')}<details><summary>Execution provenance and environment</summary><pre>${escapeHtml(JSON.stringify({harness: report.harness, trees: report.trees, environment: report.environment}, null, 2))}</pre></details></html>`;
}

async function main() {
  const id = runId(); let directory = join(root, 'artifacts/inference-analysis', id);
  let config = {}, browser;
  const locks = [], sections = [];
  let interrupted = false;
  const onSignal = () => { interrupted = true; void browser?.close().catch(() => {}); };
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(signal, onSignal);
  const report = {schema: 1, runId: id, startedAt: new Date().toISOString(), policy,
    decision: {status: 'invalid', exitCode: 1, reason: 'run-incomplete'}, trees: {}, trials: {}, profiles: {}, quality: {complete: false},
    environment: {node: process.version, os: `${platform()} ${release()}`, esbuild: esbuildVersion, runtime: runtimeManifest, host: []}};
  try {
    config = options();
    if (!config) {
      console.log('npm run analyze:inference -- [--candidate WORKTREE] [--baseline INCUMBENT] [--anchor IMMUTABLE] [--quality-dir JFLEG] [--include-holdout] [--webgpu] [--output artifacts/UNIQUE] [--trials 5..20] [--iterations 5..100] [--profile-ms 500..5000]');
      return;
    }
    if (config.output) {
      const selected = resolve(config.output);
      assert(selected.startsWith(join(root, 'artifacts') + sep), 'Output must be a fresh directory under trusted checkout artifacts/');
      directory = selected;
    }
    await mkdir(directory, {recursive: false, mode: 0o700}).catch(async error => {
      if (error.code !== 'ENOENT') throw error;
      await mkdir(dirname(directory), {recursive: true}); await mkdir(directory, {mode: 0o700});
    });
    assert((await realpath(directory)).startsWith((await realpath(root)) + sep + 'artifacts' + sep), 'Artifact symlink escapes ignored directory');
    report.options = {...config, output: directory};
    report.backend = config.webgpu ? 'webgpu' : 'wasm';
    report.environment.temporaryDirectory = tmpdir();
    const trees = {candidate: await realpath(resolve(config.candidate))};
    if (config.baseline) trees.baseline = await realpath(resolve(config.baseline));
    if (config.anchor) trees.anchor = await realpath(resolve(config.anchor));
    const commonDirs = new Map();
    for (const tree of [root, ...Object.values(trees)]) commonDirs.set(resolve(tree, await git(tree, 'rev-parse', '--git-common-dir')), tree);
    for (const [, tree] of [...commonDirs].sort(([a], [b]) => a.localeCompare(b))) locks.push(await acquireLock(tree));
    console.log(JSON.stringify({phase: 'analysis-start', pid: process.pid, runId: id}));
    report.harness = {root, ...await snapshot(root), hashes: {}};
    for (const path of protectedHarness) report.harness.hashes[path] = digest(await readFile(join(root, path)));
    report.harness.sha256 = digest(JSON.stringify(report.harness.hashes));
    report.environment.host.push(await hostSample());
    const assets = new Map([['/', Buffer.from('<!doctype html><title>Local inference analysis</title>')],
      ['/probe.mjs', await readFile(join(root, 'scripts/inference-probe.mjs'))]]);
    for (const [name, tree] of Object.entries(trees)) {
      const stamp = await snapshot(tree), built = await compile(name, tree, directory, assets);
      report.trees[name] = {...stamp, root: tree, ...built}; delete report.trees[name].sourceMap;
      report.trees[name].sourceMapForProfile = built.sourceMap;
      report.trials[name] = [];
      if (name !== 'candidate') {
        for (const key of ['modelAssets', 'guarded', 'lockSha256', 'dependencyInputsSha256']) assert.deepEqual(report.trees[name][key], report.trees.candidate[key], `Comparison ${key} mismatch`);
      }
    }
    const qualityRows = [];
    if (config['quality-dir']) {
      const manifestBytes = await readFile(join(resolve(config['quality-dir']), 'manifest.json'));
      const manifest = JSON.parse(manifestBytes.toString('utf8'));
      const selections = config['include-holdout'] ? ['dev', 'test'] : ['dev'];
      for (const split of selections) qualityRows.push(...validateFrozenQuality(manifest,
        await readFile(join(resolve(config['quality-dir']), `${split}.jsonl`)), split));
      report.quality = {complete: true, selection: selections, manifestSha256: digest(manifestBytes), frozen: frozenQuality,
        rows: qualityRows.length, conditions: ['model:1', 'model:2', 'combined:1', 'combined:2'], selectionPurpose: config['include-holdout'] ? 'final-confirmation' : 'hill-climb-development'};
    }
    const regressionBytes = await readFile(join(root, 'data/regression.json'));
    const regression = JSON.parse(regressionBytes.toString('utf8')).cases;
    assert.equal(regression.length, 26, 'Bundled regression population changed; review trusted policy');
    const smoke = [...regression, ...[0, 100000, 200000].map((seed, index) => ({id: `workload-${index}`, source: paragraph(seed)})),
      {id: 'workload-edit', source: paragraph(0).replace('24 notebooks', '30000 notebooks')}];
    report.regressionSha256 = digest(regressionBytes);
    const qualityOutputs = {}, smokeOutputs = {}, logits = {}, trialOutputs = {}, profileSources = new Set();
    await withLocalHttp(async origin => {
      assert(!interrupted, 'ANALYSIS_INTERRUPTED');
      browser = await chromium.launch({executablePath: await findChromium(), headless: true,
        handleSIGINT: false, handleSIGTERM: false, handleSIGHUP: false,
        args: browserArguments({...process.env, GAMMA_TEST_WEBGPU: config.webgpu ? '1' : '0'})});
      report.environment.chromium = browser.version();
      report.environment.playwright = JSON.parse(await readFile(join(root, 'node_modules/@playwright/test/package.json'), 'utf8')).version;
      async function withPage(name, operation, coldCapture = false) {
        assert(!interrupted, 'ANALYSIS_INTERRUPTED');
        const context = await browser.newContext(), violations = [], errors = [];
        try {
          await context.route('**/*', route => {
            if (new URL(route.request().url()).origin !== new URL(origin).origin) { violations.push('nonlocal-request'); return route.abort(); }
            return route.continue();
          });
          const page = await context.newPage();
          // Playwright's evaluate has no execution timeout. Close its owned
          // context before propagating a deadline, so hung kernels cannot linger.
          const evaluate = page.evaluate.bind(page);
          page.evaluate = async (...args) => {
            let timer;
            try {
              return await Promise.race([evaluate(...args), new Promise((_resolve, reject) => {
                timer = setTimeout(() => reject(new Error('BROWSER_OPERATION_DEADLINE')), policy.browserOperationTimeoutMs);
              })]);
            } catch (error) { await context.close(); throw error; }
            finally { clearTimeout(timer); }
          };
          page.on('pageerror', () => errors.push('page-error'));
          await page.goto(origin);
          let initialization = await page.evaluate(async ({origin, name, backend, coldCapture}) => {
            const {initializeProbe} = await import(origin + 'probe.mjs');
            return initializeProbe(origin + name + '.mjs', origin + name + '/models/', backend, coldCapture);
          }, {origin, name, backend: report.backend, coldCapture});
          if (coldCapture) {
            const cdp = await context.newCDPSession(page); await cdp.send('Profiler.enable');
            await cdp.send('Profiler.setSamplingInterval', {interval: 1000}); await cdp.send('Profiler.start');
            initialization = await page.evaluate(() => globalThis.gammaProbe.cold());
            initialization.cpuProfile = (await cdp.send('Profiler.stop')).profile; await cdp.detach();
          }
          if ('adapter' in report.environment) assert.deepEqual(initialization.adapter, report.environment.adapter, 'Execution adapter changed across comparisons');
          else report.environment.adapter = initialization.adapter;
          const result = await operation(page, initialization);
          assert.equal(violations.length, 0, 'Inference must stay local'); assert.equal(errors.length, 0, 'Browser errors fail analysis');
          return result;
        } finally { await context.close(); }
      }
      // A new context per trial fixes session/cache initial state. Alternate
      // candidate/incumbent order; anchor rotates with them rather than always last.
      for (let trial = 0; trial < config.trials; trial++) {
        const names = Object.keys(trees);
        if (trial % 2) names.reverse();
        report.environment.host.push(await hostSample());
        for (const name of names) await withPage(name, async (page, initialization) => {
          const measurements = {trial, order: names, coldMs: initialization.coldMs, coldRuntimeCalls: initialization.coldRuntimeCalls};
          report.trees[name].probe = initialization;
          trialOutputs[name] ??= [];
          for (const workload of workloadNames) {
            const {outputs, ...timing} = await page.evaluate(
              ({workload, iterations, ordinal}) => globalThis.gammaProbe.timing(workload, iterations, ordinal),
              {workload, iterations: config.iterations, ordinal: 100 + trial * 1000});
            trialOutputs[name].push(...outputs); measurements[workload] = timingSummary(timing);
          }
          report.trials[name].push(measurements);
        });
        report.environment.host.push(await hostSample());
        console.log(JSON.stringify({phase: 'unprofiled-timing', completedTrials: trial + 1, totalTrials: config.trials}));
      }
      for (const name of Object.keys(trees)) {
        report.profiles[name] = {};
        for (const workload of ['cold', ...workloadNames]) await withPage(name, async (page, initialization) => {
          let profile, spans;
          if (workload === 'cold') { const {cpuProfile, ...wall} = initialization; profile = cpuProfile; spans = wall; }
          else {
            await page.evaluate(workload => globalThis.gammaProbe.prepareProfile(workload), workload);
            const cdp = await page.context().newCDPSession(page);
            await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', {interval: 1000});
            await cdp.send('Profiler.start');
            spans = await page.evaluate(({workload, durationMs}) => globalThis.gammaProbe.profile(workload, durationMs, 1000000),
              {workload, durationMs: config['profile-ms']});
            profile = (await cdp.send('Profiler.stop')).profile; await cdp.detach();
            for (const source of await page.evaluate(iterations => globalThis.gammaProbe.profileInputs(iterations), spans.iterations)) profileSources.add(source);
          }
          const namePrefix = `${name}-${workload}`;
          await jsonFile(join(directory, `${namePrefix}.cpuprofile`), profile);
          await jsonFile(join(directory, `${namePrefix}.trace.json`), {traceEvents: spans.traceEvents,
            displayTimeUnit: 'ms', metadata: {meaning: 'Elapsed wall spans; async runtime waits included', truncated: spans.traceTruncated}});
          const resolver = sourceResolver(report.trees[name].sourceMapForProfile), fallback = sourceResolver(null);
          const tree = profileTree(profile, frame => frame.url?.endsWith(`/${name}.mjs`) ? resolver(frame) : fallback(frame));
          const svg = flamegraphSvg(tree, `${name} / ${workload} / ${report.backend}`);
          await writeFile(join(directory, `${namePrefix}.svg`), svg, {mode: 0o600});
          const {root: ignoredRoot, ...sampling} = tree; void ignoredRoot;
          const {traceEvents, ...wall} = spans;
          report.profiles[name][workload] = {...sampling, ...wall, profileSha256: digest(JSON.stringify(profile)),
            artifacts: {cpu: `${namePrefix}.cpuprofile`, flamegraph: `${namePrefix}.svg`, wall: `${namePrefix}.trace.json`}};
          sections.push(`<details open><summary>${escapeHtml(name)} / ${escapeHtml(workload)}</summary><p><a href="${namePrefix}.cpuprofile">Chrome CPU profile</a> · <a href="${namePrefix}.svg">Standalone flamegraph</a> · <a href="${namePrefix}.trace.json">Chrome/Perfetto wall trace</a></p>${svg}${wallSvg(traceEvents)}<pre>${escapeHtml(JSON.stringify({sampling, wall}, null, 2))}</pre></details>`);
          if (workload !== 'cold') assert(tree.samples >= policy.minimumProfileSamples, 'Insufficient CPU profile samples');
          console.log(JSON.stringify({phase: 'profile', tree: name, workload, samples: tree.samples}));
        }, workload === 'cold');
      }
      const profileCases = [...profileSources].sort().map(source => ({id: `profile-${digest(Buffer.from(source))}`, source}));
      report.profileWorkloadInputs = {rows: profileCases.length, sha256: digest(JSON.stringify(profileCases))};
      for (const name of Object.keys(trees)) await withPage(name, async page => {
          smokeOutputs[name] = await page.evaluate(cases => globalThis.gammaProbe.smoke(cases), [...smoke, ...profileCases]);
          logits[name] = await page.evaluate(() => globalThis.gammaProbe.logits());
          if (qualityRows.length) qualityOutputs[name] = await page.evaluate(rows => globalThis.gammaProbe.quality(rows), qualityRows);
          await jsonFile(join(directory, `${name}-quality.private.json`), {smoke: smokeOutputs[name], timing: trialOutputs[name], quality: qualityOutputs[name], logits: logits[name]});
          console.log(JSON.stringify({phase: 'quality', tree: name, rows: qualityRows.length}));
        });
    }, () => createServer((request, response) => {
      const pathname = new URL(request.url ?? '/', 'http://localhost').pathname, bytes = assets.get(pathname);
      if (!bytes) { response.writeHead(404).end(); return; }
      response.writeHead(200, {'Content-Type': pathname.endsWith('.mjs') ? 'text/javascript' : pathname.endsWith('.json') ? 'application/json' : pathname === '/' ? 'text/html' : 'application/octet-stream'});
      response.end(bytes);
    }));
    report.parity = {};
    for (const name of ['baseline', 'anchor'].filter(name => trees[name])) {
      report.parity[name] = {smoke: compareOutputs(smokeOutputs[name], smokeOutputs.candidate), timing: compareOutputs(trialOutputs[name], trialOutputs.candidate), logits: compareLogits(logits[name], logits.candidate)};
      if (qualityRows.length) report.parity[name].quality = compareOutputs(qualityOutputs[name], qualityOutputs.candidate);
    }
    for (const [name, tree] of Object.entries(trees)) {
      const after = await snapshot(tree), before = report.trees[name];
      for (const key of ['commit', 'treeHash', 'dirtyPaths', 'guarded', 'lockSha256']) assert.deepEqual(after[key], before[key], 'Source changed during analysis');
      for (const [input, hash] of Object.entries(before.inputs)) assert.equal(digest(await readFile(isAbsolute(input) ? input : join(tree, input))), hash, 'Executed input changed during analysis');
      delete before.sourceMapForProfile;
    }
    assert.deepEqual(await snapshot(root), Object.fromEntries(Object.entries(report.harness).filter(([key]) => !['root', 'hashes', 'sha256'].includes(key))), 'Trusted harness changed during analysis');
    for (const [path, hash] of Object.entries(report.harness.hashes)) assert.equal(digest(await readFile(join(root, path))), hash, 'Protected harness input changed');
    report.environment.host.push(await hostSample());
    const hosts = report.environment.host;
    const swapChanged = hosts.some((row, index) => index > 0 && swapAdvanced(hosts[index - 1], row));
    report.environment.underPressure = hosts.some(row => row.underPressure) || swapChanged;
    let anchorGuard = true;
    if (trees.anchor) {
      report.anchorRegressions = Object.fromEntries([...Object.keys(policy.weights), 'nearby', 'cold'].map(name => [name,
        percentile(report.trials.candidate.map((row, index) => (name === 'cold' ? row.coldMs / report.trials.anchor[index].coldMs : row[name].medianMs / report.trials.anchor[index][name].medianMs) - 1), 0.5)]));
      anchorGuard = Object.values(report.anchorRegressions).every(value => value <= policy.maximumRegression);
      report.anchorP95Regressions = Object.fromEntries(workloadNames.map(name => [name,
        percentile(report.trials.candidate.map((row, index) => row[name].p95Ms / report.trials.anchor[index][name].p95Ms - 1), 0.5)]));
      anchorGuard &&= Object.values(report.anchorP95Regressions).every(value => value <= policy.maximumP95Regression);
    }
    report.performance = trees.baseline ? timingGate(report.trials.baseline, report.trials.candidate, report.quality.complete, anchorGuard) : null;
    report.decision = interrupted ? {status: 'invalid', exitCode: 1, reason: 'analysis-interrupted'} :
      !trees.baseline ? {status: 'inconclusive', exitCode: 3, reason: 'baseline-required-for-promotion'} :
      report.environment.underPressure ? {status: 'inconclusive', exitCode: 3, reason: 'host-pressure-or-active-swapping'} :
        report.performance;
  } catch (error) {
    // Machine-readable failure replaces success; never print natural text or
    // browser exception bodies (they can contain source sentences).
    if (report.options) await writeFile(join(directory, 'error.private.log'), String(error.stack ?? error), {mode: 0o600}).catch(() => {});
    report.decision = {status: 'invalid', exitCode: 1, reason: 'analysis-validation-failed',
      failure: String(error.message).startsWith('page.evaluate:') ? 'Browser operation failed; inspect error.private.log' : String(error.message).split('\n')[0].slice(0, 180)};
  } finally {
    if (browser) await browser.close().catch(() => { report.decision = {status: 'invalid', exitCode: 1, reason: 'browser-cleanup-failed'}; });
    for (const lock of locks.reverse()) await lock.release().catch(() => { report.decision = {status: 'invalid', exitCode: 1, reason: 'lock-cleanup-failed'}; });
    for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.off(signal, onSignal);
    // Cleanup awaits can receive a signal after the timing decision is made.
    // Seal interruption status before writing any receipt.
    if (interrupted) report.decision = {status: 'invalid', exitCode: 1, reason: 'analysis-interrupted'};
    if (config) {
      report.finishedAt = new Date().toISOString();
      // Never reuse an output directory. If mkdir failed, emit a fresh failure
      // artifact instead of overwriting an earlier receipt.
      try {
        if (!report.options) {
          directory = join(root, 'artifacts/inference-analysis', id + '-invalid');
          await mkdir(directory, {recursive: true, mode: 0o700});
        }
        for (const tree of Object.values(report.trees)) delete tree.sourceMapForProfile;
        await jsonFile(join(directory, 'report.json'), report);
        await writeFile(join(directory, 'report.html'), htmlReport(report, sections), {mode: 0o600});
        console.log(JSON.stringify({report: join(directory, 'report.json'), html: join(directory, 'report.html'), ...report.decision}));
      } catch { console.log(JSON.stringify({status: 'invalid', exitCode: 1, reason: 'artifact-write-failed'})); report.decision.exitCode = 1; }
      process.exitCode = report.decision.exitCode;
    }
  }
}

await main();
