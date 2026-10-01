import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';

const root = fileURLToPath(new URL('../', import.meta.url));
const temporary = await mkdtemp(join(tmpdir(), 'gamma-spelling-bench-'));
try {
  const output = join(temporary, 'engine.mjs');
  await build({entryPoints: [join(root, 'packages/engine/src/rules.ts')], outfile: output,
    bundle: true, platform: 'node', format: 'esm', logLevel: 'silent'});
  const {analyzeRules} = await import(pathToFileURL(output).href);
  const generated = await readFile(join(root, 'packages/engine/src/dictionary.generated.ts'), 'utf8');
  const words = generated.match(/dictionaryWords = '([^']+)'/u)[1].split(' ');
  const typos = [...new Set(words.slice(0, 4000).filter(word => word.length >= 5 && word.length <= 20)
    .map(word => word.slice(0, 2) + word.slice(3)))].slice(0, 1000);
  const measure = text => { const started = performance.now(); analyzeRules(text); return performance.now() - started; };
  globalThis.gc?.();
  const heapBefore = process.memoryUsage().heapUsed;
  const coldKnownMs = measure('A clean sentence.');
  const coldIndexMs = measure('helo');
  globalThis.gc?.();
  const heapAfter = process.memoryUsage().heapUsed;
  const summary = values => {
    const sorted = values.toSorted((a, b) => a - b);
    return {samples: values.length, p50Ms: sorted[Math.floor(sorted.length * 0.5)],
      p95Ms: sorted[Math.floor(sorted.length * 0.95)], maxMs: sorted.at(-1)};
  };
  const firstLookups = summary(typos.map(measure));
  const cachedLookups = summary(typos.map(measure));
  const draft = 'Speling matters in a sentnce. '.repeat(700).slice(0, 20_000);
  measure(draft);
  const draftLookups = summary(Array.from({length: 20}, () => measure(draft)));
  const evidence = {node: process.version, kind: 'Node/V8 microbenchmark; synthetic typos, not spelling accuracy or browser/device performance',
    coldKnownMs, coldIndexMs, retainedHeapDeltaMiB: (heapAfter - heapBefore) / 1024 / 1024,
    forcedGC: typeof globalThis.gc === 'function', firstLookups, cachedLookups,
    draft: {characters: draft.length, ...draftLookups}};
  const artifacts = join(root, 'artifacts');
  await mkdir(artifacts, {recursive: true});
  await writeFile(join(artifacts, 'spelling-benchmark.json'), JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify(evidence, null, 2));
} finally { await rm(temporary, {recursive: true, force: true}); }
