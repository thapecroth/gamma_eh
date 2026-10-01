import { describe, expect, it } from 'vite-plus/test';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { applyGold, summarizeRows, validateCorpus } from '../scripts/challenge-metrics.mjs';

const row = (overrides = {}) => ({id: 'case-1', source: 'They is ready.', target: 'They are ready.',
  family: 'agreement', clean: false, requiredTags: ['REPLACE:are'],
  goldEdits: [{start: 5, end: 7, original: 'is', replacement: 'are'}], ...overrides});
const corpus = cases => ({schema: 1, license: 'CC0-1.0', cases});

describe('synthetic challenge evidence', () => {
  it('verifies exact targets with UTF-16 spans and multiple edits', () => {
    const source = '😀. They is ready.\r\nI recieved it.';
    const edits = [{start: source.indexOf('is'), end: source.indexOf('is') + 2, original: 'is', replacement: 'are'},
      {start: source.indexOf('recieved'), end: source.indexOf('recieved') + 8, original: 'recieved', replacement: 'received'}];
    expect(applyGold(source, edits)).toBe('😀. They are ready.\r\nI received it.');
  });
  it('rejects stale, overlapping and invalid gold instead of scoring it', () => {
    expect(() => validateCorpus(corpus([row({target: 'They were ready.'})]))).toThrow(/reconstruct/);
    expect(() => validateCorpus(corpus([row({clean: true})]))).toThrow(/clean label/);
    expect(() => validateCorpus(corpus([row(), row()]))).toThrow(/Duplicate/);
    expect(() => applyGold('They is ready.', [{start: 5, end: 7, original: 'are', replacement: 'are'}])).toThrow(/Stale/);
    expect(() => applyGold('abc', [{start: 0, end: 2, original: 'ab', replacement: 'x'},
      {start: 1, end: 3, original: 'bc', replacement: 'y'}])).toThrow(/Overlapping/);
    expect(() => applyGold('abc', [{start: -1, end: 2, original: '', replacement: ''}])).toThrow(/Invalid/);
    expect(() => validateCorpus(corpus([]))).toThrow(/Empty/);
  });
  it('accepts explicit insertion gold without changing whitespace', () => {
    expect(applyGold('We bought book.', [{start: 10, end: 10, original: '', replacement: 'a '}])).toBe('We bought a book.');
  });
  it('separates missed corrections from damage to clean sentences', () => {
    const results = [
      row({actual: 'They are ready.', suggestions: [{source: 'model'}]}),
      row({id: 'missed', actual: 'They is ready.', suggestions: []}),
      row({id: 'partial', actual: 'They is happy.', suggestions: [{source: 'dictionary'}]}),
      row({id: 'clean', clean: true, source: 'They are ready.', target: 'They are ready.', actual: 'They are ready.', suggestions: []}),
      row({id: 'damaged', clean: true, source: 'They are ready.', target: 'They are ready.', actual: 'They is ready.', suggestions: [{source: 'rule'}]}),
    ];
    const score = summarizeRows(results);
    expect(score.exact_matches).toBe(2);
    expect(score.clean_false_positive_rate).toBe(0.5);
    expect(score.complete_correction_rate).toBe(1 / 3);
    expect(score.error_abstention_rate).toBe(1 / 3);
    expect(score.incorrect_or_partial_outputs).toBe(1);
    expect(score.suggestion_sources).toEqual({model: 1, dictionary: 1, rule: 1});
  });
  it('uses null for absent populations instead of reporting perfect accuracy', () => {
    const score = summarizeRows([row({clean: true, source: 'Fine.', target: 'Fine.', actual: 'Fine.', suggestions: []})]);
    expect(score.complete_correction_rate).toBeNull();
    expect(score.error_abstention_rate).toBeNull();
    expect(score.by_family.agreement.sentences).toBe(1);
  });
  it('replaces a previous successful receipt when corpus setup fails', async () => {
    const root = fileURLToPath(new URL('../', import.meta.url));
    const temporary = await mkdtemp(join(tmpdir(), 'gamma-invalid-challenge-'));
    const profile = `synthetic-invalid-${process.pid}`;
    const artifactDir = join(root, 'artifacts', profile);
    const report = join(artifactDir, 'synthetic-challenge.json');
    try {
      await mkdir(artifactDir, {recursive: true});
      await writeFile(report, JSON.stringify({infrastructure_passed: true}));
      const input = join(temporary, 'invalid.json');
      await writeFile(input, JSON.stringify(corpus([])));
      expect(() => execFileSync(process.execPath, ['scripts/evaluate-synthetic.mjs'], {cwd: root,
        env: {...process.env, GAMMA_BUILD_PROFILE: profile, GAMMA_CHALLENGE_FILE: input}, stdio: 'pipe'})).toThrow();
      const failed = JSON.parse(await readFile(report, 'utf8'));
      expect(failed.infrastructure_passed).toBe(false);
      expect(failed.failure).toMatch(/Empty challenge/);
      expect(failed.scores).toBeUndefined();
    } finally {
      await rm(temporary, {recursive: true, force: true});
      await rm(artifactDir, {recursive: true, force: true});
    }
  });
});
