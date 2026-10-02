import {it, expect} from 'vite-plus/test';
import {execFileSync} from 'node:child_process';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

it('overwrites successful browser evidence when a threshold grid is invalid', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'gamma-tuning-receipt-'));
  const output = join(directory, 'report.json');
  try {
    await writeFile(output, JSON.stringify({passed: true, rows: ['old']}));
    expect(() => execFileSync(process.execPath, ['scripts/evaluate-tuning.mjs', '--input', 'missing.jsonl',
      '--model', 'models/browser', '--output', output, '--thresholds', '0.8,,2'], {stdio: 'pipe'})).toThrow();
    const failed = JSON.parse(await readFile(output, 'utf8'));
    expect(failed.passed).toBe(false);
    expect(failed.failure).toMatch(/Invalid confidence threshold grid/);
    expect(failed.rows).toBeUndefined();
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});
