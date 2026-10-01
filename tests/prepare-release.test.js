import { describe, expect, it } from 'vite-plus/test';
import { nextVersion } from '../scripts/prepare-release.mjs';

describe('automatic release versions', () => {
  it('increments the greatest source or published version', () => {
    expect(nextVersion('0.1.0', [])).toBe('0.1.1');
    expect(nextVersion('0.1.0', ['v0.1.9', 'v0.1.2', 'v0.1.10', 'v0.1.11-beta'])).toBe('0.1.11');
    expect(nextVersion('1.0.0', ['v0.9.99'])).toBe('1.0.1');
  });
  it('rolls over browser version bounds and rejects invalid sources', () => {
    expect(nextVersion('1.2.65535', [])).toBe('1.3.0');
    expect(nextVersion('1.65535.65535', [])).toBe('2.0.0');
    expect(() => nextVersion('65535.65535.65535', [])).toThrow(/exhausted/);
    expect(() => nextVersion('1.0.0-beta', [])).toThrow(/Invalid/);
  });
});

it('creates synchronized version metadata and rejects non-version snapshot changes', async () => {
  const {execFileSync} = await import('node:child_process');
  const {mkdtemp, writeFile, mkdir, readFile, rm} = await import('node:fs/promises');
  const {tmpdir} = await import('node:os');
  const {join} = await import('node:path');
  const {fileURLToPath} = await import('node:url');
  const root = await mkdtemp(join(tmpdir(), 'gamma-snapshot-test-'));
  const git = (...args) => execFileSync('git', args, {cwd: root, stdio: 'pipe'});
  const runScript = name => execFileSync(process.execPath, [fileURLToPath(new URL(`../scripts/${name}`, import.meta.url))], {cwd: root, stdio: 'pipe'});
  try {
    git('init', '-q');
    git('config', 'user.name', 'Fictional Test');
    git('config', 'user.email', 'test@example.invalid');
    await mkdir(join(root, 'apps/extension'), {recursive: true});
    await writeFile(join(root, 'package.json'), JSON.stringify({version: '0.1.0', name: 'fixture'}));
    await writeFile(join(root, 'package-lock.json'), JSON.stringify({version: '0.1.0', packages: {'': {version: '0.1.0'}}}));
    await writeFile(join(root, 'apps/extension/manifest.json'), JSON.stringify({version: '0.1.0', name: 'fixture'}));
    git('add', '.');
    git('commit', '-qm', 'fixture');
    git('tag', 'v0.1.5');
    expect(runScript('prepare-release.mjs').toString().trim()).toBe('v0.1.6');
    for (const name of ['package.json', 'package-lock.json', 'apps/extension/manifest.json']) {
      expect(JSON.parse(await readFile(join(root, name), 'utf8')).version).toBe('0.1.6');
    }
    git('add', '.');
    git('commit', '-qm', 'release snapshot');
    expect(() => runScript('verify-release-snapshot.mjs')).not.toThrow();
    await writeFile(join(root, 'package.json'), JSON.stringify({version: '0.1.6', name: 'tampered'}));
    git('add', '.');
    git('commit', '--amend', '--no-edit', '-q');
    expect(() => runScript('verify-release-snapshot.mjs')).toThrow();
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});
