import { execFileSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

export function nextVersion(current, tags) {
  const parse = value => /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.test(value)
    && value.split('.').every(part => Number(part) <= 65535) ? value.split('.').map(Number) : null;
  let latest = parse(current);
  if (!latest) throw new Error('Invalid current release version.');
  for (const tag of tags) {
    const candidate = parse(tag.replace(/^v/u, ''));
    if (candidate && candidate.some((part, index) => part > latest[index] && candidate.slice(0, index).every((previous, i) => previous === latest[i]))) latest = candidate;
  }
  for (let index = 2; index >= 0; index--) {
    if (latest[index] < 65535) {
      latest[index]++;
      latest.fill(0, index + 1);
      return latest.join('.');
    }
  }
  throw new Error('Release version space exhausted.');
}

async function main() {
  const pkg = JSON.parse(await readFile('package.json', 'utf8'));
  const version = nextVersion(pkg.version, execFileSync('git', ['tag', '--list', 'v*'], {encoding: 'utf8'}).trim().split('\n'));
  for (const filename of ['package.json', 'package-lock.json', 'apps/extension/manifest.json']) {
    const value = JSON.parse(await readFile(filename, 'utf8'));
    value.version = version;
    if (filename === 'package-lock.json') value.packages[''].version = version;
    await writeFile(filename, JSON.stringify(value, null, 2) + '\n');
  }
  console.log(`v${version}`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
