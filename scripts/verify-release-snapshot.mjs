import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

const git = args => execFileSync('git', args, {encoding: 'utf8'});
const expected = ['apps/extension/manifest.json', 'package-lock.json', 'package.json'];
assert.deepEqual(git(['diff', '--name-only', 'HEAD^', 'HEAD']).trim().split('\n').sort(), expected);
for (const filename of expected) {
  const before = JSON.parse(git(['show', `HEAD^:${filename}`]));
  const after = JSON.parse(git(['show', `HEAD:${filename}`]));
  assert.equal(typeof after.version, 'string');
  before.version = after.version;
  if (filename === 'package-lock.json') before.packages[''].version = after.packages[''].version;
  assert.deepEqual(after, before, `Release snapshot changed more than version metadata: ${filename}`);
}
