import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkEnvironment } from './browser-environment.mjs';
import { buildPaths } from './paths.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const {artifactDir} = buildPaths(root);
const report = await checkEnvironment();
await mkdir(artifactDir, {recursive: true});
await writeFile(join(artifactDir, 'environment.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
if (!report.passed) process.exitCode = 1;
