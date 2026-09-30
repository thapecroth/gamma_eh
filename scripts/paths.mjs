import { join, resolve } from 'node:path';

export function buildPaths(root, environment = process.env) {
  const profile = environment.GAMMA_BUILD_PROFILE ?? '';
  if (profile && !/^[a-z][a-z0-9-]{0,63}$/u.test(profile)) throw new Error('GAMMA_BUILD_PROFILE must be a lowercase slug.');
  const modelDirectory = resolve(root, environment.GAMMA_MODEL_DIR ?? 'models/browser');
  if (modelDirectory !== join(root, 'models/browser') && !profile) throw new Error('An alternate model requires an isolated GAMMA_BUILD_PROFILE.');
  const output = profile ? join(root, 'dist', profile) : join(root, 'dist');
  return {modelDirectory, webOutput: join(output, 'web'), extensionOutput: join(output, 'extension'),
    artifactDir: profile ? join(root, 'artifacts', profile) : join(root, 'artifacts')};
}
