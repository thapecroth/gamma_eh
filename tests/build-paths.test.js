import { describe, expect, it } from 'vitest';
import { buildPaths } from '../scripts/paths.mjs';

describe('isolated model experiments', () => {
  it('keeps default shipping locations unchanged', () => {
    expect(buildPaths('/project', {})).toEqual({modelDirectory: '/project/models/browser',
      webOutput: '/project/dist/web', extensionOutput: '/project/dist/extension', artifactDir: '/project/artifacts'});
  });
  it('places alternate models, builds, and evidence in isolated locations', () => {
    expect(buildPaths('/project', {GAMMA_BUILD_PROFILE: 'glm-flash-pilot', GAMMA_MODEL_DIR: 'artifacts/pilot/model'}))
      .toEqual({modelDirectory: '/project/artifacts/pilot/model', webOutput: '/project/dist/glm-flash-pilot/web',
        extensionOutput: '/project/dist/glm-flash-pilot/extension', artifactDir: '/project/artifacts/glm-flash-pilot'});
  });
  it('rejects path traversal and replacing the shipping model implicitly', () => {
    expect(() => buildPaths('/project', {GAMMA_MODEL_DIR: 'artifacts/pilot/model'})).toThrow(/isolated/);
    for (const profile of ['../', '/tmp', 'MODEL', 'x/y']) {
      expect(() => buildPaths('/project', {GAMMA_BUILD_PROFILE: profile})).toThrow(/slug/);
    }
  });
});
