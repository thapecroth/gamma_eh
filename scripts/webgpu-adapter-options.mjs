import { readFile } from 'node:fs/promises';

// Both the pinned JAX backend and the older native ORT loader pass an options
// object. Assert the call count so an upstream layout change cannot go unnoticed.
export function patchWebGPUAdapterRequest(source) {
  const requests = source.match(/navigator\.gpu\.requestAdapter\s*\(/gu)?.length ?? 0;
  if (requests !== 1) throw new Error(`Expected one WebGPU adapter request; found ${requests}. Review the pinned runtime before building.`);
  let count = 0;
  const code = source.replace(/navigator\.gpu\.requestAdapter\((\{[^()]*powerPreference[^()]*\}|[A-Za-z_$][\w$]*)\)/gu,
    (_request, argument) => {
      count++;
      return `navigator.gpu.requestAdapter((options => {
        const platform = globalThis.navigator?.userAgentData?.platform || globalThis.navigator?.platform || globalThis.navigator?.userAgent || '';
        if (!/(?:^Win|Windows)/i.test(platform)) return options;
        const supported = {...options};
        delete supported.powerPreference;
        return supported;
      })(${argument}))`;
    });
  if (count !== 1) throw new Error(`Expected one WebGPU adapter request; found ${count}. Review the pinned runtime before building.`);
  return code;
}

const jaxModule = /[/\\]@jax-js[/\\]jax[/\\].*\.[cm]?js$/u;

function adapterGuard() {
  const modules = new Set();
  const patched = new Set();
  return {
    transform(source, id) {
      modules.add(id);
      if (!/\brequestAdapter\b/u.test(source)) return;
      const code = patchWebGPUAdapterRequest(source);
      patched.add(id);
      return code;
    },
    finish() {
      if (modules.size && patched.size !== 1) {
        throw new Error(`Expected one JAX module with a patched WebGPU adapter request; found ${patched.size}. Review the pinned runtime before building.`);
      }
    },
  };
}

export function webGPUAdapterVitePlugin() {
  let guard = adapterGuard();
  return {
    name: 'windows-webgpu-adapter-options',
    enforce: 'pre',
    buildStart() { guard = adapterGuard(); },
    transform(source, id) {
      const path = id.split('?')[0];
      if (!jaxModule.test(path)) return;
      const code = guard.transform(source, path);
      if (code !== undefined) return {code, map: null};
    },
    buildEnd(error) { if (!error) guard.finish(); },
  };
}

export function webGPUAdapterEsbuildPlugin() {
  return {
    name: 'windows-webgpu-adapter-options',
    setup(build) {
      let guard;
      build.onStart(() => { guard = adapterGuard(); });
      // esbuild's Go regexp parser does not accept JavaScript's Unicode flag.
      build.onLoad({filter: new RegExp(jaxModule.source)}, async ({path}) => {
        const source = await readFile(path, 'utf8');
        return {contents: guard.transform(source, path) ?? source, loader: 'js'};
      });
      build.onEnd(result => { if (!result.errors.length) guard.finish(); });
    },
  };
}
