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
        const platform = globalThis.navigator?.userAgentData?.platform ?? globalThis.navigator?.platform ?? globalThis.navigator?.userAgent ?? '';
        if (!/(?:^Win|Windows)/i.test(platform)) return options;
        const supported = {...options};
        delete supported.powerPreference;
        return supported;
      })(${argument}))`;
    });
  if (count !== 1) throw new Error(`Expected one WebGPU adapter request; found ${count}. Review the pinned runtime before building.`);
  return code;
}

const jaxBackend = /[/\\]@jax-js[/\\]jax[/\\]dist[/\\]backend-[^/\\]+\.js$/u;

export function webGPUAdapterVitePlugin() {
  return {
    name: 'windows-webgpu-adapter-options',
    enforce: 'pre',
    transform(source, id) {
      if (!jaxBackend.test(id.split('?')[0])) return;
      return {code: patchWebGPUAdapterRequest(source), map: null};
    },
  };
}

export function webGPUAdapterEsbuildPlugin() {
  return {
    name: 'windows-webgpu-adapter-options',
    setup(build) {
      // esbuild's Go regexp parser does not accept JavaScript's Unicode flag.
      build.onLoad({filter: new RegExp(jaxBackend.source)}, async ({path}) => ({
        contents: patchWebGPUAdapterRequest(await readFile(path, 'utf8')),
        loader: 'js',
      }));
    },
  };
}
