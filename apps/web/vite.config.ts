import { fileURLToPath } from 'node:url';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { join } from 'node:path';
import { defineConfig } from 'vite-plus';
import react from '@vitejs/plugin-react';
import { webGPUAdapterVitePlugin } from '../../scripts/webgpu-adapter-options.mjs';
import { webBase } from '../../scripts/web-base.mjs';

const base = webBase();

export default defineConfig({
  base,
  root: fileURLToPath(new URL('.', import.meta.url)),
  plugins: [react(), webGPUAdapterVitePlugin(), {
    name: 'local-inference-assets',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
        const match = /^models\/([a-zA-Z0-9_.-]+)$/u.exec(pathname.startsWith(base) ? pathname.slice(base.length) : '');
        if (!match) return next();
        const filename = join(fileURLToPath(new URL('../../models/browser/', import.meta.url)), match[1]);
        try {
          const info = await stat(filename);
          if (!info.isFile()) return next();
          response.setHeader('Content-Type', filename.endsWith('.wasm') ? 'application/wasm' : filename.endsWith('.mjs') ? 'text/javascript' : filename.endsWith('.json') ? 'application/json' : 'application/octet-stream');
          createReadStream(filename).pipe(response);
        } catch { next(); }
      });
    },
  }],
  resolve: { alias: { '@gamma/engine': fileURLToPath(new URL('../../packages/engine/src/index.ts', import.meta.url)) } },
  // Keep the adapter request visible to the dev transform, including ONNX's
  // transitive JAX import, rather than hiding it in a preoptimized dependency.
  optimizeDeps: {exclude: ['@jax-js/jax', '@jax-js/onnx']},
  build: { outDir: '../../dist/web', emptyOutDir: true },
  worker: { format: 'es', plugins: () => [webGPUAdapterVitePlugin()] },
  server: { host: '127.0.0.1' },
});
