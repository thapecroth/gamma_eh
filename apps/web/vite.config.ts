import { fileURLToPath } from 'node:url';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { join } from 'node:path';
import { defineConfig } from 'vite-plus';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  plugins: [react(), {
    name: 'local-inference-assets',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
        const match = /^\/(models|runtime)\/([a-zA-Z0-9_.-]+)$/u.exec(pathname);
        if (!match) return next();
        const directory = match[1] === 'models' ? '../../models/browser/' : '../../node_modules/onnxruntime-web/dist/';
        const filename = join(fileURLToPath(new URL(directory, import.meta.url)), match[2]);
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
  build: { outDir: '../../dist/web', emptyOutDir: true },
  worker: { format: 'es' },
  server: { host: '127.0.0.1' },
});
