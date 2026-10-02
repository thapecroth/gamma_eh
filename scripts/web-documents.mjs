import {readFile} from 'node:fs/promises';
import {Marked} from 'marked';

export const webContentPolicy = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; form-action 'none'";
const documents = {'privacy.html': ['privacy.md', 'Privacy policy'], 'help.html': ['support.md', 'Help and limitations']};
const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
// Only trusted, committed documentation enters this build-time parser. It never
// processes a draft or runs in the shipped browser application.
const parser = new Marked({
  renderer: {html: token => escape(token.text)},
  walkTokens(token) {
    if (token.type !== 'link') return;
    if (token.href === 'privacy.md') token.href = 'privacy.html';
    else if (token.href === 'support.md') token.href = 'help.html';
    else if (token.href === 'https://thapecroth.github.io/gamma_eh/') token.href = './';
    else if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:/u.test(token.href)) {
      token.href = new URL(token.href, 'https://github.com/thapecroth/gamma_eh/blob/main/docs/').href;
    } else if (!/^(?:https?:|mailto:)/u.test(token.href)) throw new Error(`Unsupported documentation link: ${token.href}`);
  },
});

async function renderDocument(filename) {
  const [source, title] = documents[filename];
  const markdown = await readFile(new URL(`../docs/${source}`, import.meta.url), 'utf8');
  return `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="${escape(webContentPolicy)}"><title>${title} · Gamma EH</title><link rel="stylesheet" href="documents.css"></head><body><header><a href="./">Gamma EH editor</a><nav aria-label="Help and privacy"><a href="privacy.html">Privacy policy</a><a href="help.html">Help and limitations</a></nav></header><main>${parser.parse(markdown)}</main></body></html>\n`;
}

export function webDocumentsPlugin(base) {
  return {
    name: 'bundled-product-documents',
    async generateBundle() {
      for (const filename of Object.keys(documents)) this.emitFile({type: 'asset', fileName: filename, source: await renderDocument(filename)});
      this.emitFile({type: 'asset', fileName: 'documents.css', source: await readFile(new URL('../apps/web/documents.css', import.meta.url), 'utf8')});
    },
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
        const filename = pathname.startsWith(base) ? pathname.slice(base.length) : '';
        if (!Object.hasOwn(documents, filename) && filename !== 'documents.css') return next();
        try {
          response.setHeader('Content-Type', filename.endsWith('.css') ? 'text/css' : 'text/html; charset=utf-8');
          response.end(filename.endsWith('.css') ? await readFile(new URL('../apps/web/documents.css', import.meta.url)) : await renderDocument(filename));
        } catch (error) { next(error); }
      });
    },
  };
}
