import { JaxSession } from './jax-runtime';
import { preserveCase } from './edits';
import { validArticleEdit, validVerbEdit } from './guards';
import { protectedSpans } from './rules';
import { WordPieceTokenizer } from './tokenizer';
import type { EngineOptions, Suggestion } from './types';

interface ModelManifest { confidenceThreshold: number; maxSequenceLength: number }
interface LoadedModel {
  session: JaxSession;
  tokenizer: WordPieceTokenizer;
  labels: string[];
  manifest: ModelManifest;
  backend: 'webgpu' | 'wasm';
}

const sessions = new Map<string, Promise<LoadedModel>>();
const assetUrl = (base: string, name: string) => `${base.replace(/\/?$/u, '/')}${name}`;

async function fetchAsset(base: string, filename: string): Promise<Response> {
  const response = await fetch(assetUrl(base, filename));
  if (!response.ok) throw new Error(`Could not load ${filename} (${response.status})`);
  return response;
}

async function load(options: EngineOptions): Promise<LoadedModel> {
  const base = options.modelBaseUrl ?? '/models/';
  const [manifestResponse, labelsResponse, vocabResponse] = await Promise.all([
    fetchAsset(base, 'manifest.json'), fetchAsset(base, 'labels.json'), fetchAsset(base, 'vocab.txt'),
  ]);
  const manifest: unknown = await manifestResponse.json();
  const labels: unknown = await labelsResponse.json();
  if (!manifest || typeof manifest !== 'object' ||
      !('confidenceThreshold' in manifest) || typeof manifest.confidenceThreshold !== 'number' ||
      !Number.isFinite(manifest.confidenceThreshold) || manifest.confidenceThreshold < 0 || manifest.confidenceThreshold > 1 ||
      !('maxSequenceLength' in manifest) || typeof manifest.maxSequenceLength !== 'number' ||
      !Number.isInteger(manifest.maxSequenceLength) || manifest.maxSequenceLength < 3 || manifest.maxSequenceLength > 512) {
    throw new Error('Invalid model manifest');
  }
  if (!Array.isArray(labels) || labels[0] !== 'KEEP' || !labels.every(label => typeof label === 'string')) {
    throw new Error('Invalid model edit vocabulary');
  }
  const tokenizer = new WordPieceTokenizer(await vocabResponse.text());
  const model = new Uint8Array(await (await fetchAsset(base, 'model.onnx')).arrayBuffer());
  const session = await JaxSession.create(model, options.preferWebGPU !== false);
  const backend = session.backend;
  return {session, tokenizer, labels, manifest: manifest as ModelManifest, backend};
}

function getModel(options: EngineOptions): Promise<LoadedModel> {
  const key = JSON.stringify([options.modelBaseUrl, options.preferWebGPU]);
  let pending = sessions.get(key);
  if (!pending) {
    pending = load(options).catch(error => { sessions.delete(key); throw error; });
    sessions.set(key, pending);
  }
  return pending;
}

export async function analyzeModel(text: string, options: EngineOptions, ruleSuggestions: Suggestion[] = []): Promise<{suggestions: Suggestion[]; backend: 'webgpu' | 'wasm'}> {
  const {session, tokenizer, labels, manifest, backend} = await getModel(options);
  const threshold = options.confidenceThreshold ?? manifest.confidenceThreshold;
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Confidence threshold must be between zero and one');
  const suggestions: Suggestion[] = [];
  const protectedRanges = protectedSpans(text);
  for (const chunk of tokenizer.chunks(text, manifest.maxSequenceLength)) {
    const first = chunk.positions[0]?.word.start ?? 0;
    const last = chunk.positions.at(-1)?.word.end ?? 0;
    // This model was not trained on code/URLs or on valid text after a rule
    // deletes a duplicate. Do not infer conflicting edits from that context.
    if (protectedRanges.some(range => first < range.end && range.start < last) ||
        ruleSuggestions.some(edit => !edit.replacement && first < edit.end && edit.start < last)) continue;
    const logits = await session.run(chunk.ids);
    if (logits.dims.length !== 3 || logits.dims[0] !== 1 || logits.dims[1] !== chunk.ids.length ||
        logits.dims[2] !== labels.length || logits.data.length !== chunk.ids.length * labels.length) {
      throw new Error('Unexpected model output shape');
    }
    for (const [index, {word, position}] of chunk.positions.entries()) {
      if (!/^[A-Za-z]+(?:['’][A-Za-z]+)*$/u.test(word.text)) continue;
      if (protectedRanges.some(range => word.start < range.end && range.start < word.end)) continue;
      const row = logits.data.subarray(position * labels.length, (position + 1) * labels.length);
      let best = 0;
      for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
      if (!best) continue;
      const peak = row[best];
      const confidence = 1 / row.reduce((sum, value) => sum + Math.exp(value - peak), 0);
      if (confidence < threshold) continue;
      const tag = labels[best];
      let {start, end} = word;
      let replacement: string;
      if (tag === 'DELETE') {
        // Training deletion labels come only from adjacent-token duplication.
        // An out-of-domain DELETE must not erase an arbitrary valid word.
        const lower = word.text.toLowerCase();
        if (['had', 'that'].includes(lower) ||
            (chunk.positions[index - 1]?.word.text.toLowerCase() !== lower &&
             chunk.positions[index + 1]?.word.text.toLowerCase() !== lower)) continue;
        // Remove adjacent horizontal whitespace, never a paragraph boundary.
        if (/[ \t]/u.test(text[end] ?? '')) { while (/[ \t]/u.test(text[end] ?? '')) end++; }
        else { while (start > 0 && /[ \t]/u.test(text[start - 1])) start--; }
        replacement = '';
      } else if (tag.startsWith('REPLACE:')) {
        const proposed = tag.slice(8);
        if (!validVerbEdit(text, word, proposed)) continue;
        if (['a', 'an'].includes(proposed) && !validArticleEdit(proposed, chunk.positions[index + 1]?.word.text ?? '')) continue;
        replacement = preserveCase(word.text, tag.slice(8));
        if (replacement === word.text) continue;
      } else if (tag.startsWith('APPEND:')) {
        const proposed = tag.slice(7);
        if (['a', 'an'].includes(proposed) && !validArticleEdit(proposed, chunk.positions[index + 1]?.word.text ?? '')) continue;
        start = end;
        replacement = ' ' + tag.slice(7);
      } else continue;
      suggestions.push({id: `model-${start}-${end}-${tag}`, start, end,
        original: text.slice(start, end), replacement,
        message: tag === 'DELETE' ? 'The local model suggests removing this word.' :
          tag.startsWith('APPEND:') ? 'The local model suggests a missing word here.' :
            'The local model suggests this word change.',
        category: 'grammar', confidence, source: 'model'});
    }
  }
  return {suggestions, backend};
}
