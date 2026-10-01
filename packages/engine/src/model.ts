import { JaxSession } from './jax-runtime';
import { applySuggestions, overlaps } from './edits';
import { decodeProposal, tagCategory } from './edit-tags';
import { EditHistory } from './edit-history';
import { analyzeRules, protectedSpans } from './rules';
import { WordPieceTokenizer } from './tokenizer';
import type { EngineOptions, Suggestion } from './types';

interface ModelManifest {
  confidenceThreshold: number;
  maxSequenceLength: number;
  editSchema?: number;
  maxPasses?: number;
  disableModelEdits?: boolean;
  confidenceThresholds?: Record<string, number>;
}
interface LoadedModel {
  session?: JaxSession;
  tokenizer: WordPieceTokenizer;
  labels: string[];
  manifest: ModelManifest;
  backend: 'webgpu' | 'wasm' | 'rules';
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
  const policy = manifest as ModelManifest;
  if (!Number.isFinite(policy.confidenceThreshold) || policy.confidenceThreshold < 0 || policy.confidenceThreshold > 1 ||
      !Number.isInteger(policy.maxSequenceLength) || policy.maxSequenceLength < 3 || policy.maxSequenceLength > 512 ||
      (policy.editSchema !== undefined && ![1, 2].includes(policy.editSchema)) ||
      (policy.maxPasses !== undefined && (!Number.isInteger(policy.maxPasses) || policy.maxPasses < 1 || policy.maxPasses > 3)) ||
      (policy.disableModelEdits !== undefined && typeof policy.disableModelEdits !== 'boolean') ||
      (policy.confidenceThresholds !== undefined && (!policy.confidenceThresholds || typeof policy.confidenceThresholds !== 'object' ||
        Array.isArray(policy.confidenceThresholds) || Object.values(policy.confidenceThresholds).some(value => !Number.isFinite(value) || value < policy.confidenceThreshold || value > 1)))) {
    throw new Error('Invalid model correction policy');
  }
  const tokenizer = new WordPieceTokenizer(await vocabResponse.text());
  if (policy.disableModelEdits) return {tokenizer, labels, manifest: policy, backend: 'rules'};
  const bytes = new Uint8Array(await (await fetchAsset(base, 'model.onnx')).arrayBuffer());
  const session = await JaxSession.create(bytes, options.preferWebGPU !== false);
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

async function analyzePass(text: string, options: EngineOptions, model: LoadedModel, ruleSuggestions: Suggestion[]): Promise<{suggestions: Suggestion[]; modelRuns: number}> {
  const {session, tokenizer, labels, manifest} = model;
  const threshold = options.confidenceThreshold ?? manifest.confidenceThreshold;
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Confidence threshold must be between zero and one');
  const suggestions: Suggestion[] = [];
  let modelRuns = 0;
  if (manifest.disableModelEdits) return {suggestions, modelRuns};
  if (!session) throw new Error('Model session unavailable');
  const protectedRanges = protectedSpans(text);
  for (const chunk of tokenizer.chunks(text, manifest.maxSequenceLength)) {
    const first = chunk.positions[0]?.word.start ?? 0;
    const last = chunk.positions.at(-1)?.word.end ?? 0;
    // This model was not trained on code/URLs or on valid text after a rule
    // deletes a duplicate. Do not infer conflicting edits from that context.
    if (protectedRanges.some(range => first < range.end && range.start < last) ||
        ruleSuggestions.some(edit => !edit.replacement && first < edit.end && edit.start < last)) continue;
    const logits = await session.run(chunk.ids);
    modelRuns++;
    if (logits.dims.length !== 3 || logits.dims[0] !== 1 || logits.dims[1] !== chunk.ids.length ||
        logits.dims[2] !== labels.length || logits.data.length !== chunk.ids.length * labels.length) {
      throw new Error('Unexpected model output shape');
    }
    if (!logits.data.every(Number.isFinite)) throw new Error('Nonfinite model output');
    const words = chunk.positions.map(({word}) => word);
    for (const [index, {word, position}] of chunk.positions.entries()) {
      if (protectedRanges.some(range => word.start < range.end && range.start < word.end)) continue;
      const row = logits.data.subarray(position * labels.length, (position + 1) * labels.length);
      let best = 0;
      for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
      if (!best) continue;
      const peak = row[best];
      const confidence = 1 / row.reduce((sum, value) => sum + Math.exp(value - peak), 0);
      const tag = labels[best];
      const requiredConfidence = Math.max(threshold, manifest.confidenceThresholds?.[tagCategory(tag)] ?? threshold);
      if (!Number.isFinite(confidence) || confidence < requiredConfidence) continue;
      const edit = decodeProposal(text, words, index, tag, confidence, manifest.editSchema ?? 1);
      if (!edit) continue;
      const {start, end, replacement} = edit;
      if (suggestions.some(previous => overlaps(previous, edit))) continue;
      suggestions.push({id: `model-${start}-${end}-${tag}`, start, end,
        original: text.slice(start, end), replacement, ...(start === end ? {checkedText: text} : {}),
        message: tag === 'DELETE' ? 'The local model suggests removing this word.' :
          tag.startsWith('APPEND:') ? 'The local model suggests a missing word here.' :
            'The local model suggests this word change.',
        category: tagCategory(tag) === 'punctuation' ? 'punctuation' : 'grammar', confidence, source: 'model'});
    }
  }
  return {suggestions, modelRuns};
}

export async function analyzeModel(text: string, options: EngineOptions, ruleSuggestions: Suggestion[] = []): Promise<{suggestions: Suggestion[]; backend: 'webgpu' | 'wasm' | 'rules'; modelRuns: number}> {
  const model = await getModel(options);
  const passes = options.maxPasses ?? model.manifest.maxPasses ?? 1;
  if (!Number.isInteger(passes) || passes < 1 || passes > 3) throw new Error('Use between one and three correction passes');
  if (passes === 1) return {...await analyzePass(text, options, model, ruleSuggestions), backend: model.backend};
  const history = new EditHistory(text);
  const seen = new Set([text]);
  let modelRuns = 0;
  for (let pass = 0; pass < passes; pass++) {
    const rules = options.mode === 'model' ? [] : pass === 0 ? ruleSuggestions : analyzeRules(history.text);
    const result = await analyzePass(history.text, options, model, rules);
    modelRuns += result.modelRuns;
    const edits = result.suggestions.filter(edit => !rules.some(rule => overlaps(rule, edit)));
    if (!edits.length) break;
    const next = applySuggestions(history.text, edits);
    if (seen.has(next)) break;
    seen.add(next);
    history.apply(edits);
  }
  return {suggestions: history.suggestions(), backend: model.backend, modelRuns};
}
