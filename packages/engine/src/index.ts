import { overlaps } from './edits';
import { analyzeModel } from './model';
import { analyzeRules } from './rules';
import type { AnalysisResult, EngineOptions } from './types';

export type { AnalysisResult, EngineOptions, Suggestion, SuggestionCategory } from './types';
export { analyzeRules } from './rules';
export { applySuggestion, applySuggestions } from './edits';

export async function analyzeText(text: string, options?: EngineOptions): Promise<AnalysisResult> {
  if (typeof text !== 'string') throw new TypeError('Text must be a string');
  if (text.length > 20_000) throw new Error('Check up to 20,000 characters at a time.');
  const started = performance.now();
  const suggestions = analyzeRules(text);
  if (!text.trim()) return {text, suggestions, backend: 'rules', elapsedMs: performance.now() - started};
  try {
    const model = await analyzeModel(text, options ?? {}, suggestions);
    for (const edit of model.suggestions) {
      if (!suggestions.some(previous => overlaps(previous, edit))) suggestions.push(edit);
    }
    return {text, suggestions: suggestions.sort((a, b) => a.start - b.start), backend: model.backend,
      elapsedMs: performance.now() - started};
  } catch (error) {
    return {text, suggestions, backend: 'rules', elapsedMs: performance.now() - started,
      modelError: error instanceof Error ? error.message : 'Local model could not start'};
  }
}
