export type SuggestionCategory = 'grammar' | 'spelling' | 'punctuation' | 'style';

export interface Suggestion {
  id: string;
  start: number;
  end: number;
  original: string;
  replacement: string;
  message: string;
  category: SuggestionCategory;
  confidence: number;
  source: 'rule' | 'model' | 'dictionary';
  /** Source snapshot for insertions and composed edits, whose empty spans cannot detect staleness. */
  checkedText?: string;
}

export interface AnalysisResult {
  text: string;
  suggestions: Suggestion[];
  backend: 'rules' | 'wasm' | 'webgpu';
  elapsedMs: number;
  modelError?: string;
  modelRuns?: number;
}

export interface EngineOptions {
  modelBaseUrl?: string;
  wasmBaseUrl?: string;
  preferWebGPU?: boolean;
  confidenceThreshold?: number;
  mode?: 'combined' | 'model' | 'rules';
  maxPasses?: number;
}
