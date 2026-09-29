import type { AnalysisResult } from '@gamma/engine';

export interface Settings { enabled: boolean; useAI: boolean }
export const DEFAULT_SETTINGS: Settings = { enabled: true, useAI: false };
export const MAX_FIELD_LENGTH = 6_000;

export interface CheckMessage {
  target: 'background' | 'offscreen';
  action: 'analyze';
  requestId: string;
  text: string;
  useAI: boolean;
}
export interface CheckResponse { requestId: string; result?: AnalysisResult; error?: string; siteDisabled?: boolean }

export function isCheckMessage(value: unknown): value is CheckMessage {
  if (!value || typeof value !== 'object') return false;
  const message = value as Partial<CheckMessage>;
  return (message.target === 'background' || message.target === 'offscreen') && message.action === 'analyze'
    && typeof message.requestId === 'string' && message.requestId.length < 160
    && typeof message.text === 'string' && message.text.length <= MAX_FIELD_LENGTH
    && typeof message.useAI === 'boolean';
}

export function getSettings(value: Record<string, unknown>): Settings {
  return { enabled: typeof value.enabled === 'boolean' ? value.enabled : DEFAULT_SETTINGS.enabled,
    useAI: typeof value.useAI === 'boolean' ? value.useAI : DEFAULT_SETTINGS.useAI };
}

export function sitePattern(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:'].includes(parsed.protocol)) return null;
    return `${parsed.protocol}//${parsed.hostname}/*`;
  } catch { return null; }
}

export function scriptId(pattern: string): string {
  let hash = 2166136261;
  for (const char of pattern) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return `gamma-site-${(hash >>> 0).toString(16)}`;
}
