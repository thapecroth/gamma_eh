import type { AnalysisResult } from '@gamma/engine';

export interface Settings { enabled: boolean; useAI: boolean; disabledSites: string[] }
export const DEFAULT_SETTINGS: Settings = { enabled: true, useAI: true, disabledSites: [] };
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
    useAI: typeof value.useAI === 'boolean' ? value.useAI : DEFAULT_SETTINGS.useAI,
    disabledSites: Array.isArray(value.disabledSites) ? value.disabledSites.filter((site): site is string => typeof site === 'string') : [] };
}

export function sitePattern(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:'].includes(parsed.protocol)) return null;
    return `${parsed.protocol}//${parsed.hostname}/*`;
  } catch { return null; }
}

export function isSiteEnabled(url: string, settings: Settings): boolean {
  const pattern = sitePattern(url);
  return settings.enabled && pattern !== null && !settings.disabledSites.includes(pattern);
}
