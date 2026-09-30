import { describe, expect, it } from 'vite-plus/test';
import { getSettings, isCheckMessage, MAX_FIELD_LENGTH, scriptId, sitePattern } from '../apps/extension/src/protocol';
import { matchesSuggestion, sensitiveAutocomplete } from '../apps/extension/src/editable';
import type { Suggestion } from '@gamma/engine';

describe('extension boundary', () => {
  it('allows only bounded, explicitly addressed analysis messages', () => {
    const valid = { target: 'background', action: 'analyze', requestId: 'request-1', text: 'A short draft.', useAI: true };
    expect(isCheckMessage(valid)).toBe(true);
    expect(isCheckMessage({ ...valid, target: 'page' })).toBe(false);
    expect(isCheckMessage({ ...valid, text: 'x'.repeat(MAX_FIELD_LENGTH + 1) })).toBe(false);
    expect(isCheckMessage({ ...valid, useAI: 'yes' })).toBe(false);
    expect(isCheckMessage(null)).toBe(false);
  });

  it('restricts host grants to the current website rather than every origin', () => {
    expect(sitePattern('https://example.org/editor?secret=private')).toBe('https://example.org/*');
    expect(sitePattern('https://sub.example.org:8443/editor')).toBe('https://sub.example.org/*');
    expect(sitePattern('chrome://extensions')).toBeNull();
    expect(sitePattern('file:///private/draft.txt')).toBeNull();
    expect(sitePattern('not a URL')).toBeNull();
    expect(scriptId('https://example.org/*')).toBe(scriptId('https://example.org/*'));
    expect(scriptId('https://example.org/*')).not.toBe(scriptId('http://example.org/*'));
  });

  it('defaults malformed stored settings without enabling typed-text persistence', () => {
    expect(getSettings({ enabled: false, useAI: false })).toEqual({ enabled: false, useAI: false });
    expect(getSettings({ enabled: 'no', useAI: undefined })).toEqual({ enabled: true, useAI: false });
  });

  it('excludes payment, password, username, and one-time-code autocomplete fields', () => {
    for (const value of ['billing cc-number', 'shipping cc-name', 'CC-CSC', 'current-password', 'new-password', 'one-time-code', 'username webauthn']) expect(sensitiveAutocomplete(value)).toBe(true);
    expect(sensitiveAutocomplete('off')).toBe(false);
    expect(sensitiveAutocomplete('on')).toBe(false);
  });

  it('requires exact UTF-16 source offsets before editing', () => {
    const text = '😀 She go home.';
    const suggestion: Suggestion = { id: '1', start: 7, end: 9, original: 'go', replacement: 'goes', category: 'grammar', message: 'Use goes.', confidence: 1, source: 'rule' };
    expect(matchesSuggestion(text, suggestion)).toBe(true);
    expect(matchesSuggestion(text, { ...suggestion, start: 6, end: 8 })).toBe(false);
    expect(matchesSuggestion('😀 She went home.', suggestion)).toBe(false);
    expect(matchesSuggestion(text, { ...suggestion, start: -1 })).toBe(false);
    expect(matchesSuggestion(text, { ...suggestion, end: 100 })).toBe(false);
    expect(matchesSuggestion(text, { ...suggestion, start: 7.5 })).toBe(false);
  });
});
