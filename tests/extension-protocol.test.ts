import { describe, expect, it } from 'vite-plus/test';
import { getSettings, isCheckMessage, isSiteEnabled, MAX_FIELD_LENGTH, sitePattern } from '../apps/extension/src/protocol';
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

  it('uses only HTTP and HTTPS websites for site settings', () => {
    expect(sitePattern('https://example.org/editor?secret=private')).toBe('https://example.org/*');
    expect(sitePattern('https://sub.example.org:8443/editor')).toBe('https://sub.example.org/*');
    expect(sitePattern('chrome://extensions')).toBeNull();
    expect(sitePattern('file:///private/draft.txt')).toBeNull();
    expect(sitePattern('not a URL')).toBeNull();
  });

  it('defaults malformed stored settings without enabling typed-text persistence', () => {
    expect(getSettings({ enabled: false, useAI: false })).toEqual({ enabled: false, useAI: false, disabledSites: [] });
    expect(getSettings({ enabled: 'no', useAI: undefined, disabledSites: 'all' })).toEqual({ enabled: true, useAI: false, disabledSites: [] });
    expect(getSettings({ disabledSites: [null, 42, 'https://example.org/*'] }).disabledSites).toEqual(['https://example.org/*']);
  });

  it('enables new websites by default while respecting global and site pauses', () => {
    const defaults = getSettings({});
    expect(isSiteEnabled('https://example.org/editor', defaults)).toBe(true);
    expect(isSiteEnabled('http://another.example/draft', defaults)).toBe(true);
    expect(isSiteEnabled('chrome://extensions', defaults)).toBe(false);
    expect(isSiteEnabled('file:///private/draft.txt', defaults)).toBe(false);
    const paused = getSettings({ disabledSites: ['https://example.org/*'] });
    expect(isSiteEnabled('https://example.org:8443/another-page', paused)).toBe(false);
    expect(isSiteEnabled('https://sub.example.org/editor', paused)).toBe(true);
    expect(isSiteEnabled('https://another.example/editor', paused)).toBe(true);
    expect(isSiteEnabled('https://another.example/editor', getSettings({ enabled: false }))).toBe(false);
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
