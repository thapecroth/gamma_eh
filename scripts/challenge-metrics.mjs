import assert from 'node:assert/strict';

export function applyGold(text, edits) {
  let previousEnd = -1;
  for (const edit of edits) {
    assert(Number.isInteger(edit.start) && Number.isInteger(edit.end) && edit.start >= 0 &&
      edit.end >= edit.start && edit.end <= text.length, 'Invalid UTF-16 gold span');
    assert(edit.start >= previousEnd, 'Overlapping gold spans');
    assert.equal(text.slice(edit.start, edit.end), edit.original, 'Stale gold original');
    assert.equal(typeof edit.replacement, 'string');
    previousEnd = edit.end;
  }
  return [...edits].reverse().reduce((value, edit) => value.slice(0, edit.start) + edit.replacement + value.slice(edit.end), text);
}

export function validateCorpus(corpus) {
  assert.equal(corpus.schema, 1);
  assert.equal(corpus.license, 'CC0-1.0');
  assert(Array.isArray(corpus.cases) && corpus.cases.length > 0, 'Empty challenge corpus');
  const ids = new Set();
  for (const row of corpus.cases) {
    assert(typeof row.id === 'string' && !ids.has(row.id), 'Duplicate case ID');
    ids.add(row.id);
    assert(typeof row.source === 'string' && row.source.length > 0 && row.source.length <= 20_000);
    assert(typeof row.target === 'string' && typeof row.family === 'string');
    assert.equal(row.clean, row.source === row.target, 'Incorrect clean label');
    assert.equal(applyGold(row.source, row.goldEdits), row.target, 'Gold edits must reconstruct target');
    assert.equal(row.goldEdits.length === 0, row.clean);
    assert(Array.isArray(row.requiredTags) && row.requiredTags.every(tag => typeof tag === 'string'));
  }
}

const rate = (numerator, denominator) => denominator ? numerator / denominator : null;

function summary(rows) {
  const clean = rows.filter(row => row.clean);
  const erroneous = rows.filter(row => !row.clean);
  const exact = rows.filter(row => row.actual === row.target).length;
  const cleanChanged = clean.filter(row => row.suggestions.length > 0).length;
  const corrected = erroneous.filter(row => row.actual === row.target).length;
  const abstained = erroneous.filter(row => row.suggestions.length === 0).length;
  const provenance = {};
  for (const row of rows) for (const edit of row.suggestions) provenance[edit.source] = (provenance[edit.source] ?? 0) + 1;
  return {sentences: rows.length, exact_matches: exact, exact_match_rate: rate(exact, rows.length),
    clean_sentences: clean.length, clean_preserved: clean.length - cleanChanged,
    clean_false_positives: cleanChanged, clean_false_positive_rate: rate(cleanChanged, clean.length),
    erroneous_sentences: erroneous.length, fully_corrected: corrected,
    complete_correction_rate: rate(corrected, erroneous.length), abstentions: abstained,
    error_abstention_rate: rate(abstained, erroneous.length),
    incorrect_or_partial_outputs: erroneous.length - corrected - abstained, suggestion_sources: provenance};
}

export function summarizeRows(rows) {
  const families = [...new Set(rows.map(row => row.family))].sort();
  return {...summary(rows), by_family: Object.fromEntries(families.map(family => [family, summary(rows.filter(row => row.family === family))]))};
}
