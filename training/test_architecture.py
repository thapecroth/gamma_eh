from collections import Counter
import pytest
import spacy
from architecture_data import sentence_pairs, word_windows
from evaluate import choose_calibration, score_predictions, wilson_interval
from generation_pilot import make_example, masked_inputs, decode_span, gold_span


def test_context_windows_preserve_all_supervision():
    words = ['a', 'b', 'c', 'd', 'e']
    tags = ['KEEP', 'DELETE', 'KEEP', 'CASE:UPPER', 'KEEP']
    windows = word_windows(words, tags, [1, 2, 1, 2, 1], 5)
    assert [word for row, _ in windows for word in row] == words
    assert [tag for _, row in windows for tag in row] == tags
    assert windows == [(['a', 'b'], tags[:2]), (['c', 'd'], tags[2:4]), (['e'], tags[4:])]
    with pytest.raises(ValueError): word_windows(words, tags, [1, 2, 9, 2, 1], 5)


def test_null_detection_is_not_a_deletion_and_final_insertion_is_kept():
    nlp = spacy.blank('en'); nlp.add_pipe('sentencizer')
    document = {'id': 'test', 'userid': 'author', 'text': 'We go. They play',
                'edits': [[0, [[3, 5, None], [16, 16, '.']]]]}
    counts = Counter()
    rows = list(sentence_pairs(document, nlp, counts))
    assert len(rows) == 1
    assert rows[0]['source'] == 'They play'
    assert rows[0]['target'] == 'They play.'
    assert counts['unresolved_detection_windows'] == 1


def test_cross_sentence_correction_is_excluded_before_scoring():
    nlp = spacy.blank('en'); nlp.add_pipe('sentencizer')
    document = {'id': 'test', 'text': 'One. Two.', 'edits': [[0, [[2, 7, 'replacement']]]]}
    counts = Counter()
    assert list(sentence_pairs(document, nlp, counts)) == []
    assert counts['cross_boundary_edit_windows'] == 2


def test_one_correct_edit_does_not_qualify_for_deployment():
    rows = [{'source': 'She have books.', 'references': ['She has books.']},
            *[{'source': 'A clean sentence.', 'references': ['A clean sentence.']} for _ in range(50)]]
    score = {**score_predictions(rows, ['She has books.', *['A clean sentence.'] * 50]), 'threshold': .9}
    identity = {**score_predictions(rows, [row['source'] for row in rows]), 'threshold': 1.}
    assert not choose_calibration([score], identity, min_edits=30, min_clean=50)['constraints_met']
    assert wilson_interval(1, 1)[0] < .21


def test_span_repair_keeps_source_content_for_identity_recovery():
    prompt, target = make_example({'source': 'Zorvani arrived.', 'target': 'Zorvani arrived.'}, 'span', 0)
    assert '<extra_id_0>Zorvani<extra_id_1>' in prompt
    assert decode_span(target) == 'Zorvani'
    source, target = 'She have two books.', 'She has two books.'
    start, end, replacement = gold_span(source, target)
    assert source[:start] + replacement + source[end:] == target
    assert decode_span('malformed response') is None


def test_bounding_spans_cannot_enclose_protected_code():
    row = {'source': 'She have `secret_code` two book.'}
    records = [{'failed': False, 'proposals': [{'start': 4, 'end': 8, 'confidence': .9},
                {'start': 28, 'end': 32, 'confidence': .9}]}]
    assert masked_inputs([row], records, .5) == [None]
