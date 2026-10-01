"""Frozen, author-disjoint research data; restricted corpora never become release assets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

from data import edit_tags, tokens
from edit_ops import reconstruct
from pairs import evaluation_keys, hash_file, normalized, pair_id

ARCHIVE_SHA256 = 'd5cbf68cda3da0c3af69dd672614d07287bfe996b87da0c75051d5349d76c666'


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sentence_pairs(document, sentencizer, counts):
    text = document['text']
    annotations = document['edits']
    if len(annotations) != 1 or annotations[0][0] != 0:
        raise ValueError('Unexpected research annotation protocol')
    edits = annotations[0][1]
    previous = 0
    for start, end, correction in edits:
        if not 0 <= start <= end <= len(text) or start < previous:
            raise ValueError('Invalid source edit offsets')
        if correction is not None and not isinstance(correction, str):
            raise ValueError('Invalid correction payload')
        previous = end
    spans = []
    for span in sentencizer(text).sents:
        start, end = span.start_char, span.end_char
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end - 1].isspace(): end -= 1
        if start != end: spans.append((start, end))
    owners = [[] for _ in spans]
    for edit in edits:
        left, right, _ = edit
        affected = [index for index, (start, end) in enumerate(spans)
                    if (start <= left <= end if left == right else left < end and start < right)]
        if left == right and affected: affected = affected[:1]
        if not affected and spans:
            # Corrections in trimmed gaps belong to the preceding window and
            # fail its bounds check; they never silently become clean examples.
            affected = [max((i for i, (start, _) in enumerate(spans) if start <= left), default=0)]
            counts['gap_edit_assignments'] += 1
        for index in affected: owners[index].append(edit)
    for index, (start, end) in enumerate(spans):
        counts['sentence_windows'] += 1
        selected = []
        valid = True
        for left, right, correction in owners[index]:
            if left < start or right > end:
                counts['cross_boundary_edit_windows'] += 1; valid = False; break
            if correction is None:
                counts['unresolved_detection_windows'] += 1; valid = False; break
            selected.append((left, right, correction))
        if not valid: continue
        source = text[start:end]
        target = source
        for left, right, correction in reversed(selected):
            target = target[:left - start] + correction + target[right - start:]
        if not target.strip():
            counts['empty_target_windows'] += 1; continue
        group = document.get('userid') or document['id']
        row = {'id': document['id'] + ':' + str(index), 'document_id': document['id'],
               'author_group': str(group), 'source': source, 'target': target,
               'references': [target], 'category': 'clean' if normalized(source) == normalized(target) else 'mixed',
               'origin': 'WI+LOCNESS-v2.1', 'source_revision': ARCHIVE_SHA256,
               'license': 'noncommercial-research-only', 'publication_allowed': False,
               'review_status': 'original-human-reference', 'pair_id': pair_id(source, target)}
        counts['correction_evaluable_windows'] += 1
        yield row


def stable_sample(rows, limit, seed=42):
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).hexdigest())[:limit]


def freeze(archive_path, template_path, c4_path, jfleg_path, output, rows=20000):
    if output.exists(): raise ValueError('Choose a fresh experiment data directory')
    if hash_file(archive_path) != ARCHIVE_SHA256: raise ValueError('Research archive hash mismatch')
    import spacy
    nlp = spacy.blank('en'); nlp.add_pipe('sentencizer')
    documents = {'train': [], 'heldout': []}
    licenses = {}
    with tarfile.open(archive_path) as archive:
        for split, levels, key in [('train', 'ABC', 'train'), ('dev', 'ABCN', 'heldout')]:
            for level in levels:
                name = f'wi+locness/json/{level}.{split}.json'
                documents[key] += [json.loads(line) for line in archive.extractfile(name)]
        for name in ['wi+locness/licence.wi.txt', 'wi+locness/license.locness.txt']:
            licenses[name] = archive.extractfile(name).read().decode()
    holdout_authors = {str(document.get('userid') or document['id']) for document in documents['heldout']}
    counts = {key: Counter() for key in documents}
    heldout = []
    for document in documents['heldout']:
        heldout += list(sentence_pairs(document, nlp, counts['heldout']))
    evaluation = {'dev': [], 'test': []}
    for row in heldout:
        bucket = int(hashlib.sha256(('gamma-architecture:' + row['author_group']).encode()).hexdigest()[:8], 16) % 2
        evaluation['dev' if bucket == 0 else 'test'].append(row)
    dev_keys = evaluation_keys(evaluation['dev'])
    before = len(evaluation['test'])
    evaluation['test'] = [row for row in evaluation['test'] if normalized(row['source']).casefold() not in dev_keys
                          and normalized(row['target']).casefold() not in dev_keys]
    counts['heldout']['test_rows_removed_for_exact_dev_overlap'] = before - len(evaluation['test'])
    keys = evaluation_keys(heldout)
    for split in ['dev', 'test']: keys |= evaluation_keys(read_rows(jfleg_path / f'{split}.jsonl'))
    training = []
    excluded_documents = 0
    for document in documents['train']:
        if str(document.get('userid') or document['id']) in holdout_authors:
            excluded_documents += 1; continue
        training += list(sentence_pairs(document, nlp, counts['train']))
    training = [row for row in training if normalized(row['source']).casefold() not in keys
                and normalized(row['target']).casefold() not in keys]
    # Single deterministic reference per source; reject conflicting supervision.
    conflicts, by_source = set(), {}
    for row in training:
        source = normalized(row['source']).casefold()
        if source in by_source and normalized(by_source[source]['target']) != normalized(row['target']):
            conflicts.add(source)
        else: by_source[source] = row
    training = [row for source, row in by_source.items() if source not in conflicts]
    templates = read_rows(template_path)
    templates = [{**row, 'origin': 'original-template', 'license': 'CC0-1.0',
                  'pair_id': pair_id(row['source'], row['target'])} for row in templates]
    c4 = read_rows(c4_path)
    def eligible(population):
        return [row for row in population if 3 <= len(row['source']) <= 600 and 3 <= len(row['target']) <= 600
                and normalized(row['source']).casefold() not in keys and normalized(row['target']).casefold() not in keys]
    templates, c4, training = eligible(templates), eligible(c4), eligible(training)
    mixed = stable_sample(training, rows // 2) + stable_sample(c4, rows // 4) + stable_sample(templates, rows // 4)
    baseline = stable_sample(templates, len(mixed))
    if len(baseline) != len(mixed): raise ValueError('Insufficient matched raw template rows')
    output.mkdir(parents=True)
    for split, population in evaluation.items():
        write_rows(output / 'evaluation' / f'{split}.jsonl', population)
    write_rows(output / 'template.jsonl', baseline)
    write_rows(output / 'mixed.jsonl', mixed)
    for name, contents in licenses.items():
        path = output / 'licenses' / Path(name).name; path.parent.mkdir(exist_ok=True); path.write_text(contents)
    manifest = {'schema': 1, 'archive_sha256': ARCHIVE_SHA256, 'research_only': True,
                'publication_allowed': False, 'train_documents_excluded_for_author_overlap': excluded_documents,
                'window_counts': {key: dict(value) for key, value in counts.items()},
                'definition': 'Correction-evaluable sentence windows: unresolved detection-only or cross-window edits excluded before freezing, independent of model/vocabulary; all retained evaluation rows must be scored.',
                'holdout': 'Original BEA development documents split by author hash; official blind test gold unavailable. This is an author-disjoint research holdout, not official BEA-test.',
                'evaluation': {split: {'rows': len(population), 'clean': sum(row['category'] == 'clean' for row in population),
                                     'sha256': hash_file(output / 'evaluation' / f'{split}.jsonl')}
                               for split, population in evaluation.items()},
                'raw_training': {'template': len(baseline), 'mixed': len(mixed),
                                 'mixed_composition': dict(Counter(row['origin'] for row in mixed)),
                                 'template_sha256': hash_file(output / 'template.jsonl'),
                                 'mixed_sha256': hash_file(output / 'mixed.jsonl')},
                'secondary_evaluation': {'dataset': 'JFLEG', 'evaluation_only': True},
                'source_hashes': {'templates': hash_file(template_path), 'c4': hash_file(c4_path)}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def word_windows(words, tags, piece_counts, max_length):
    """Split at whole-word boundaries, keeping every supervised word exactly once."""
    if len(words) != len(tags) or len(words) != len(piece_counts):
        raise ValueError('Word supervision length mismatch')
    if max_length < 4 or any(size < 1 or size > max_length - 2 for size in piece_counts):
        raise ValueError('Individual word exceeds context budget')
    windows, start, size = [], 0, 2
    for index, pieces in enumerate(piece_counts):
        if size + pieces > max_length:
            windows.append((words[start:index], tags[start:index])); start = index; size = 2
        size += pieces
    if start < len(words): windows.append((words[start:], tags[start:]))
    return windows


def tag_dataset(raw_path, output, evaluation_dir, schema, max_labels=2048, token_budget=256, tokenizer=None, matched_keys=None):
    if output.exists(): raise ValueError('Choose a fresh tagged directory')
    counts, tags, records = Counter(), Counter(), []
    for row in read_rows(raw_path):
        counts['raw_rows'] += 1
        if matched_keys is not None and row['pair_id'] not in matched_keys: continue
        try:
            words, labels = edit_tags(row['source'], row['target'], schema)
            expected = row['target'] if schema == 2 else row['target']
            if normalized(reconstruct(words, labels, schema)) != normalized(expected):
                counts['rejected_case_spacing'] += 1; continue
            if tokenizer and len(tokenizer(words, is_split_into_words=True)['input_ids']) > token_budget:
                counts['rejected_context'] += 1; continue
            records.append({**row, 'tokens': words, 'tags': labels}); tags.update(labels)
        except ValueError:
            counts['rejected_alignment'] += 1
    labels = ['KEEP'] + [tag for tag, _ in tags.most_common() if tag != 'KEEP'][:max_labels - 1]
    allowed = set(labels)
    supported = [row for row in records if all(tag in allowed for tag in row['tags'])]
    counts['aligned_rows'] = len(records); counts['unsupported_vocabulary_rows'] = len(records) - len(supported)
    counts['training_rows'] = len(supported)
    output.mkdir(parents=True)
    write_rows(output / 'train.jsonl', supported)
    # Natural evaluation stays complete; optional token-tag diagnostics are empty.
    for split in ['dev', 'test']: write_rows(output / f'{split}.jsonl', [])
    (output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
    manifest = {'schema': 1, 'edit_schema': schema, 'publication_allowed': False,
                'evaluation_scope': 'Complete author-disjoint minimal-edit research holdout; research-only source licenses.',
                'label_count': len(labels), 'counts': dict(counts), 'raw_sha256': hash_file(raw_path),
                'splits': {'train': {'accepted': len(supported), 'sha256': hash_file(output / 'train.jsonl')}},
                'evaluation': {split: {'sha256': hash_file(evaluation_dir / f'{split}.jsonl')}
                               for split in ['dev', 'test']}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--c4', type=Path, required=True)
    parser.add_argument('--jfleg', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=20000)
    args = parser.parse_args()
    print(json.dumps(freeze(args.archive, args.templates, args.c4, args.jfleg, args.output, args.rows), indent=2))
