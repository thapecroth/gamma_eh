"""Sequential, reproducible architecture pilots; no release assets are replaced."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from architecture_data import read_rows, stable_sample, tag_dataset, write_rows
from pairs import hash_file

TINY = ('google/bert_uncased_L-2_H-128_A-2', '30b0a37ccaaa32f332884b96992754e246e48c5f')
MINI = ('google/bert_uncased_L-4_H-256_A-4', '387825ce42dbb39b87911cdf8e383ee3b25184f8')
MINILM = ('nreimers/MiniLM-L6-H384-uncased', '3276f0fac9d818781d7a1327b3ff818fc4e643c0')
ARMS = {
    'template-tiny64': ('template-basic', TINY, 64),
    'mixed-basic-tiny64': ('matched-basic', TINY, 64),
    'mixed-rich-tiny64': ('matched-rich', TINY, 64),
    'mixed-rich-mini64': ('matched-rich', MINI, 64),
    'mixed-rich-minilm64': ('matched-rich', MINILM, 64),
    'mixed-rich-tiny128': ('matched-rich', TINY, 128),
    'mixed-rich-tiny256': ('matched-rich', TINY, 256),
    'mixed-rich-full-tiny64': ('full-rich', TINY, 64),
    'mixed-rich-tiny64-long': ('matched-rich', TINY, 64),
    'mixed-rich-mini64-long': ('matched-rich', MINI, 64),
    'mixed-rich-minilm64-long': ('matched-rich', MINILM, 64),
}


def prepare(root):
    evaluation = root / 'evaluation'
    full = {}
    for schema, name in [(1, 'full-basic'), (2, 'full-rich')]:
        full[schema] = tag_dataset(root / 'mixed.jsonl', root / name, evaluation, schema)
    common = {row['pair_id'] for row in read_rows(root / 'full-basic/train.jsonl')} & {
        row['pair_id'] for row in read_rows(root / 'full-rich/train.jsonl')}
    for schema, name in [(1, 'matched-basic'), (2, 'matched-rich')]:
        tag_dataset(root / 'mixed.jsonl', root / name, evaluation, schema, matched_keys=common)
    mixed = [row for row in read_rows(root / 'mixed.jsonl') if row['pair_id'] in common]
    write_rows(root / 'matched-raw.jsonl', mixed)
    template_manifest = tag_dataset(root / 'template.jsonl', root / 'template-basic', evaluation, 1)
    path = root / 'template-basic/train.jsonl'
    templates = stable_sample(read_rows(path), len(mixed))
    write_rows(path, templates)
    template_manifest['counts']['training_rows'] = len(templates)
    template_manifest['splits']['train'] = {'accepted': len(templates), 'sha256': hash_file(path)}
    (root / 'template-basic/manifest.json').write_text(json.dumps(template_manifest, indent=2) + '\n')
    report = {'raw_mixed_rows': full[1]['counts']['raw_rows'], 'matched_rows': len(mixed),
              'basic_coverage': full[1], 'rich_coverage': full[2],
              'definition': 'Basic/rich/backbone/context arms use identical source-target pairs. Full-rich arm measures extra representation coverage separately. Template arm matches row count. Context arms supervise every word once using whole-word windows.',
              'arms': {key: {'data': data, 'base_model': base[0], 'revision': base[1], 'max_length': length}
                       for key, (data, base, length) in ARMS.items()}}
    (root / 'sweep.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'matched_rows': len(mixed), 'basic': full[1]['counts'], 'rich': full[2]['counts']}), flush=True)


def run(root, output, arms, epochs):
    output.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        data, base, length = ARMS[arm]
        arm_epochs = 12 if arm.endswith('-long') else epochs
        destination = output / arm
        if (destination / 'evaluation.json').exists():
            print(f'Completed receipt retained: {arm}', flush=True); continue
        log = output / f'{arm}.log'
        command = [sys.executable, 'training/train.py', '--data', str(root / data),
                   '--evaluation-dir', str(root / 'evaluation'), '--output', str(destination),
                   '--checkpoint', str(output / 'checkpoints' / arm), '--epochs', str(arm_epochs),
                   '--batch-size', '32', '--inference-batch-size', '32', '--learning-rate', '.0001',
                   '--max-length', str(length), '--base-model', base[0], '--base-revision', base[1],
                   '--local-files-only', '--device', 'cuda', '--scorer', 'errant']
        print(f'Starting {arm}: {arm_epochs} epochs; log {log}', flush=True)
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads((destination / 'evaluation.json').read_text())
        print(json.dumps({'arm': arm, 'seconds': report['training_seconds'],
                          'qualified': report['calibration']['constraints_met'],
                          'test': {key: report['diagnostic_unconstrained_test'][key] for key in
                                   ['edit_precision', 'edit_recall', 'edit_f0_5', 'clean_sentence_false_positive_rate']}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/prepared/architecture-sweep'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/architecture-sweep/models'))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    if args.prepare: prepare(args.data)
    else: run(args.data, args.output, args.arms.split(','), args.epochs)
