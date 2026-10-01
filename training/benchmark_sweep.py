"""Actual, sequential browser evaluation of frozen local candidates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from architecture_sweep import ARMS
from evaluate import score_browser_report
from jfleg_gleu import score as gleu_score
from pairs import hash_file


def diagnostic_model(source, destination):
    if destination.exists(): return destination
    destination.mkdir(parents=True)
    report = json.loads((source / 'evaluation.json').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest.update(disableModelEdits=False, confidenceThreshold=report['calibration']['best_unconstrained']['threshold'],
                    confidenceThresholds={}, experimental=True, diagnosticOnly=True,
                    originalManifestSha256=hash_file(source / 'manifest.json'))
    for filename in ['labels.json', 'vocab.txt', 'model.onnx', 'model_quantized.onnx']:
        (destination / filename).symlink_to((source / filename).resolve())
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return destination


def benchmark(name, model, directory, destination, modes='model', passes='1,2'):
    raw = destination / f'{name}-browser.json'
    scored = destination / f'{name}-scores.json'
    if not raw.exists():
        subprocess.run(['node', 'scripts/evaluate-browser.mjs', '--model-dir', str(model),
                        '--input', str(directory / 'test.jsonl'), '--output', str(raw),
                        '--modes', modes, '--passes', passes], check=True)
    if not scored.exists():
        report = score_browser_report(raw, directory, 'test', 'errant')
        if directory.name == 'jfleg-evaluation':
            predictions = json.loads(raw.read_text())['runs']
            for condition, run in zip(report['conditions'], predictions):
                condition['official_gleu'] = gleu_score([prediction['corrected'] for prediction in run['predictions']],
                    directory / 'original', destination / 'gleu' / (name + '-' + run['mode'] + str(run['maxPasses'])))
        scored.write_text(json.dumps(report, indent=2) + '\n')
    report = json.loads(scored.read_text())
    print(json.dumps({'name': name, 'execution_passed': report['passed_execution'], 'conditions': [
        {key: condition[key] for key in ['mode', 'max_passes', 'edit_precision', 'edit_recall', 'edit_f0_5',
         'clean_sentence_false_positive_rate', 'inference_failures', 'browser']} for condition in report['conditions']]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=Path('artifacts/architecture-sweep/models'))
    parser.add_argument('--data', type=Path, default=Path('data/prepared/architecture-sweep/evaluation'))
    parser.add_argument('--jfleg', type=Path, default=Path('data/imported/jfleg-evaluation'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/architecture-sweep/browser'))
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    for name, directory in [('research', args.data), ('jfleg', args.jfleg)]:
        benchmark('shipped-' + name, Path('models/browser'), directory, args.output, 'rules,model,combined', '1,2,3')
    if not args.baseline_only:
        winner = max(ARMS, key=lambda arm: json.loads((args.models / arm / 'evaluation.json').read_text())[
            'calibration']['best_unconstrained']['edit_f0_5'])
        (args.output / 'selection.json').write_text(json.dumps({'winner': winner,
            'criterion': 'Highest INT8 development diagnostic F0.5; test not used for selection. Diagnostic-only policies bypass failed safety gates in private artifacts.'}, indent=2) + '\n')
        for arm in ARMS:
            model = diagnostic_model(args.models / arm, args.output / 'diagnostic-models' / arm)
            benchmark(arm + '-research', model, args.data, args.output, 'model', '1,2')
            benchmark(arm + '-jfleg', model, args.jfleg, args.output, 'model', '1')
        model = args.output / 'diagnostic-models' / winner
        for name, directory in [('research', args.data), ('jfleg', args.jfleg)]:
            benchmark('winner-combined-' + name, model, directory, args.output, 'combined', '1,2,3')
