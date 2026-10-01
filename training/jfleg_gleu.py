"""Run the pinned official JFLEG scorer from ignored research artifacts."""
import hashlib
import json
from pathlib import Path
import ast
import math
import subprocess
import sys
import urllib.request

REVISION = 'ee06ff806a208aba815ac45313f4e750a48330a5'
SHA256 = 'b8bf3605b3c23a899734b406855dcfd8404ec7d8aa84cdfb25114439e5af9c29'


def score(predictions, original, workspace, split='test'):
    workspace.mkdir(parents=True, exist_ok=True)
    scorer = workspace / 'official-gleu.py'
    if not scorer.exists():
        url = f'https://raw.githubusercontent.com/keisks/jfleg/{REVISION}/eval/gleu.py'
        with urllib.request.urlopen(url, timeout=60) as response: payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256: raise ValueError('Official scorer checksum mismatch')
        scorer.write_bytes(payload)
    if hashlib.sha256(scorer.read_bytes()).hexdigest() != SHA256: raise ValueError('Scorer changed')
    source = original / f'{split}.src'
    if len(source.read_text().splitlines()) != len(predictions): raise ValueError('JFLEG population mismatch')
    if any(len((original / f'{split}.ref{i}').read_text().splitlines()) != len(predictions) for i in range(4)):
        raise ValueError('JFLEG reference population mismatch')
    if any('\n' in prediction or '\r' in prediction for prediction in predictions): raise ValueError('Multiline hypothesis')
    hypothesis = workspace / 'hypotheses.txt'
    hypothesis.write_text('\n'.join(predictions) + '\n')
    command = [sys.executable, str(scorer), '-r', *[str(original / f'{split}.ref{i}') for i in range(4)],
               '-s', str(source), '--hyp', str(hypothesis)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    (workspace / 'official-gleu-output.txt').write_text(result.stdout)
    values = ast.literal_eval(result.stdout.strip().splitlines()[-1])
    if len(values) != 1 or len(values[0]) != 3: raise ValueError('Unexpected GLEU statistics')
    mean, std, interval = values[0]
    limits = [float(value) for value in interval.strip('()').split(',')]
    return {'gleu': float(mean), 'reference_sampling_std': float(std),
            'reference_sampling_interval': [value if math.isfinite(value) else None for value in limits],
            'revision': REVISION, 'scorer_sha256': SHA256,
            'note': 'Official JFLEG 4-gram GLEU, 500 seeded reference draws. Interval describes reference sampling, not population uncertainty.'}
