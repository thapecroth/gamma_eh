"""Local T5-small full-sentence versus detector-gated single-span research pilot.

Gold spans are training supervision only. End-to-end inference uses a frozen
tagger selected on development data. No model or restricted text is uploaded.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from architecture_data import read_rows
from evaluate import encoded_chunks, load_evaluation, onnx_predictor, score_predictions, PROTECTED
from jfleg_gleu import score as gleu_score
from pairs import hash_file

BASE = 'google-t5/t5-small'
REVISION = 'df1b051c49625cf57a3d0d8d3863ed4d13564fe4'


def extend_word(text, start, end):
    while start > 0 and not text[start - 1].isspace(): start -= 1
    while end < len(text) and not text[end].isspace(): end += 1
    return start, end


def gold_span(source, target):
    changes = [op for op in difflib.SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes() if op[0] != 'equal']
    if not changes: return None
    start, end = extend_word(source, changes[0][1], changes[-1][2])
    target_end = len(target) - (len(source) - end)
    replacement = target[start:target_end]
    if source[:start] + replacement + source[end:] != target: raise ValueError('Span alignment failed')
    return start, end, replacement


def make_example(row, objective, index):
    source, target = row['source'], row['target']
    if objective == 'full': return 'correct grammar: ' + source, target
    span = gold_span(source, target)
    # Teach the decoder to recover false detections by copying a masked clean word.
    if span is None:
        choices = list(re.finditer(r'\S+', source))
        if not choices: raise ValueError('Empty source')
        chosen = choices[index % len(choices)]
        span = chosen.start(), chosen.end(), chosen.group()
    start, end, replacement = span
    return ('repair span: ' + source[:start] + '<extra_id_0>' + source[start:end] + '<extra_id_1>' + source[end:],
            '<extra_id_0>' + replacement + '<extra_id_1>')


def masked_inputs(rows, records, threshold):
    selected = []
    for row, record in zip(rows, records):
        edits = [edit for edit in record['proposals'] if edit['confidence'] >= threshold]
        if not edits or record['failed']:
            selected.append(None); continue
        start, end = extend_word(row['source'], min(edit['start'] for edit in edits), max(edit['end'] for edit in edits))
        if any(start < match.end() and match.start() < end for match in PROTECTED.finditer(row['source'])):
            selected.append(None); continue
        selected.append((start, end, 'repair span: ' + row['source'][:start] + '<extra_id_0>' +
                         row['source'][start:end] + '<extra_id_1>' + row['source'][end:]))
    return selected


def decode_span(output):
    match = re.search(r'<extra_id_0>(.*?)<extra_id_1>', output, re.S)
    return match.group(1).strip() if match else None


def predict(model, tokenizer, rows, objective, masks=None, batch_size=8):
    predictions = [row['source'] for row in rows]
    pending = [(i, ('correct grammar: ' + row['source']) if objective == 'full' else masks[i][2])
               for i, row in enumerate(rows) if objective == 'full' or masks[i] is not None]
    supported = []
    failures = 0
    for item in pending:
        if len(tokenizer(item[1])['input_ids']) > 512: failures += 1
        else: supported.append(item)
    pending = supported
    started = time.monotonic(); generated_tokens = 0
    batch_times = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        encoded = tokenizer([text for _, text in batch], padding=True, return_tensors='pt').to(model.device)
        torch.cuda.synchronize(); before = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=256,
                                       num_beams=1, do_sample=False)
        torch.cuda.synchronize(); batch_times.append((time.monotonic() - before) * 1000 / len(batch))
        for (index, _), ids in zip(batch, generated):
            generated_tokens += int((ids != tokenizer.pad_token_id).sum())
            if not bool((ids == tokenizer.eos_token_id).any()): failures += 1; continue
            decoded = tokenizer.decode(ids, skip_special_tokens=objective == 'full', clean_up_tokenization_spaces=False)
            if objective == 'full':
                if decoded.strip(): predictions[index] = decoded.strip()
                else: failures += 1
            else:
                replacement = decode_span(decoded)
                if replacement is None: failures += 1; continue
                left, right, _ = masks[index]
                predictions[index] = rows[index]['source'][:left] + replacement + rows[index]['source'][right:]
    return predictions, {'backend': 'PyTorch CUDA FP32 greedy batch8', 'seconds': time.monotonic() - started,
                         'decoded_sentences': len(pending), 'generated_tokens': generated_tokens, 'failures': failures,
                         'amortized_batch_ms_p50': float(np.percentile(batch_times, 50)) if batch_times else None,
                         'note': 'Amortized batched GPU throughput, not single-request browser latency.'}


def detector_records(rows, model_dir):
    tokenizer, infer, labels, manifest = onnx_predictor(model_dir)
    # Sum non-KEEP probability instead of requiring a particular closed edit
    # label to win. This lets the open decoder consider uncertain edit payloads.
    records = [{'proposals': [], 'failed': False} for _ in rows]
    pending = [(index, chunk) for index, row in enumerate(rows)
               for chunk in encoded_chunks(row['source'], tokenizer, manifest['maxSequenceLength'])]
    before = time.monotonic()
    for start in range(0, len(pending), 32):
        batch = pending[start:start + 32]
        encoded = tokenizer([[word['text'] for word in words] for _, words in batch],
                            is_split_into_words=True, padding=True, return_tensors='np')
        logits = infer({key: encoded[key].astype(np.int64) for key in ['input_ids', 'attention_mask', 'token_type_ids']})
        if not np.isfinite(logits).all(): raise ValueError('Nonfinite detector logits')
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        for item, (index, words) in enumerate(batch):
            protected = [(match.start(), match.end()) for match in PROTECTED.finditer(rows[index]['source'])]
            previous = None
            for position, word_index in enumerate(encoded.word_ids(item)):
                if word_index is None or word_index == previous: continue
                previous = word_index; word = words[word_index]
                if any(word['start'] < end and start < word['end'] for start, end in protected): continue
                records[index]['proposals'].append({'start': word['start'], 'end': word['end'],
                    'confidence': float(1 - probabilities[item, position, 0])})
    return records, {'inference_failures': 0, 'seconds': time.monotonic() - before,
                     'signal': '1 - P(KEEP), independent of winning payload label; frozen INT8 tagger'}


def run(args):
    if args.output.exists(): raise ValueError('Choose a fresh generation output')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.manual_seed(42); random.seed(42); np.random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE, revision=REVISION, local_files_only=True,
                                               attn_implementation='eager').cuda()
    train = read_rows(args.data / 'matched-raw.jsonl')
    examples = [make_example(row, args.objective, index) for index, row in enumerate(train)]
    kept = [example for example in examples if len(tokenizer(example[0])['input_ids']) <= 256
            and len(tokenizer(example[1])['input_ids']) <= 256]
    if len(kept) != len(examples): raise ValueError('Matched training population exceeds generation budget; do not silently filter')
    dev = load_evaluation(args.data / 'evaluation', 'dev')
    records, detector_evidence = detector_records(dev, args.detector) if args.objective == 'span' else (None, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
    best = None; history = []; started = time.monotonic()
    for epoch in range(args.epochs):
        order = list(range(len(kept))); random.Random(42 + epoch).shuffle(order)
        model.train(); total_loss = 0
        for start in range(0, len(order), 8):
            examples = [kept[index] for index in order[start:start + 8]]
            inputs = tokenizer([source for source, _ in examples], padding=True, return_tensors='pt').to('cuda')
            gold = tokenizer(text_target=[target for _, target in examples], padding=True, return_tensors='pt')['input_ids'].cuda()
            gold[gold == tokenizer.pad_token_id] = -100
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=gold).loss
            if not torch.isfinite(loss): raise ValueError('Nonfinite generation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); total_loss += float(loss.detach())
        model.eval(); trials = []
        for threshold in ([.2, .5, .8] if args.objective == 'span' else [None]):
            masks = masked_inputs(dev, records, threshold) if records is not None else None
            predictions, evidence = predict(model, tokenizer, dev, args.objective, masks)
            score = score_predictions(dev, predictions, 'errant', evidence['failures'])
            qualified = (score['edit_precision'] >= .95 and score['predicted_edits'] >= 30 and
                         score['clean_sentences'] >= 50 and score['clean_sentence_false_positive_rate'] <= .02 and not evidence['failures'])
            trials.append({'threshold': threshold, 'metrics': score, 'inference': evidence, 'constraints_met': qualified})
        selected = max(trials, key=lambda trial: (trial['constraints_met'], trial['metrics']['edit_f0_5']))
        history.append({'epoch': epoch + 1, 'loss': total_loss / ((len(kept) + 7) // 8), 'trials': trials})
        print(json.dumps({'epoch': epoch + 1, 'objective': args.objective, 'dev_f0_5': selected['metrics']['edit_f0_5'],
                          'threshold': selected['threshold'], 'loss': history[-1]['loss']}), flush=True)
        if best is None or (selected['constraints_met'], selected['metrics']['edit_f0_5']) > (best['constraints_met'], best['metrics']['edit_f0_5']):
            best = selected; model.save_pretrained(args.output / 'checkpoint'); tokenizer.save_pretrained(args.output / 'checkpoint')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.output / 'checkpoint', local_files_only=True, attn_implementation='eager').cuda().eval()
    report = {'objective': args.objective, 'base': BASE, 'revision': REVISION,
              'publication_allowed': False, 'training_rows': len(kept), 'training_sha256': hash_file(args.data / 'matched-raw.jsonl'),
              'parameters': sum(parameter.numel() for parameter in model.parameters()), 'epochs': history,
              'training_seconds': time.monotonic() - started, 'selected_development': best,
              'detector': str(args.detector) if records is not None else None, 'detector_development_evidence': detector_evidence,
              'detector_hashes': {name: hash_file(args.detector / name) for name in ['manifest.json', 'model_quantized.onnx', 'labels.json']} if records is not None else None,
              'evaluation_hashes': {name: hash_file(args.data / 'evaluation' / f'{name}.jsonl') for name in ['dev', 'test']},
              'limitations': 'One seed, four-epoch pilot. Span infiller uses gold span supervision but actual tagger detections at inference; clean masked words teach identity recovery. No browser decoder or calibrated generator confidence is shipped.'}
    for name, directory in [('research', args.data / 'evaluation'), ('jfleg', args.jfleg)]:
        rows = load_evaluation(directory, 'test')
        candidate_records, detection = detector_records(rows, args.detector) if args.objective == 'span' else (None, None)
        masks = masked_inputs(rows, candidate_records, best['threshold']) if candidate_records is not None else None
        predictions, evidence = predict(model, tokenizer, rows, args.objective, masks)
        score = score_predictions(rows, predictions, 'errant', evidence['failures'])
        report[name] = {'metrics': score, 'inference': evidence, 'detection': detection}
        (args.output / f'{name}-predictions.json').write_text(json.dumps(predictions))
        if name == 'jfleg': report[name]['official_gleu'] = gleu_score(predictions, directory / 'original', args.output / 'gleu')
    report['development_constraints_met'] = best['constraints_met']
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'objective': args.objective, 'test': report['research']['metrics'], 'gleu': report['jfleg']['official_gleu']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', choices=['full', 'span'], required=True)
    parser.add_argument('--data', type=Path, default=Path('data/prepared/architecture-sweep'))
    parser.add_argument('--jfleg', type=Path, default=Path('data/imported/jfleg-evaluation'))
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    run(parser.parse_args())
