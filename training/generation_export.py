"""Export a bounded, uncached decoder benchmark, never a release model."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from architecture_data import read_rows
from generation_pilot import detector_records, masked_inputs
from pairs import hash_file


class Encoder(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.encoder = model.encoder
    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.decoder = model.decoder; self.head = model.lm_head
        self.scale = model.model_dim ** -.5 if model.config.tie_word_embeddings else 1.
    def forward(self, input_ids, encoder_hidden_states, attention_mask):
        hidden = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                              encoder_attention_mask=attention_mask, use_cache=False, return_dict=False)[0]
        return self.head(hidden * self.scale)


def greedy(encoder, decoder, ids, maximum=256):
    mask = np.ones((1, len(ids)), dtype=np.int64)
    hidden = encoder.run(None, {'input_ids': np.array([ids], dtype=np.int64), 'attention_mask': mask})[0]
    output = [0]
    for _ in range(maximum):
        logits = decoder.run(None, {'input_ids': np.array([output], dtype=np.int64),
                            'encoder_hidden_states': hidden, 'attention_mask': mask})[0]
        output.append(int(logits[0, -1].argmax()))
        if output[-1] == 1: break
    return output


def run(args):
    output = args.model / 'browser'; output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model / 'checkpoint', local_files_only=True,
                                                attn_implementation='eager').cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'checkpoint', local_files_only=True)
    sample = tokenizer('correct grammar: She have a book.', return_tensors='pt')
    hidden = model.encoder(**sample).last_hidden_state.detach()
    specifications = [('encoder', Encoder(model), (sample['input_ids'], sample['attention_mask']),
                       ['input_ids', 'attention_mask'], 'hidden_states',
                       {'input_ids': {0: 'batch', 1: 'source_length'}, 'attention_mask': {0: 'batch', 1: 'source_length'},
                        'hidden_states': {0: 'batch', 1: 'source_length'}}),
                      ('decoder', Decoder(model), (torch.tensor([[0, 3]], dtype=torch.long), hidden, sample['attention_mask']),
                       ['input_ids', 'encoder_hidden_states', 'attention_mask'], 'logits',
                       {'input_ids': {0: 'batch', 1: 'target_length'}, 'encoder_hidden_states': {0: 'batch', 1: 'source_length'},
                        'attention_mask': {0: 'batch', 1: 'source_length'}, 'logits': {0: 'batch', 1: 'target_length'}})]
    files = {}
    for name, wrapper, sample_args, inputs, result, axes in specifications:
        path = output / f'{name}.onnx'
        torch.onnx.export(wrapper.eval(), sample_args, str(path), input_names=inputs, output_names=[result],
                          dynamic_axes=axes, opset_version=17, dynamo=False)
        onnx.checker.check_model(str(path))
        quantized = output / f'{name}_quantized.onnx'
        quantize_dynamic(str(path), str(quantized), weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gather'])
        for asset in [path, quantized]: files[asset.name] = {'sha256': hash_file(asset), 'bytes': asset.stat().st_size}
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(output / 'encoder_quantized.onnx'), options, providers=['CPUExecutionProvider'])
    decoder = ort.InferenceSession(str(output / 'decoder_quantized.onnx'), options, providers=['CPUExecutionProvider'])
    report = json.loads((args.model / 'evaluation.json').read_text())
    all_rows = read_rows(args.jfleg / 'test.jsonl')
    indices = np.linspace(0, len(all_rows) - 1, 32, dtype=int).tolist()
    rows = [all_rows[index] for index in indices]
    if report['objective'] == 'span':
        records, _ = detector_records(rows, Path(report['detector']))
        masks = masked_inputs(rows, records, report['selected_development']['threshold'])
    else: masks = [None] * len(rows)
    entries = []
    for index, row, mask in zip(indices, rows, masks):
        prompt = 'correct grammar: ' + row['source'] if report['objective'] == 'full' else mask[2] if mask else None
        ids = tokenizer(prompt)['input_ids'] if prompt else None
        expected = greedy(encoder, decoder, ids) if ids else None
        entries.append({'index': index, 'source': row['source'], 'inputIds': ids, 'expectedIds': expected,
                        'span': list(mask[:2]) if mask else None})
    (output / 'inputs.json').write_text(json.dumps(entries) + '\n')
    manifest = {'schema': 1, 'objective': report['objective'], 'files': files, 'inputSha256': hash_file(output / 'inputs.json'),
                'population': '32 evenly spaced fixed JFLEG test rows, selected before inspecting outputs',
                'limitations': 'Prototype uncached greedy encoder/decoder; pretokenized inputs. Measures browser model computation only, excluding tokenizer, detection and UI. Not an optimized production decoder or complete-population quality evaluation.',
                'publication_allowed': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'objective': report['objective'], 'exported_bytes': sum(files[name]['bytes'] for name in files if 'quantized' in name),
                      'rows': len(rows), 'triggered': sum(entry['inputIds'] is not None for entry in entries)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--jfleg', type=Path, default=Path('data/imported/jfleg-evaluation'))
    run(parser.parse_args())
