# gamma-eh-tiny-edit-v1

Experimental English edit classifier. Fine-tuned from Google's Apache-2.0
`google/bert_uncased_L-2_H-128_A-2`, pinned at
`30b0a37ccaaa32f332884b96992754e246e48c5f`.

| Property | Value |
| --- | --- |
| Parameters | 4,377,793 |
| Transformer | 2 encoder layers, hidden size 128, 2 heads |
| Edit labels | 65: KEEP, DELETE, REPLACE and APPEND variants |
| Context | 64 WordPieces per window |
| Training | 8 epochs, RTX 3070/CUDA, seed 42, 28.21 seconds excluding export |
| Training/dev/test pairs | 47,477 / 5,839 / 6,163 |
| FP32 ONNX | 17,552,569 bytes |
| Quantized INT8 ONNX | 4,463,879 bytes |
| Development-selected confidence | 0.85 |
| License | Apache-2.0 derivative; original synthetic data CC0-1.0 |

## Synthetic classifier evaluation

| Held-out metric | FP32 | INT8 |
| --- | --- | --- |
| Edit precision | 0.999096 | 0.999276 |
| Edit recall | 0.996753 | 0.996212 |
| Edit F0.5 | 0.998627 | 0.998662 |
| Clean-sentence false-positive rate | 0 | 0 |
| Sentence tag accuracy | 0.996593 | 0.996268 |

Full-precision ONNX argmax agrees with PyTorch on all evaluated word positions.
INT8 agreement is 0.999916. Complete training/calibration/export evidence is in
[evaluation.json](browser/evaluation.json), and data hashes/counts are in
[dataset-manifest.json](browser/dataset-manifest.json).

**These high scores measure held-out combinations of the same synthetic template
families and vocabulary. They do not establish general grammar accuracy.**
The separate [controlled tuning study](../docs/model-tuning.md#measured-result)
reports actual-browser JFLEG outputs with custom best-reference ERRANT scoring,
not official JFLEG GLEU. The study retained these weights and improved decoder
safety: test full-engine F0.5 is 43.56 versus 38.02 before the guard, with unchanged
261 correct edits and false edits reduced from 211 to 102 across 747 sentences.
Neural-only recall remains 0.47%; its eight predictions are too few to qualify
a broadly reliable correction policy.

## Browser engine validation and limitations

The browser apps now execute the FP32 graph through JAX JS on WASM and WebGPU;
the INT8 export remains for evaluation. See [runtime validation](../docs/jax-js-runtime.md).
The original ONNX Runtime validation executed the quantized model in headless
Chromium with the real web editor and MV3 offscreen/worker path. WebGPU also executed with Chrome's explicit
headless WebGPU flag, using its SwiftShader software adapter. Physical-GPU
browser execution and hardware latency remain unverified. A small original natural-text smoke set
matched 14/15 expected outputs after conservative runtime guards. The model
missed `My friend go to school every morning.` This 15-case set is a smoke test,
not a representative benchmark.

Before guards, the model produced incorrect article edits and arbitrary
deletions on unseen text. The engine consequently restricts article changes to
known sound classes, permits model deletions only for adjacent duplicates,
and skips neural inference in windows containing protected code/URLs or
rule-driven deletions. Verb replacements must agree with a supported simple
sentence-initial subject and preserve verb family and tense. Unknown or
ambiguous contexts abstain, regardless of model confidence; see
[the verb-suggestion safeguard](../docs/dataset-and-training.md#conservative-verb-suggestions).
Rule/model overlap gives rules priority. Classifier
metrics above do not include these runtime guards.

Lexical replacements now preserve known words, require bounded spelling evidence
for unknown words, and reject pieces inside identifiers/compounds. Article insertion
requires a closed supported singular object; duplicate deletion retains the first
whole word. These restrictions intentionally abstain on ambiguous homophones and
unsupported grammar. See [decoder restrictions](../docs/model-tuning.md#decoder-restrictions).

Local AI is disabled by default and marked experimental; users can explicitly
enable it. The finite vocabulary cannot perform general rewriting, arbitrary
insertions, clause rearrangement, broad spelling correction, or style/tone
explanations. Long texts lose context across64-token windows. Do not use the
checkpoint to label its own evaluation set. Hardware/OS coverage remains limited
to this local Chromium validation; the browser's chosen adapter must be recorded
before making hardware latency claims.

The next quality step is broader licensed correction data and teacher-generated
training pairs plus independently reviewed natural development/test data. See
[the larger dataset workflow](../docs/massive-dataset.md). Raw teacher outputs
must remain train-only weak supervision until independently reviewed.
