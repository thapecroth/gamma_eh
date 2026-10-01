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
An independent human-reference evaluation now exposes the generalization gap;
see below. Synthetic token-label scores and decoded browser scores have different
populations, guards, and scoring methods.

## Independent human-reference evaluation

The complete JFLEG test population contains 747 sentences with four fluency
references each, including 182 sources accepted unchanged by at least one
reference. Actual Chromium/WASM predictions at the existing 0.85 threshold,
one correction pass, produced:

| Engine | Edit precision | Edit recall | Edit F0.5 | Clean sources changed |
| --- | --- | --- | --- | --- |
| Rules | 90.91% | 0.67% | 3.27% | 0/182 |
| Neural | 6.27% | 1.06% | 3.16% | 52/182 |
| Combined | 7.72% | 1.32% | 3.92% | 52/182 |

These are custom best-sentence-reference scores using actual ERRANT edit
extraction, not official JFLEG GLEU, BEA scores, or universal grammar accuracy.
No sentences were removed for vocabulary, alignment, or inference; no inference
failed. JFLEG includes stylistic rewrites. The model remains an experimental
opt-in baseline, and the poor clean-text result is a reason to require natural
development calibration before promoting another student. The independent
benchmark was never added to training. See the [quality workflow](../docs/model-quality.md)
for frozen population hashes, methodology, controlled candidates, and deployment
gates. Raw benchmark text and browser predictions stay local.

## Browser engine validation and limitations

The browser apps now execute the FP32 graph through JAX JS on WASM and WebGPU;
the INT8 export remains for evaluation. See [runtime validation](../docs/jax-js-runtime.md).
The original ONNX Runtime validation executed the quantized model in headless
Chromium with the real web editor and MV3 offscreen/worker path. WebGPU also executed with Chrome's explicit
headless WebGPU flag, using its SwiftShader software adapter. Physical-GPU
browser execution and hardware latency remain unverified. A small original natural-text smoke set
matched 15/15 expected outputs after conservative runtime guards and the narrow
habitual-agreement rule. This 15-case set is a smoke test,
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
