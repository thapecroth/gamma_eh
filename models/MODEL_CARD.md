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

The [public natural benchmark](../docs/public-benchmark.md) reruns the shipped
policy on all 747 JFLEG and 6,845 CWEB test sentences in the actual browser.
Official JFLEG corpus GLEU is 40.62 for guarded neural inference, 46.51 for
rules/combined, and 40.47 for unchanged input. Standard CWEB ERRANT F0.5 is
0.77 for neural and 4.05 for rules/combined. Neural proposes only two CWEB edits;
rules/combined match 25 edits and produce 423 unmatched edits, changing
315/5,881 reference-accepted sources. These results expose limited correction
coverage and full-engine overcorrection; they do not qualify a reliable general
model. See the [aggregate receipt](../docs/public-benchmark-results.json).

The complete JFLEG test population contains 747 sentences with four fluency
references each, including 182 sources accepted unchanged by at least one
reference. Actual JAX JS FP32 Chromium/WASM predictions with the current engine
at the existing 0.85 threshold, one correction pass, produced:

| Engine | Edit precision | Edit recall | Edit F0.5 | Clean sources changed |
| --- | --- | --- | --- | --- |
| Rules, including local dictionary | 71.90% | 16.90% | 43.56% | 2/182 |
| Neural | 87.50% | 0.47% | 2.31% | 0/182 |
| Combined | 71.90% | 16.90% | 43.56% | 2/182 |

These are custom best-sentence-reference scores using actual ERRANT edit
extraction, not official JFLEG GLEU, BEA scores, or universal grammar accuracy.
No sentences were removed for vocabulary, alignment, or inference; no inference
failed. JFLEG includes stylistic rewrites. Neural precision is based on only eight
predictions (seven correct); its very low recall and small support do not qualify
it as a broadly accurate model. The model remains an experimental opt-in baseline.
The independent
benchmark was never added to training. See the [quality workflow](../docs/model-quality.md)
for frozen population hashes, methodology, controlled candidates, and deployment
gates. Raw benchmark text and browser predictions stay local.

Development has 754 sentences and 216 accepted unchanged sources. Rules change
38/216 development clean sources, neural changes 1/216, and combined changes
38/216. The low test rules clean-change rate does not certify development
quality. Four controlled schema-2 students were also trained and executed in the
browser. The best development diagnostic was mixed data with a larger BERT;
its test neural F0.5 reached 7.78%, but edit precision was only 19.77%. None
met the precision, clean-text and minimum-support gates, so their exports
disable neural edits and the bundled weights remain unchanged. See the
[completed results](../docs/model-quality-results.md) and aggregate hashes.

A further controlled synthetic tuning study trained four students on 58,479 CC0
pairs, varying KEEP loss weight and learning rate. None met natural development
qualification, so no new weights were promoted. General safeguards for the bundled
schema-1 decoder instead reduced false neural test edits from 131 to 1, retaining
all seven true edits. Combined test F0.5 rose from 38.02% to 43.56%, and changed
clean sources fell from 24/182 to 2/182. The final integrated engine was compared
with the preceding main engine on the complete test population in the actual
browser, verifying public API equivalence for every sentence. Combined scores now
equal rules alone on this population. The historical quality-round measurements
remain in their original results page. See [controlled model tuning](../docs/model-tuning.md)
and its aggregate receipts for the distinct study and decoder comparison.

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
Legacy lexical replacements require a known canonical typo or a nearby dictionary
word for an unknown source; known words and homophones abstain. Whole spelling
boundaries protect identifiers, compounds and Unicode fragments. Legacy article
insertion requires bounded subject and countable-object evidence. These restrictions
apply to the bundled schema-1 classifier; the richer schema-2 edit contract is preserved.
Rule/model overlap gives rules priority. Classifier
metrics above do not include these runtime guards.

Local AI is disabled by default and marked experimental; users can explicitly
enable it. The finite vocabulary cannot perform general rewriting, arbitrary
insertions, clause rearrangement, or style/tone explanations. Broad local spelling
suggestions now come from the separate bundled dictionary. Long texts lose
context across 64-token windows. Do not use the
checkpoint to label its own evaluation set. Hardware/OS coverage remains limited
to this local Chromium validation; the browser's chosen adapter must be recorded
before making hardware latency claims.

The completed quality round added 10,000 C4 and 10,436 GLM teacher pairs.
Next steps include correction-label review and clean-text precision improvements.
See [the quality results](../docs/model-quality-results.md)
and [the larger dataset workflow](../docs/massive-dataset.md). Raw teacher outputs
remain train-only weak supervision until independently reviewed.
