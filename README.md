# Gamma EH

An open-source English writing assistant: a web editor, Chrome extension, and
tiny transformer correction engine. Text stays on your device. Suggestions are
applied only when you accept them.

**Experimental baseline:** the first model is fine-tuned on original synthetic
templates. It demonstrates local model training and browser inference; it is
not yet a general Grammarly replacement. See the [model card](models/MODEL_CARD.md).

## Run the editor

Requires Node.js 24.11+ (24.x) and npm. The project-local Vite+ toolchain requires
no global installation. The trained model and browser runtime are bundled
by the build script.

```sh
npm ci
npm run build
npm run dev
```

Open the local address printed by Vite+. Local AI uses WebGPU where available
and falls back to a quantized CPU model. Rules are enabled by default; explicitly
switch on experimental Local AI in the editor or extension popup to try the model.
No API keys, accounts, or inference server are needed.

For browser verification, run `npm run check:environment` and see
[Browser testing](docs/browser-testing.md). Hosted PR checks include actual
WASM/WebGPU inference in the web editor and Chrome extension.

## Chrome extension

```sh
npm ci
npm run build
```

In `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and
select `dist/extension`. Open the extension popup on a regular website and grant
checking for that site. See the [extension instructions](apps/extension/README.md).

The first extension supports textarea and plain contenteditable fields in the
top frame. Rich document editors, Google Docs, shadow roots, and nested frames
are not supported. Password and sensitive fields are excluded. No page text is
sent over the network or stored in extension settings.

## Develop and train

For large teacher-generated runs or ready-made C4 pairs, see
[Scaling the dataset](docs/massive-dataset.md). The resumable generator supports
OpenAI-compatible and Anthropic endpoints; it defaults to a dry run.

```sh
npm run check
uv venv --python 3.10
uv pip install --python .venv/bin/python -r training/requirements.txt
.venv/bin/python training/data.py
.venv/bin/python -m pytest training/test_data.py
.venv/bin/python training/train.py --epochs 8
npm run build
```

Heavy jobs run sequentially on a shared machine. Model training automatically
uses CUDA when available. The seed, base revision, dataset hashes, calibration,
test results, and quantization comparison are recorded beside the browser model.

The frontend uses Vite+ for development, linting, tests, and the Vite core build.
`npm run build` also packages the extension and local inference assets; bare
`vp build` is not a replacement for it. See [Frontend toolchain](docs/toolchain.md).

## Layout

| Path | Purpose |
| --- | --- |
| `apps/web` | React editor; inference in a dedicated worker |
| `apps/extension` | Manifest V3 popup, content script, offscreen inference worker |
| `packages/engine` | Rules, offset-safe edit operations, BERT tokenization, ONNX inference |
| `training` | Dataset generator, training, export, corpus safety tests |
| `models/browser` | Trained weights, tokenizer, manifest, evaluation evidence |
| `docs` | Research, architecture, data provenance, roadmap |

[Documentation index](docs/README.md) · [Research](docs/research-and-architecture.md)
· [Dataset/training](docs/dataset-and-training.md) · [Roadmap](docs/roadmap.md)

Code and trained derivative model: Apache-2.0. Original generated corpus:
CC0-1.0. Google's base checkpoint retains its Apache-2.0 attribution. See
[NOTICE](NOTICE). This independent project is not affiliated with Grammarly.
