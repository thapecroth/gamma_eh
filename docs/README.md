# Project documentation

- [Windows WebGPU initialization](windows-webgpu.md): adapter options, runtime warnings, and rebuilding an unpacked extension.
- [Chrome Web Store](chrome-web-store.md): unlisted submission material and automated approved updates.
- [Extension privacy policy](privacy.md): local text processing, settings, site access, and controls.
- [JAX JS inference](jax-js-runtime.md): bundled WebGPU/WASM runtime, FP32 parity, GPU fallback, and release metadata.
- [Research and architecture](research-and-architecture.md): current WebGPU stack, model choices, correction flow, extension boundaries.
- [Grammar correction literature review](gec-literature-review.md): evidence for compact edit models and span generation, browser tradeoffs, and a controlled experiment plan.
- [Local spelling](local-spelling.md): bundled dictionary, symmetric-delete lookup, ranking, provenance and safety boundaries.
- [Dataset and training](dataset-and-training.md): provenance, deterministic generation, training, calibration, exports, evaluation limits.
- [Harder synthetic challenge](synthetic-challenge.md): separate adversarial evaluation, original training augmentation, and current full-engine baseline.
- [Controlled model tuning](model-tuning.md): frozen CC0 training comparison, actual browser/natural evaluation, promotion gates and lexical safeguards.
- [Scaling the dataset](massive-dataset.md): resumable teacher generation, pinned C4 streaming, and train-only weak supervision.
- [End-to-end teacher pilot](live-pilot.md): real CLIProxyAPI calls, isolated student training and builds, and browser verification.
- [Independent model quality](model-quality.md): frozen human references, expanded edits, GLM/C4 data, safe export calibration, and controlled browser comparisons.
- [Model quality results](model-quality-results.md): completed 10,436-pair GLM corpus, four trained students, actual browser comparisons, and promotion decisions.
- [Roadmap](roadmap.md): baseline status and the gates for a useful general writing assistant.
- [Initial validation](validation.md): local checks, browser execution, and known verification limits.
- [Browser verification](browser-testing.md): capability preflight, hosted E2E lane, automatic site activation, HTTP startup regression, popup pauses, and strict correction checks.
- [Agentic development](agentic-development.md): supervised local Codex coding loop, Luna-max extension scenarios, evidence and bounded repair.
- [Frontend toolchain](toolchain.md): project-local Vite+, version alignment, and web/extension packaging.
- [Releases](releases.md): ready-built Chrome/web ZIPs, checksums, installation, and gated publication.
- [Chrome extension](../apps/extension/README.md): unpacked installation, automatic site access, and per-site pauses.
- [Inline suggestions](inline-suggestions.md): red underlines, hover cards, keyboard access, measurement and browser checks.
- [Model card](../models/MODEL_CARD.md): trained checkpoint scope and measured results.

- [Firefox extension](firefox.md): installation and browser-specific background setup.
