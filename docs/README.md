# Project documentation

- [Research and architecture](research-and-architecture.md): current WebGPU stack, model choices, correction flow, extension boundaries.
- [Local spelling](local-spelling.md): bundled dictionary, symmetric-delete lookup, ranking, provenance and safety boundaries.
- [Dataset and training](dataset-and-training.md): provenance, deterministic generation, training, calibration, exports, evaluation limits.
- [Architecture experiments](architecture-experiments.md): natural-text holdouts, matched tagger comparisons, browser measurements and local span generation.
- [Scaling the dataset](massive-dataset.md): resumable teacher generation, pinned C4 streaming, and train-only weak supervision.
- [End-to-end teacher pilot](live-pilot.md): real CLIProxyAPI calls, isolated student training and builds, and browser verification.
- [Roadmap](roadmap.md): baseline status and the gates for a useful general writing assistant.
- [Initial validation](validation.md): local checks, browser execution, and known verification limits.
- [Browser verification](browser-testing.md): capability preflight, hosted E2E lane, strict correction checks, and remaining verification gates.
- [Agentic development](agentic-development.md): supervised local Codex coding loop, Luna-max extension scenarios, evidence and bounded repair.
- [Frontend toolchain](toolchain.md): project-local Vite+, version alignment, and web/extension packaging.
- [Releases](releases.md): ready-built Chrome/web ZIPs, checksums, installation, and gated publication.
- [Chrome extension](../apps/extension/README.md): unpacked installation and site permissions.
- [Model card](../models/MODEL_CARD.md): trained checkpoint scope and measured results.
