# Roadmap

## First implementation

- Original reproducible English dataset with clean/corrupted pairs and provenance.
- Fine-tuned tiny edit transformer with development calibration and held-out metrics.
- Checked FP32/INT8 ONNX exports and bundled browser assets.
- Shared local correction engine, worker-based web editor, and automatically enabled MV3 extension with per-site pauses, default-enabled Local AI, and compatibility fallback.
- Rules remain useful if model loading fails; UI identifies the actual backend.

## Quality work before a general release

1. Build independently reviewed natural English dev/test sets, with clean text,
   informal messages, technical prose, dialect variation, and real spelling errors.
2. Add a bounded, attributed C4_200M sample and broader human corrections where
   licenses allow; filter and align data with rejection reports.
3. Compare Tiny/MiniLM/BERT-Mini students, larger edit vocabularies, morphology
   transforms, and teacher distillation on the independent development set.
4. Report ERRANT edit F0.5, clean-text false positives, suggestion acceptance,
   calibration, browser p50/p95 latency, startup time, and memory/download size.
5. Validate Chrome on Windows/macOS/Linux hardware with both WebGPU and fallback;
   add narrow adapters for rich editors only after preserving selections/undo/DOM.
6. Package Chrome Web Store metadata, privacy disclosures, signed release assets,
   accessibility checks, and a reproducible model-release process.

This baseline does not provide plagiarism detection, multilingual correction,
tone rewriting, cloud inference, or Grammarly-level quality. Those require
separate product decisions and independent evidence.
