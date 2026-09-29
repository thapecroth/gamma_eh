# Gamma EH

An English writing assistant with a Chrome Manifest V3 extension, a web editor,
and a locally executed tiny transformer correction engine.

- Keep typed text local. Bundle inference runtime and model assets; never fetch executable code from a CDN.
- A model trained on synthetic templates is an experimental baseline. Do not call synthetic scores real-world grammar accuracy.
- Preserve source offsets (UTF-16 in JavaScript). Check for stale text before applying a suggestion.
- Generate datasets deterministically; split by clean sentence before corruption; keep licenses and provenance.
- Run heavy jobs sequentially. `npm run check` covers the app. Python training lives in `training/`.
- Use Conventional Commits. Document infrastructure in `docs/` and link from `docs/README.md`.
