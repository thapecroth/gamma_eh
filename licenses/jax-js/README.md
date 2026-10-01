# JAX JS notices

The MIT license is copied byte for byte from `@jax-js/jax` 0.1.25 and matches
`@jax-js/onnx` 0.1.2. The `onnx-buf` 1.20.0-1 package wrapper uses the same MIT
license in the [upstream repository](https://github.com/ekzhang/jax-js), while its
generated ONNX schema declares Apache-2.0; see `licenses/onnx/` for that notice.

Both downloadable applications include this notice. Runtime versions are pinned
in `package.json`, the lockfile, and `scripts/inference-runtime.mjs`; update the
notices and release metadata together when changing those versions.
