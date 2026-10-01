// These pins also identify the upstream notices included in release packages.
export const runtimeVersions = {
  '@jax-js/jax': '0.1.25',
  '@jax-js/onnx': '0.1.2',
  'onnx-buf': '1.20.0-1',
  '@bufbuild/protobuf': '2.16.0',
};
export const runtimeManifest = {schema: 1, engine: 'jax-js', weights: 'model.onnx', dtype: 'float32', versions: runtimeVersions};
