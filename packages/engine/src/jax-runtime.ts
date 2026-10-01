import { defaultDevice, init, jit, numpy as np } from '@jax-js/jax';
import { ONNXModel } from '@jax-js/onnx';

export type ModelBackend = 'wasm' | 'webgpu';

let wasmReady: Promise<unknown> | undefined;
let gpuReady: Promise<boolean> | undefined;

// Initialize only the supported worker backends; WebGL needs a canvas.
async function initialize(preferWebGPU: boolean): Promise<ModelBackend> {
  await (wasmReady ??= init('wasm'));
  if (!preferWebGPU) return 'wasm';
  gpuReady ??= init('webgpu').then(devices => devices.includes('webgpu')).catch(() => false);
  return await gpuReady ? 'webgpu' : 'wasm';
}

export class JaxSession {
  private readonly model: ONNXModel;
  private readonly forward: ReturnType<typeof jit<ONNXModel['run']>>;

  private constructor(bytes: Uint8Array<ArrayBuffer>, readonly backend: ModelBackend) {
    // ONNX initializers use the default device. Restore it before any await so
    // concurrent WASM/GPU sessions cannot change each other's array placement.
    const previous = defaultDevice();
    defaultDevice(backend);
    try { this.model = new ONNXModel(bytes); }
    finally { defaultDevice(previous); }
    this.forward = jit(this.model.run);
  }

  static async create(bytes: Uint8Array<ArrayBuffer>, preferWebGPU = true): Promise<JaxSession> {
    const backend = await initialize(preferWebGPU);
    if (backend === 'webgpu') {
      // Kernel compilation is lazy. Check actual graph execution before claiming
      // a GPU backend, and keep the same local weights for the WASM fallback.
      let session: JaxSession | undefined;
      try {
        session = new JaxSession(bytes, backend);
        await session.run([101, 102]);
        return session;
      }
      catch {
        session?.dispose();
      }
    }
    return new JaxSession(bytes, 'wasm');
  }

  async run(ids: number[]): Promise<{data: Float32Array; dims: number[]}> {
    if (!ids.length || ids.length > 512) throw new Error('Use between one and 512 token IDs');
    // JIT specializes on shape. Share kernels across eight nearby lengths,
    // adding at most seven masked positions instead of compiling every draft.
    const paddedLength = Math.ceil(ids.length / 8) * 8;
    const shape = [1, paddedLength];
    const inputIds = new Int32Array(paddedLength);
    inputIds.set(ids);
    const attentionMask = new Int32Array(paddedLength).fill(1, 0, ids.length);
    // jax-js uses int32 indices; all WordPiece IDs fit exactly. The exported
    // graph's int64 index/shape tensors are also converted by its ONNX loader.
    const feed: Record<string, np.Array> = {};
    let outputs: Record<string, np.Array> | undefined;
    try {
      feed.input_ids = np.array(inputIds, {shape, device: this.backend});
      feed.attention_mask = np.array(attentionMask, {shape, device: this.backend});
      feed.token_type_ids = np.zeros(shape, {dtype: np.int32, device: this.backend});
      const previous = defaultDevice();
      defaultDevice(this.backend);
      try { outputs = this.forward(feed); }
      finally { defaultDevice(previous); }
      const logits = outputs.logits;
      if (!logits) throw new Error('Missing model logits');
      const dims = logits.shape;
      if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== paddedLength) throw new Error('Unexpected model output shape');
      const data = await logits.data();
      if (!(data instanceof Float32Array)) throw new Error('Unexpected model output dtype');
      if (data.length !== paddedLength * dims[2]) throw new Error('Unexpected model output length');
      return {data: data.subarray(0, ids.length * dims[2]), dims: [1, ids.length, dims[2]]};
    } finally {
      // JAX operations move their inputs; data() also consumes its output.
      // Dispose only references still owned here, including failed evaluations.
      for (const tensor of [...Object.values(feed), ...Object.values(outputs ?? {})]) {
        if (tensor.refCount > 0) tensor.dispose();
      }
    }
  }

  dispose(): void {
    this.forward.dispose();
    this.model.dispose();
  }
}
