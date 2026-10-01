/** Session-local LRU of raw logits; source offsets and edits are never retained. */
export class InferenceCache {
  private readonly entries = new Map<string, Float32Array>();
  private usedBytes = 0;

  constructor(private readonly maxEntries = 512, private readonly maxBytes = 4 * 1024 * 1024) {}

  get size(): number { return this.entries.size; }
  get bytes(): number { return this.usedBytes; }

  get(key: string): Float32Array | undefined {
    const data = this.entries.get(key);
    if (data) {
      this.entries.delete(key);
      this.entries.set(key, data);
    }
    return data;
  }

  set(key: string, data: Float32Array): void {
    const bytes = data.byteLength + key.length * 2;
    if (this.maxEntries < 1 || bytes > this.maxBytes) return;
    const previous = this.entries.get(key);
    if (previous) {
      this.entries.delete(key);
      this.usedBytes -= previous.byteLength + key.length * 2;
    }
    while (this.entries.size >= this.maxEntries || this.usedBytes + bytes > this.maxBytes) {
      const oldest = this.entries.entries().next().value!;
      this.entries.delete(oldest[0]);
      this.usedBytes -= oldest[1].byteLength + oldest[0].length * 2;
    }
    this.entries.set(key, data.slice());
    this.usedBytes += bytes;
  }
}
