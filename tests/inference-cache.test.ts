import { describe, expect, it } from 'vite-plus/test';
import { InferenceCache } from '../packages/engine/src/inference-cache';

describe('bounded inference logits cache', () => {
  it('owns a copy of runtime output and promotes reads before entry eviction', () => {
    const cache = new InferenceCache(2);
    const original = new Float32Array([1, 2]);
    cache.set('a', original);
    original.fill(NaN);
    cache.set('b', new Float32Array([3]));
    expect(cache.get('a')).toEqual(new Float32Array([1, 2]));
    cache.set('c', new Float32Array([4]));
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('a')).toEqual(new Float32Array([1, 2]));
    expect(cache.size).toBe(2);
  });

  it('enforces the default 512-entry limit', () => {
    const cache = new InferenceCache();
    for (let i = 0; i < 512; i++) cache.set(String(i), new Float32Array([i]));
    cache.get('0');
    cache.set('512', new Float32Array([512]));
    expect(cache.size).toBe(512);
    expect(cache.get('0')).toEqual(new Float32Array([0]));
    expect(cache.get('1')).toBeUndefined();
    expect(cache.get('512')).toEqual(new Float32Array([512]));
  });

  it('counts UTF-16 keys and adjusts the byte budget when replacing an entry', () => {
    const cache = new InferenceCache(512, 20);
    cache.set('😀', new Float32Array([1, 2])); // 8 data bytes + 4 key bytes.
    cache.set('a', new Float32Array([3]));
    expect(cache.bytes).toBe(18);
    cache.set('😀', new Float32Array([4]));
    expect(cache.bytes).toBe(14);
    cache.set('bb', new Float32Array([5]));
    expect(cache.get('a')).toBeUndefined();
    expect(cache.get('😀')).toEqual(new Float32Array([4]));
    expect(cache.bytes).toBe(16);
  });

  it('enforces the default four MiB bound and refuses oversized entries', () => {
    const cache = new InferenceCache();
    const megabyte = new Float32Array(1024 * 1024 / 4);
    for (const key of ['a', 'b', 'c', 'd']) cache.set(key, megabyte);
    expect(cache.get('a')).toBeUndefined();
    expect(cache.size).toBe(3);
    expect(cache.bytes).toBe(3 * (megabyte.byteLength + 2));
    cache.set('huge', new Float32Array(4 * 1024 * 1024 / 4));
    expect(cache.get('huge')).toBeUndefined();
    expect(cache.size).toBe(3);
    expect(cache.bytes).toBeLessThanOrEqual(4 * 1024 * 1024);
  });
});
