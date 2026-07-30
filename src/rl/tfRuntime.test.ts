import { describe, expect, test } from 'vitest';
import { initializeTensorRuntime, tf } from './tfRuntime';

describe('TensorFlow.js runtime', () => {
  test('initializes a backend and executes a tensor operation', async () => {
    expect(await initializeTensorRuntime('wasm')).toBe('wasm');
    const result = tf.tidy(() => tf.scalar(2).add(tf.scalar(3)));
    await expect(result.data()).resolves.toEqual(new Float32Array([5]));
    result.dispose();
  });
});
