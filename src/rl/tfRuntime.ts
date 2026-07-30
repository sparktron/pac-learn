/**
 * Shared TensorFlow.js runtime for the browser trainer and Node-based bench.
 *
 * Deliberately use the pure-JS package rather than `@tensorflow/tfjs-node`:
 * the latter is native/platform-specific and cannot be bundled into the Vite
 * app. TensorFlow.js selects WebGL in capable browsers and CPU in the headless
 * bench; T6 records throughput before treating either as viable for training.
 * WebGPU is loaded only when explicitly requested because training support is
 * experimental and the benchmark must surface missing kernels as failures.
 */
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';

export { tf };

export type TensorBackend = 'cpu' | 'wasm' | 'webgl' | 'webgpu';

/** Ensure TensorFlow.js has selected a backend before model construction. */
export const initializeTensorRuntime = async (requestedBackend?: TensorBackend): Promise<string> => {
  if (requestedBackend === 'webgpu') await import('@tensorflow/tfjs-backend-webgpu');
  if (requestedBackend && !await tf.setBackend(requestedBackend)) {
    throw new Error(`TensorFlow.js backend '${requestedBackend}' is unavailable`);
  }
  await tf.ready();
  return tf.getBackend();
};
