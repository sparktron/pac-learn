/**
 * Shared TensorFlow.js runtime for the browser trainer and Node-based bench.
 *
 * Deliberately use the pure-JS package rather than `@tensorflow/tfjs-node`:
 * the latter is native/platform-specific and cannot be bundled into the Vite
 * app. TensorFlow.js selects WebGL in capable browsers and CPU in the headless
 * bench; T6 records throughput before treating either as viable for training.
 */
import * as tf from '@tensorflow/tfjs';

export { tf };

/** Ensure TensorFlow.js has selected a backend before model construction. */
export const initializeTensorRuntime = async (): Promise<string> => {
  await tf.ready();
  return tf.getBackend();
};
