import { ACTIONS, toAction } from '../engine/types';
import {
  CNN_STATE_SIZE,
  CnnDqnAgent,
  type CnnState,
  type CnnTrainProfile,
  type CnnTransition,
} from './cnnDqn';
import { initializeTensorRuntime, tf } from './tfRuntime';

export type PortableCnnBackend = 'webgl' | 'webgpu';

export interface GpuDeviceInfo {
  renderer: string;
  vendor: string;
  isSoftwareRenderer: boolean | null;
}

export interface PortableCnnBatchResult {
  batchSize: number;
  coldLatencyMs: number;
  warmupUpdates: number;
  timedUpdates: number;
  timedElapsedMs: number;
  updatesPerSec: number;
  samplesPerSec: number;
  lastLoss: number;
  profile: CnnTrainProfile;
  tensorMemory: ReturnType<typeof tf.memory>;
}

export interface PortableCnnBenchmarkResult {
  backend: PortableCnnBackend;
  device: GpuDeviceInfo;
  batches: PortableCnnBatchResult[];
}

export interface PortableCnnBenchmarkOptions {
  backend?: PortableCnnBackend;
  timedUpdates?: number;
  warmupUpdates?: number;
  batchSizes?: readonly number[];
}

const softwareRendererPattern = /swiftshader|llvmpipe|software rasterizer|software renderer/i;

const webglDeviceInfo = (): GpuDeviceInfo => {
  type WebglBackend = {
    getGPGPUContext?: () => { gl: WebGLRenderingContext };
  };
  const gl = (tf.backend() as unknown as WebglBackend).getGPGPUContext?.().gl;
  if (!gl) return { renderer: 'unknown', vendor: 'unknown', isSoftwareRenderer: null };
  const debug = gl.getExtension('WEBGL_debug_renderer_info') as {
    UNMASKED_RENDERER_WEBGL: number;
    UNMASKED_VENDOR_WEBGL: number;
  } | null;
  const renderer = debug
    ? String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL))
    : String(gl.getParameter(gl.RENDERER));
  const vendor = debug
    ? String(gl.getParameter(debug.UNMASKED_VENDOR_WEBGL))
    : String(gl.getParameter(gl.VENDOR));
  return { renderer, vendor, isSoftwareRenderer: softwareRendererPattern.test(`${renderer} ${vendor}`) };
};

const webgpuDeviceInfo = async (): Promise<GpuDeviceInfo> => {
  type GpuAdapter = {
    info?: { vendor?: string; architecture?: string; device?: string; description?: string };
    requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; device?: string; description?: string }>;
  };
  type NavigatorWithGpu = {
    gpu?: { requestAdapter: () => Promise<GpuAdapter | null> };
  };
  const adapter = await (navigator as unknown as NavigatorWithGpu).gpu?.requestAdapter();
  const info = adapter?.info ?? await adapter?.requestAdapterInfo?.();
  const renderer = [info?.architecture, info?.device, info?.description].filter(Boolean).join(' ') || 'unknown';
  const vendor = info?.vendor || 'unknown';
  return {
    renderer,
    vendor,
    isSoftwareRenderer: renderer === 'unknown'
      ? null
      : softwareRendererPattern.test(`${renderer} ${vendor}`),
  };
};

const makeState = (phase: number): CnnState => {
  const data = new Float32Array(CNN_STATE_SIZE);
  for (let index = phase; index < data.length; index += 17) data[index] = 1;
  return { data };
};

const makeBatch = (batchSize: number): CnnTransition[] => {
  const state = makeState(0);
  const nextState = makeState(7);
  return Array.from({ length: batchSize }, (_, index) => ({
    state,
    action: toAction(index % ACTIONS.length),
    reward: index % 3 === 0 ? 1 : -0.25,
    nextState,
    done: false,
    nextLegalActions: [...ACTIONS],
  }));
};

/**
 * Development-only T6 benchmark. Every batch shape gets a separately reported
 * cold update, disposable warm-up, one profiled update, and a sustained timing
 * window. The real Double-DQN bootstrap/backward path runs for both backends.
 */
export const runPortableCnnBenchmark = async (
  options: PortableCnnBenchmarkOptions = {},
): Promise<PortableCnnBenchmarkResult> => {
  const backend = options.backend ?? 'webgl';
  const timedUpdates = options.timedUpdates ?? 30;
  const warmupUpdates = options.warmupUpdates ?? 2;
  const batchSizes = options.batchSizes ?? [1, 16, 64];
  if (!Number.isInteger(timedUpdates) || timedUpdates < 30 || timedUpdates > 100) {
    throw new Error('timedUpdates must be an integer between 30 and 100');
  }
  if (!Number.isInteger(warmupUpdates) || warmupUpdates < 1) {
    throw new Error('warmupUpdates must be a positive integer');
  }
  if (batchSizes.length === 0 || batchSizes.some((size) => !Number.isInteger(size) || size < 1)) {
    throw new Error('batchSizes must contain positive integers');
  }

  const selectedBackend = await initializeTensorRuntime(backend);
  if (selectedBackend !== backend) {
    throw new Error(`${backend} was requested but TensorFlow.js selected '${selectedBackend}'`);
  }
  const device = backend === 'webgl' ? webglDeviceInfo() : await webgpuDeviceInfo();
  const batches: PortableCnnBatchResult[] = [];

  for (const batchSize of batchSizes) {
    const agent = new CnnDqnAgent({
      batchSize,
      replayCapacity: batchSize,
      targetSyncSteps: 10_000,
      seed: 7,
    });
    const batch = makeBatch(batchSize);
    try {
      const coldStartedAt = performance.now();
      let lastLoss = await agent.trainBatch(batch);
      const coldLatencyMs = performance.now() - coldStartedAt;

      for (let update = 0; update < warmupUpdates; update += 1) {
        lastLoss = await agent.trainBatch(batch);
      }
      const profile = await agent.profileTrainBatch(batch);

      const timedStartedAt = performance.now();
      for (let update = 0; update < timedUpdates; update += 1) {
        lastLoss = await agent.trainBatch(batch);
      }
      const timedElapsedMs = performance.now() - timedStartedAt;
      const timedElapsedSec = Math.max(timedElapsedMs / 1_000, 0.001);
      batches.push({
        batchSize,
        coldLatencyMs,
        warmupUpdates,
        timedUpdates,
        timedElapsedMs,
        updatesPerSec: timedUpdates / timedElapsedSec,
        samplesPerSec: (timedUpdates * batchSize) / timedElapsedSec,
        lastLoss,
        profile,
        tensorMemory: tf.memory(),
      });
    } finally {
      agent.dispose();
    }
  }

  return { backend, device, batches };
};

/** Backwards-compatible WebGL entry used by the existing query-gated panel. */
export const runWebglCnnBenchmark = async (
  timedUpdates = 30,
): Promise<PortableCnnBenchmarkResult> => runPortableCnnBenchmark({ backend: 'webgl', timedUpdates });
