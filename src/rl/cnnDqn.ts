import { type Action, ACTIONS } from '../engine/types';
import type { PacmanEnvironment } from '../env/environment';
import { initializeTensorRuntime, tf } from './tfRuntime';

export const CNN_GRID_WIDTH = 28;
export const CNN_GRID_HEIGHT = 31;
export const CNN_INPUT_PLANES = 6;
export const CNN_STATE_SIZE = CNN_GRID_WIDTH * CNN_GRID_HEIGHT * CNN_INPUT_PLANES;

export interface CnnState {
  data: Float32Array;
}

/**
 * Encode the classic board as channels-last planes: wall, pellet, power pellet,
 * Pac-Man, dangerous ghosts, edible ghosts. Padding outside a smaller maze is
 * wall-filled so the fixed CNN input remains unambiguous.
 */
export const encodeCnnState = (env: PacmanEnvironment): CnnState => {
  const { width, height } = env.world;
  if (width > CNN_GRID_WIDTH || height > CNN_GRID_HEIGHT) {
    throw new Error(`CNN encoder supports boards up to ${CNN_GRID_WIDTH}×${CNN_GRID_HEIGHT}; received ${width}×${height}`);
  }
  const data = new Float32Array(CNN_STATE_SIZE);
  const offset = (x: number, y: number, plane: number): number =>
    (y * CNN_GRID_WIDTH + x) * CNN_INPUT_PLANES + plane;

  // Padding is wall, preserving a rectangular input without introducing paths.
  for (let y = 0; y < CNN_GRID_HEIGHT; y += 1) {
    for (let x = 0; x < CNN_GRID_WIDTH; x += 1) data[offset(x, y, 0)] = 1;
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      data[offset(x, y, 0)] = env.world.isWall(x, y) ? 1 : 0;
      if (env.world.pellets[y]?.[x]) data[offset(x, y, 1)] = 1;
      if (env.world.powerPellets[y]?.[x]) data[offset(x, y, 2)] = 1;
    }
  }
  const pac = env.getPacmen()[0];
  if (pac) data[offset(pac.pos.x, pac.pos.y, 3)] = 1;
  for (const ghost of env.ghosts) {
    if (ghost.inBox || ghost.releaseDelay > 0) continue;
    data[offset(ghost.pos.x, ghost.pos.y, ghost.edibleTimer > 0 ? 5 : 4)] = 1;
  }
  return { data };
};

export interface CnnTransition {
  state: CnnState;
  action: Action;
  reward: number;
  nextState: CnnState;
  done: boolean;
  nextLegalActions: Action[];
}

/** Fixed-capacity ring buffer; sampling is deterministic when callers supply a seeded RNG. */
export class ReplayBuffer {
  private readonly items: Array<CnnTransition | undefined>;
  private cursor = 0;
  private count = 0;

  constructor(readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new Error(`replay capacity must be positive; received ${capacity}`);
    this.items = new Array(capacity);
  }

  get size(): number { return this.count; }

  push(transition: CnnTransition): void {
    this.items[this.cursor] = {
      ...transition,
      state: { data: new Float32Array(transition.state.data) },
      nextState: { data: new Float32Array(transition.nextState.data) },
      nextLegalActions: [...transition.nextLegalActions],
    };
    this.cursor = (this.cursor + 1) % this.capacity;
    this.count = Math.min(this.capacity, this.count + 1);
  }

  sample(batchSize: number, random: () => number): CnnTransition[] {
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > this.count) {
      throw new Error(`cannot sample batch ${batchSize} from replay size ${this.count}`);
    }
    const available = Array.from({ length: this.count }, (_, i) => i);
    const batch: CnnTransition[] = [];
    for (let i = 0; i < batchSize; i += 1) {
      const selected = Math.floor(random() * available.length);
      const index = available.splice(Math.max(0, Math.min(selected, available.length - 1)), 1)[0];
      const item = this.items[index];
      if (!item) throw new Error(`replay slot ${index} unexpectedly empty`);
      batch.push(item);
    }
    return batch;
  }
}

/** Pure Double-DQN bootstrap with legal-action masking. */
export const doubleDqnTarget = (
  reward: number,
  done: boolean,
  gamma: number,
  onlineNext: readonly number[],
  targetNext: readonly number[],
  nextLegalActions: readonly Action[],
): number => {
  if (done || nextLegalActions.length === 0) return reward;
  let bestAction = nextLegalActions[0];
  for (const action of nextLegalActions) if (onlineNext[action] > onlineNext[bestAction]) bestAction = action;
  return reward + gamma * targetNext[bestAction];
};

export interface CnnDqnHyperParams {
  gamma: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  learningRate: number;
  replayCapacity: number;
  batchSize: number;
  targetSyncSteps: number;
  seed?: number;
}

export const CNN_DQN_DEFAULTS: CnnDqnHyperParams = {
  gamma: 0.997,
  epsilon: 0.3,
  epsilonDecay: 0.9995,
  epsilonMin: 0.05,
  learningRate: 0.001,
  replayCapacity: 50_000,
  batchSize: 64,
  targetSyncSteps: 2_000,
  seed: 7,
};

const createModel = (hyper: CnnDqnHyperParams): tf.LayersModel => {
  const kernelInitializer = tf.initializers.glorotUniform({ seed: hyper.seed });
  return tf.sequential({
    layers: [
      tf.layers.conv2d({ inputShape: [CNN_GRID_HEIGHT, CNN_GRID_WIDTH, CNN_INPUT_PLANES], filters: 16, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', kernelInitializer }),
      tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', kernelInitializer }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer }),
      tf.layers.dense({ units: ACTIONS.length, activation: 'linear', kernelInitializer }),
    ],
  });
};

export interface PackedCnnBatch {
  states: Float32Array;
  nextStates: Float32Array;
  actions: Int32Array;
  rewards: Float32Array;
  bootstrapMask: Float32Array;
  legalActionBias: Float32Array;
}

/**
 * Pack replay objects into contiguous typed arrays once per update. These
 * arrays upload directly to TensorFlow.js without the allocations caused by
 * flatMap(Array.from(...)).
 */
export const packCnnBatch = (batch: readonly CnnTransition[]): PackedCnnBatch => {
  if (batch.length === 0) throw new Error('cannot train an empty batch');
  const states = new Float32Array(batch.length * CNN_STATE_SIZE);
  const nextStates = new Float32Array(batch.length * CNN_STATE_SIZE);
  const actions = new Int32Array(batch.length);
  const rewards = new Float32Array(batch.length);
  const bootstrapMask = new Float32Array(batch.length);
  const legalActionBias = new Float32Array(batch.length * ACTIONS.length);
  legalActionBias.fill(-1e9);

  batch.forEach((transition, index) => {
    if (transition.state.data.length !== CNN_STATE_SIZE || transition.nextState.data.length !== CNN_STATE_SIZE) {
      throw new Error(`CNN transition ${index} must contain ${CNN_STATE_SIZE} values per state`);
    }
    states.set(transition.state.data, index * CNN_STATE_SIZE);
    nextStates.set(transition.nextState.data, index * CNN_STATE_SIZE);
    actions[index] = transition.action;
    rewards[index] = transition.reward;
    if (!transition.done && transition.nextLegalActions.length > 0) bootstrapMask[index] = 1;
    for (const action of transition.nextLegalActions) {
      legalActionBias[index * ACTIONS.length + action] = 0;
    }
  });

  return { states, nextStates, actions, rewards, bootstrapMask, legalActionBias };
};

export interface CnnKernelProfile {
  name: string;
  timeMs: number | null;
  error?: string;
  inputShapes: number[][];
  outputShapes: number[][];
}

export interface CnnTrainProfile {
  loss: number;
  wallMs: number;
  readbackMs: number;
  kernelMs: number;
  newBytes: number;
  peakBytes: number;
  kernels: CnnKernelProfile[];
}

export class CnnDqnAgent {
  readonly replay: ReplayBuffer;
  readonly online: tf.LayersModel;
  readonly target: tf.LayersModel;
  private readonly optimizer: tf.Optimizer;
  private updatesSinceSync = 0;
  hyper: CnnDqnHyperParams;

  constructor(hyper: Partial<CnnDqnHyperParams> = {}) {
    this.hyper = { ...CNN_DQN_DEFAULTS, ...hyper };
    this.replay = new ReplayBuffer(this.hyper.replayCapacity);
    this.online = createModel(this.hyper);
    this.target = createModel(this.hyper);
    this.syncTarget();
    this.optimizer = tf.train.adam(this.hyper.learningRate);
  }

  async ready(): Promise<string> { return initializeTensorRuntime(); }

  async act(state: CnnState, legalActions: Action[], random: () => number): Promise<Action> {
    if (legalActions.length === 0) return ACTIONS[0];
    if (this.hyper.epsilon > 0 && random() < this.hyper.epsilon) {
      return legalActions[Math.floor(random() * legalActions.length)] ?? legalActions[0];
    }
    const qValues = await this.predict(this.online, state);
    let best = legalActions[0];
    for (const action of legalActions) if (qValues[action] > qValues[best]) best = action;
    return best;
  }

  observe(transition: CnnTransition): void { this.replay.push(transition); }

  async learn(random: () => number): Promise<number | null> {
    if (this.replay.size < this.hyper.batchSize) return null;
    return this.trainBatch(this.replay.sample(this.hyper.batchSize, random));
  }

  async trainBatch(batch: readonly CnnTransition[]): Promise<number> {
    await this.ready();
    const loss = this.createBatchLoss(batch);
    try {
      const lossValue = (await loss.data())[0];
      this.completeUpdate();
      return lossValue;
    } finally {
      loss.dispose();
    }
  }

  /** Profile one complete update, including the only intentional GPU readback. */
  async profileTrainBatch(batch: readonly CnnTransition[]): Promise<CnnTrainProfile> {
    await this.ready();
    const startedAt = performance.now();
    const profile = await tf.profile(() => this.createBatchLoss(batch));
    const loss = profile.result as tf.Scalar;
    const kernels: CnnKernelProfile[] = [];
    let kernelMs = 0;
    const readbackStartedAt = performance.now();
    try {
      const lossValue = (await loss.data())[0];
      const readbackMs = performance.now() - readbackStartedAt;
      for (const kernel of profile.kernels) {
        const timing = await Promise.resolve(kernel.kernelTimeMs);
        if (typeof timing === 'number') kernelMs += timing;
        kernels.push({
          name: kernel.name,
          timeMs: typeof timing === 'number' ? timing : null,
          ...(typeof timing === 'number' ? {} : { error: timing.error }),
          inputShapes: kernel.inputShapes,
          outputShapes: kernel.outputShapes,
        });
      }
      this.completeUpdate();
      return {
        loss: lossValue,
        wallMs: performance.now() - startedAt,
        readbackMs,
        kernelMs,
        newBytes: profile.newBytes,
        peakBytes: profile.peakBytes,
        kernels,
      };
    } finally {
      loss.dispose();
    }
  }

  endEpisode(): void {
    this.hyper.epsilon = Math.max(this.hyper.epsilonMin, this.hyper.epsilon * this.hyper.epsilonDecay);
  }

  syncTarget(): void {
    const copied = this.online.getWeights().map((weight) => weight.clone());
    this.target.setWeights(copied);
    copied.forEach((weight) => weight.dispose());
    this.updatesSinceSync = 0;
  }

  dispose(): void {
    this.online.dispose();
    this.target.dispose();
    this.optimizer.dispose();
  }

  private async predict(model: tf.LayersModel, state: CnnState): Promise<number[]> {
    const input = tf.tensor4d(state.data, [1, CNN_GRID_HEIGHT, CNN_GRID_WIDTH, CNN_INPUT_PLANES]);
    const output = model.predict(input) as tf.Tensor2D;
    const values = Array.from(await output.data());
    input.dispose();
    output.dispose();
    return values;
  }

  /**
   * Build Double-DQN targets and selected-action loss entirely as tensors.
   * The returned scalar is the sole value trainBatch reads back to the CPU.
   */
  private createBatchLoss(batch: readonly CnnTransition[]): tf.Scalar {
    const packed = packCnnBatch(batch);
    return tf.tidy(() => {
      const shape: [number, number, number, number] = [
        batch.length,
        CNN_GRID_HEIGHT,
        CNN_GRID_WIDTH,
        CNN_INPUT_PLANES,
      ];
      const states = tf.tensor4d(packed.states, shape);
      const nextStates = tf.tensor4d(packed.nextStates, shape);
      const actions = tf.tensor1d(packed.actions, 'int32');
      const rewards = tf.tensor1d(packed.rewards);
      const bootstrapMask = tf.tensor1d(packed.bootstrapMask);
      const legalActionBias = tf.tensor2d(
        packed.legalActionBias,
        [batch.length, ACTIONS.length],
      );

      // Online chooses the next action; target evaluates it. The legal-action
      // bias keeps argmax on-device and terminal/no-legal rows are zeroed by
      // bootstrapMask.
      const onlineNext = this.online.predict(nextStates) as tf.Tensor2D;
      const bestNextActions = onlineNext.add(legalActionBias).argMax(1);
      const targetNext = this.target.predict(nextStates) as tf.Tensor2D;
      const nextActionMask = tf.oneHot(bestNextActions, ACTIONS.length);
      const selectedTargetNext = targetNext.mul(nextActionMask).sum(1);
      const targets = rewards.add(selectedTargetNext.mul(bootstrapMask).mul(this.hyper.gamma));

      const loss = this.optimizer.minimize(() => {
        const predicted = this.online.apply(states) as tf.Tensor2D;
        const actionMask = tf.oneHot(actions, ACTIONS.length);
        const selectedPredictions = predicted.mul(actionMask).sum(1);
        return tf.losses.huberLoss(
          targets,
          selectedPredictions,
          undefined,
          1,
          tf.Reduction.MEAN,
        );
      }, true);
      if (!loss) throw new Error('CNN optimizer did not produce a loss');
      return loss;
    });
  }

  private completeUpdate(): void {
    this.updatesSinceSync += 1;
    if (this.updatesSinceSync >= this.hyper.targetSyncSteps) this.syncTarget();
  }
}
