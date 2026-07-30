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
      tf.layers.conv2d({ inputShape: [CNN_GRID_HEIGHT, CNN_GRID_WIDTH, CNN_INPUT_PLANES], filters: 16, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer }),
      tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer }),
      tf.layers.dense({ units: ACTIONS.length, activation: 'linear', kernelInitializer }),
    ],
  });
};

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
    if (batch.length === 0) throw new Error('cannot train an empty batch');
    await this.ready();
    const states = tf.tensor4d(batch.flatMap((t) => Array.from(t.state.data)), [batch.length, CNN_GRID_HEIGHT, CNN_GRID_WIDTH, CNN_INPUT_PLANES]);
    const nextStates = tf.tensor4d(batch.flatMap((t) => Array.from(t.nextState.data)), [batch.length, CNN_GRID_HEIGHT, CNN_GRID_WIDTH, CNN_INPUT_PLANES]);
    const [onlineCurrent, onlineNext, targetNext] = await Promise.all([
      this.online.predict(states) as tf.Tensor2D,
      this.online.predict(nextStates) as tf.Tensor2D,
      this.target.predict(nextStates) as tf.Tensor2D,
    ].map(async (tensor) => ({ tensor, values: await tensor.array() })));
    const targets = onlineCurrent.values.map((values, i) => {
      const transition = batch[i];
      const next = doubleDqnTarget(transition.reward, transition.done, this.hyper.gamma, onlineNext.values[i], targetNext.values[i], transition.nextLegalActions);
      values[transition.action] = next;
      return values;
    });
    onlineCurrent.tensor.dispose();
    onlineNext.tensor.dispose();
    targetNext.tensor.dispose();
    nextStates.dispose();
    const targetTensor = tf.tensor2d(targets, [batch.length, ACTIONS.length]);
    const loss = this.optimizer.minimize(() => tf.tidy(() => {
      const predicted = this.online.apply(states) as tf.Tensor2D;
      return tf.losses.huberLoss(targetTensor, predicted).mean();
    }), true) as tf.Scalar;
    const lossValue = (await loss.data())[0];
    loss.dispose();
    targetTensor.dispose();
    states.dispose();
    this.updatesSinceSync += 1;
    if (this.updatesSinceSync >= this.hyper.targetSyncSteps) this.syncTarget();
    return lossValue;
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
}
