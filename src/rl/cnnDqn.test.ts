import { describe, expect, test } from 'vitest';
import { toAction } from '../engine/types';
import { createDefaultEnv } from '../env/environment';
import {
  CNN_GRID_HEIGHT,
  CNN_GRID_WIDTH,
  CnnDqnAgent,
  type CnnState,
  ReplayBuffer,
  doubleDqnTarget,
  encodeCnnState,
  packCnnBatch,
} from './cnnDqn';
import { initializeTensorRuntime } from './tfRuntime';

const zeroState = (): CnnState => ({ data: new Float32Array(CNN_GRID_WIDTH * CNN_GRID_HEIGHT * 6) });

describe('CNN Double-DQN research primitives', () => {
  test('encodes the six fixed board planes with padding walls and Pac-Man', () => {
    const env = createDefaultEnv();
    const state = encodeCnnState(env);
    expect(state.data).toHaveLength(CNN_GRID_WIDTH * CNN_GRID_HEIGHT * 6);
    const at = (x: number, y: number, plane: number) => state.data[(y * CNN_GRID_WIDTH + x) * 6 + plane];
    const pac = env.getPacmen()[0].pos;
    expect(at(pac.x, pac.y, 3)).toBe(1);
    expect(at(0, 0, 0)).toBe(1);
  });

  test('replay samples deterministically and retains copied transitions', () => {
    const replay = new ReplayBuffer(2);
    const state = zeroState();
    replay.push({ state, action: toAction(0), reward: 1, nextState: state, done: false, nextLegalActions: [toAction(1)] });
    state.data[0] = 99;
    replay.push({ state: zeroState(), action: toAction(1), reward: 2, nextState: zeroState(), done: true, nextLegalActions: [] });
    const draws = [0, 0];
    const sample = replay.sample(2, () => draws.shift() ?? 0);
    expect(sample.map((item) => item.reward)).toEqual([1, 2]);
    expect(sample[0].state.data[0]).toBe(0);
  });

  test('Double-DQN chooses with online values and evaluates with target values under legal masking', () => {
    // Action 3 has the greatest online value but is illegal; action 1 wins the
    // masked argmax, then target[1] — not online[1] — supplies the bootstrap.
    expect(doubleDqnTarget(2, false, 0.5, [1, 9, 3, 100], [10, 20, 30, 40], [toAction(0), toAction(1), toAction(2)])).toBe(12);
    expect(doubleDqnTarget(2, true, 0.5, [1, 9, 3, 100], [10, 20, 30, 40], [toAction(1)])).toBe(2);
  });

  test('packs replay data into contiguous typed arrays with legal bootstrap masks', () => {
    const state = zeroState();
    state.data[3] = 0.75;
    const packed = packCnnBatch([
      { state, action: toAction(2), reward: 4, nextState: state, done: false, nextLegalActions: [toAction(1), toAction(3)] },
      { state, action: toAction(0), reward: -2, nextState: state, done: true, nextLegalActions: [] },
    ]);
    expect(packed.states).toBeInstanceOf(Float32Array);
    expect(packed.states).toHaveLength(2 * CNN_GRID_WIDTH * CNN_GRID_HEIGHT * 6);
    expect(packed.states[3]).toBe(0.75);
    expect(Array.from(packed.actions)).toEqual([2, 0]);
    expect(Array.from(packed.bootstrapMask)).toEqual([1, 0]);
    expect(Array.from(packed.legalActionBias.slice(0, 4))).toEqual([-1e9, 0, -1e9, 0]);
  });

  test('a deterministic terminal batch reduces Huber loss', async () => {
    // tfRuntime's WASM availability check can run in the same Vitest worker;
    // model construction needs an initialized backend, not a pending switch.
    await initializeTensorRuntime('cpu');
    const agent = new CnnDqnAgent({ learningRate: 0.01, targetSyncSteps: 100, seed: 7 });
    // Two stride-2 convolutions reduce the dense input to 8×7×32. The previous
    // same-resolution model had 3,561,492 parameters per network.
    expect(agent.online.countParams()).toBe(235_540);
    const state = zeroState();
    const batch = Array.from({ length: 4 }, () => ({
      state,
      action: toAction(0),
      reward: 1,
      nextState: state,
      done: true,
      nextLegalActions: [],
    }));
    const first = await agent.trainBatch(batch);
    const second = await agent.trainBatch(batch);
    expect(second).toBeLessThan(first);
    agent.dispose();
  });
});
