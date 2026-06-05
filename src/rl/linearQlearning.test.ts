import { describe, expect, test } from 'vitest';
import { LinearQLearningAgent, type LinearQHyperParams, type SerializedLinearPolicy } from './linearQlearning';
import type { Observation } from '../env/observation';
import { PacmanEnvironment } from '../env/environment';
import { DIRECTIONS } from '../engine/types';
import { SeededRng } from '../engine/prng';

const hyper = (over: Partial<LinearQHyperParams> = {}): LinearQHyperParams => ({
  alpha: 0.1, gamma: 0.9, epsilon: 0, epsilonDecay: 1, epsilonMin: 0, ...over,
});

const obs = (over: Partial<Observation> = {}): Observation => ({
  pac: { x: 0, y: 0 },
  ghosts: [],
  wallMask: 0,
  nearestPelletDir: 0,
  ghostRel: [],
  ghostsEdible: false,
  ghostCodes: [0, 0],
  ghostHeadings: [0, 0],
  lastAction: -1,
  pelletsRemainingBucket: 0,
  powerPelletsLeftBucket: 0,
  ...over,
});

const snapshotWeights = (a: LinearQLearningAgent): number[][] => a.weights.map((w) => [...w]);

describe('LinearQLearningAgent', () => {
  // D5.1: weights start at zero (no Math.random), so construction is reproducible.
  test('initializes all weights to zero (D5.1)', () => {
    const a = new LinearQLearningAgent(hyper());
    expect(a.weights).toHaveLength(4);
    expect(a.weights.every((w) => w.length === 9 && [...w].every((x) => x === 0))).toBe(true);
  });

  // D5.1: the core reproducibility guarantee — identical training input yields
  // identical weights. This fails under the old Math.random() init.
  test('training is deterministic for identical input (D5.1)', () => {
    const train = (): number[][] => {
      const a = new LinearQLearningAgent(hyper());
      for (let i = 0; i < 25; i += 1) {
        a.update(obs({ wallMask: i % 16 }), i % 4, 1, obs({ wallMask: (i + 1) % 16 }), false, [0, 1, 2, 3]);
      }
      return snapshotWeights(a);
    };
    expect(train()).toEqual(train());
  });

  // A terminal update (target = reward) nudges the acted action's weights by
  // α·tdError·f(s). With zero init and the bias feature = 1, w_a[bias] = α·reward.
  test('update() moves the acted action toward the TD target (terminal)', () => {
    const a = new LinearQLearningAgent(hyper({ alpha: 0.1 }));
    a.update(obs(), 0, 10, obs(), true, []);
    // bias feature is f[0]=1 → w0[0] = 0 + 0.1 * (10 - 0) * 1 = 1.0
    expect(a.weights[0][0]).toBeCloseTo(1.0);
    // Untouched actions stay at zero.
    expect(a.weights[1].every((x) => x === 0)).toBe(true);
  });

  // With ε=0 the agent is greedy and deterministic; after lifting action 0's
  // weights it must be chosen over the others.
  test('act() is greedy and deterministic at epsilon=0', () => {
    const a = new LinearQLearningAgent(hyper());
    // Make action 2 clearly best by giving its bias weight a large value.
    a.weights[2][0] = 5;
    const pick = (): number => a.act(obs(), [0, 1, 2, 3], () => 0.999);
    expect(pick()).toBe(2);
    expect(pick()).toBe(2); // stable
  });

  // D5.2: with λ=0 (default) the weight update is pure gradient — sanity that the
  // refactored α-scaled decay term doesn't change the no-regularization path.
  test('lambda=0 leaves the update as a pure gradient step (D5.2)', () => {
    const a = new LinearQLearningAgent(hyper({ alpha: 0.5 }));
    a.update(obs(), 1, 4, obs(), true, []);
    expect(a.weights[1][0]).toBeCloseTo(0.5 * 4); // α·reward·bias
  });

  test('serialize/load round-trips weights (matching numGhosts)', () => {
    const a = new LinearQLearningAgent(hyper());
    a.update(obs({ wallMask: 3 }), 1, 5, obs({ wallMask: 4 }), false, [0, 1, 2, 3]);
    const ser = a.serialize('pacman-classic', 2);
    expect(ser.algorithm).toBe('linear-qlearning');

    const b = new LinearQLearningAgent(hyper());
    b.load(ser, 2);
    expect(snapshotWeights(b)).toEqual(snapshotWeights(a));
  });

  test('load preserves the live agent epsilon (does not adopt the saved value)', () => {
    const a = new LinearQLearningAgent(hyper({ epsilon: 0.01 }));
    const ser = a.serialize('m', 2);
    const b = new LinearQLearningAgent(hyper({ epsilon: 0.5 }));
    b.load(ser, 2);
    expect(b.hyper.epsilon).toBe(0.5); // live exploration kept, not the saved 0.01
  });

  test('load discards weights when the feature schema version differs', () => {
    const a = new LinearQLearningAgent(hyper());
    a.update(obs(), 0, 9, obs(), true, []);
    const ser = a.serialize('m', 2);
    (ser as SerializedLinearPolicy).version = 999;

    const b = new LinearQLearningAgent(hyper());
    b.update(obs(), 0, 9, obs(), true, []); // make b non-zero first
    b.load(ser);
    expect(b.weights.every((w) => [...w].every((x) => x === 0))).toBe(true);
  });

  test('load discards weights when numGhosts mismatches', () => {
    const a = new LinearQLearningAgent(hyper());
    a.update(obs(), 0, 9, obs(), true, []);
    const ser = a.serialize('m', 3); // trained with 3 ghosts

    const b = new LinearQLearningAgent(hyper());
    b.update(obs(), 0, 9, obs(), true, []);
    b.load(ser, 2); // env has 2 → discard
    expect(b.weights.every((w) => [...w].every((x) => x === 0))).toBe(true);
  });

  // N7-style pin behavior, mirrored from the tabular agent.
  test('setTrainedNumGhosts pins, is idempotent, and rejects a different value', () => {
    const a = new LinearQLearningAgent(hyper());
    a.setTrainedNumGhosts(2);
    expect(a.trainedNumGhosts).toBe(2);
    a.setTrainedNumGhosts(2); // idempotent
    expect(a.trainedNumGhosts).toBe(2);
    a.setTrainedNumGhosts(4); // rejected
    expect(a.trainedNumGhosts).toBe(2);
  });

  test('serialize() prefers the pinned trainedNumGhosts over the caller arg', () => {
    const a = new LinearQLearningAgent(hyper());
    a.setTrainedNumGhosts(2);
    expect(a.serialize('m', 5).numGhostsEncoded).toBe(2);
  });

  test('reset() zeros weights and clears the pin', () => {
    const a = new LinearQLearningAgent(hyper());
    a.update(obs(), 0, 9, obs(), true, []);
    a.setTrainedNumGhosts(2);
    a.reset();
    expect(a.weights.every((w) => [...w].every((x) => x === 0))).toBe(true);
    expect(a.trainedNumGhosts).toBeNull();
  });

  // D5.8 + D5.9: features are normalized (no raw-magnitude term), and peekMaxQ
  // reflects the linear value. With action 0's weights set to all-ones, Q(a0) is
  // the sum of the (normalized) features. For the default obs that sum is:
  //   bias 1 + pellet 0.5 + ghost1 1.0 + ghost2 1.0 + (rest 0) = 3.5.
  // Pre-normalization this would have been 1 + 6 + 20 + 20 = 47, so the value
  // pins the normalization.
  test('features are normalized to ~[0,1]; peekMaxQ reads the linear value (D5.8/D5.9)', () => {
    const a = new LinearQLearningAgent(hyper());
    expect(a.peekMaxQ(obs())).toBe(0); // zero weights → zero value (not null)
    a.weights[0].fill(1);
    expect(a.peekMaxQ(obs())).toBeCloseTo(3.5, 5);
  });

  // D5.12: the linear agent + bootstrapping + off-policy is the "deadly triad".
  // Train on a real env and assert the weights stay finite and bounded — a
  // divergence (the failure mode normalization in D5.8 guards against) would
  // blow them up to ±Infinity/NaN.
  test('training on a real env keeps weights bounded — no divergence (D5.12)', () => {
    const env = new PacmanEnvironment();
    env.setParams({ mazeId: 'pacman-classic', numGhosts: 2, maxEpisodeSteps: 200 });
    env.reset(123);
    const a = new LinearQLearningAgent(hyper({ alpha: 0.1, gamma: 0.99, epsilon: 0.3 }));
    const rng = new SeededRng(123);
    let steps = 0;
    while (steps < 1500) {
      const o = env.observe();
      const legal = env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
      const action = a.act(o, legal, () => rng.next());
      const res = env.step(action);
      const nextLegal = res.done ? [] : env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
      a.update(o, action, res.reward, res.obs, res.done, nextLegal);
      steps += 1;
      if (res.done) { a.endEpisode(); env.reset(123 + steps); }
    }
    const all = a.weights.flatMap((w) => [...w]);
    expect(all.every((x) => Number.isFinite(x))).toBe(true);
    expect(all.every((x) => Math.abs(x) < 1e4)).toBe(true);
  }, 20_000);
});
