import { describe, expect, test } from 'vitest';
import { LinearQLearningAgent, extractFeatures, NUM_FEATURES, type LinearQHyperParams, type SerializedLinearPolicy } from './linearQlearning';
import { type Observation, PELLET_SEARCH_RADIUS } from '../env/observation';
import { PacmanEnvironment } from '../env/environment';
import { toAction } from '../engine/types';
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
  nearestPelletDist: PELLET_SEARCH_RADIUS + 1, // "none" sentinel by default
  nearestGhostDists: [Infinity, Infinity],     // no ghosts by default
  nearestGhostRel: [null, null],
  ...over,
});

const ALL = [0, 1, 2, 3].map(toAction);

describe('LinearQLearningAgent', () => {
  // D5.1: weights start at zero (no Math.random), so construction is reproducible.
  test('initializes the shared weight vector to zero (D5.1/D8)', () => {
    const a = new LinearQLearningAgent(hyper());
    expect(a.w.length).toBe(NUM_FEATURES);
    expect([...a.w].every((x) => x === 0)).toBe(true);
  });

  // D5.1: the core reproducibility guarantee — identical training input yields
  // identical weights. This fails under the old Math.random() init.
  test('training is deterministic for identical input (D5.1)', () => {
    const train = (): number[] => {
      const a = new LinearQLearningAgent(hyper());
      for (let i = 0; i < 25; i += 1) {
        a.update(obs({ wallMask: i % 16 }), toAction(i % 4), 1, obs({ wallMask: (i + 1) % 16 }), false, ALL);
      }
      return [...a.w];
    };
    expect(train()).toEqual(train());
  });

  // A terminal update (target = reward) nudges the shared weights by
  // α·tdError·f(s,a). With zero init and the bias feature f[0] = 1,
  // w[bias] = α·reward after one update.
  test('update() moves the weights toward the TD target (terminal)', () => {
    const a = new LinearQLearningAgent(hyper({ alpha: 0.1 }));
    a.update(obs(), toAction(0), 10, obs(), true, []);
    // bias feature is f[0]=1 → w[0] = 0 + 0.1 * (10 - 0) * 1 = 1.0
    expect(a.w[0]).toBeCloseTo(1.0);
    // A feature that is 0 for this (s,a) gets no credit: action 0 with
    // nearestPelletDir=0 means towardPellet f[2]=1, but reverses f[8]=0.
    expect(a.w[8]).toBe(0);
  });

  // D8: features are action-conditioned — the whole point of the rewrite. The
  // towardPellet feature must differ between the action matching
  // nearestPelletDir and any other action.
  test('features are action-conditioned: towardPellet differs per action (D8)', () => {
    const o = obs({ nearestPelletDir: 2 });
    expect(extractFeatures(o, toAction(2))[2]).toBe(1);
    expect(extractFeatures(o, toAction(0))[2]).toBe(0);
  });

  test('wall feature uses the CARD→action bit mapping (D8)', () => {
    // wallMask bits are N/E/S/W (bits 0-3); actions are up/down/left/right.
    // A mask with only the north bit set must flag action up (0), not down (1).
    const o = obs({ wallMask: 0b0001 });
    expect(extractFeatures(o, toAction(0))[1]).toBe(1); // up blocked
    expect(extractFeatures(o, toAction(1))[1]).toBe(0); // down open
    // East bit (1) → action right (3).
    const east = obs({ wallMask: 0b0010 });
    expect(extractFeatures(east, toAction(3))[1]).toBe(1);
    expect(extractFeatures(east, toAction(2))[1]).toBe(0);
  });

  test('danger features reflect post-move ghost distance (D8)', () => {
    // Dangerous ghost 2 to the right: moving right → dist 1 (danger),
    // moving left → dist 3 (safe).
    const o = obs({
      nearestGhostRel: [{ dx: 2, dy: 0, edible: false }, null],
      nearestGhostDists: [2, Infinity],
    });
    const fRight = extractFeatures(o, toAction(3));
    const fLeft = extractFeatures(o, toAction(2));
    expect(fRight[4]).toBe(1); // within 1 after moving right
    expect(fLeft[4]).toBe(0);
    expect(fRight[6]).toBeLessThan(fLeft[6]); // distAfter smaller toward the ghost
  });

  test('edible ghosts feed the chase feature, not the danger features (D8)', () => {
    const o = obs({
      nearestGhostRel: [{ dx: 2, dy: 0, edible: true }, null],
      nearestGhostDists: [2, Infinity],
    });
    const fRight = extractFeatures(o, toAction(3));
    expect(fRight[4]).toBe(0); // not dangerous
    expect(fRight[7]).toBe(1); // approaching an edible ghost
    expect(extractFeatures(o, toAction(2))[7]).toBe(0); // moving away: no chase
  });

  test('reverse feature fires only on the reversing action (D8)', () => {
    const o = obs({ lastAction: 0 }); // last move was up → down (1) reverses
    expect(extractFeatures(o, toAction(1))[8]).toBe(1);
    expect(extractFeatures(o, toAction(0))[8]).toBe(0);
    expect(extractFeatures(obs({ lastAction: -1 }), toAction(1))[8]).toBe(0);
  });

  // With ε=0 the agent is greedy and deterministic; a positive weight on the
  // towardPellet feature must steer it to the pellet direction.
  test('act() is greedy and deterministic at epsilon=0', () => {
    const a = new LinearQLearningAgent(hyper());
    a.w[2] = 5; // reward the towardPellet feature
    const o = obs({ nearestPelletDir: 2 });
    const pick = (): number => a.act(o, ALL, () => 0.999);
    expect(pick()).toBe(2);
    expect(pick()).toBe(2); // stable
  });

  test("'pellet' tie-break selects the pellet direction from tied linear values without RNG", () => {
    const a = new LinearQLearningAgent(hyper()); // zero weights guarantee a tie
    const boom = (): number => { throw new Error('tie-break consulted RNG'); };

    expect(a.act(obs({ nearestPelletDir: 1 }), [3, 1].map(toAction), boom, 'pellet')).toBe(1);
  });

  test("deterministic tie-break falls back to the lowest tied action when pellet direction is unavailable", () => {
    const a = new LinearQLearningAgent(hyper());
    const boom = (): number => { throw new Error('tie-break consulted RNG'); };

    expect(a.act(obs({ nearestPelletDir: 0 }), [3, 2].map(toAction), boom, 'pellet')).toBe(2);
    expect(a.act(obs(), [3, 1].map(toAction), boom, 'visits')).toBe(1);
  });

  test('a negative danger weight makes act() avoid stepping at a ghost', () => {
    const a = new LinearQLearningAgent(hyper());
    a.w[4] = -10; // dangerous-ghost-within-1-after-move is bad
    const o = obs({
      nearestGhostRel: [{ dx: 1, dy: 0, edible: false }, null],
      nearestGhostDists: [1, Infinity],
    });
    // right (3) steps onto the ghost; every other action is preferable.
    expect(a.act(o, ALL, () => 0.0)).not.toBe(3);
  });

  // D5.2: with λ=0 (default) the weight update is pure gradient — sanity that the
  // α-scaled decay term doesn't change the no-regularization path.
  test('lambda=0 leaves the update as a pure gradient step (D5.2)', () => {
    const a = new LinearQLearningAgent(hyper({ alpha: 0.5 }));
    a.update(obs(), toAction(1), 4, obs(), true, []);
    expect(a.w[0]).toBeCloseTo(0.5 * 4); // α·reward·bias
  });

  test('serialize/load round-trips weights (matching numGhosts)', () => {
    const a = new LinearQLearningAgent(hyper());
    a.update(obs({ wallMask: 3 }), toAction(1), 5, obs({ wallMask: 4 }), false, ALL);
    const ser = a.serialize('pacman-classic', 2);
    expect(ser.algorithm).toBe('linear-qlearning');
    expect(ser.weights).toHaveLength(1); // v4: single shared vector

    const b = new LinearQLearningAgent(hyper());
    expect(b.load(ser, 2)).toBe(true);
    expect([...b.w]).toEqual([...a.w]);
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
    a.update(obs(), toAction(0), 9, obs(), true, []);
    const ser = a.serialize('m', 2);
    (ser as SerializedLinearPolicy).version = 999;

    const b = new LinearQLearningAgent(hyper());
    b.update(obs(), toAction(0), 9, obs(), true, []); // make b non-zero first
    expect(b.load(ser)).toBe(false);
    expect([...b.w].every((x) => x === 0)).toBe(true);
  });

  test('load discards weights when numGhosts mismatches', () => {
    const a = new LinearQLearningAgent(hyper());
    a.update(obs(), toAction(0), 9, obs(), true, []);
    const ser = a.serialize('m', 3); // trained with 3 ghosts

    const b = new LinearQLearningAgent(hyper());
    b.update(obs(), toAction(0), 9, obs(), true, []);
    expect(b.load(ser, 2)).toBe(false); // env has 2 → discard
    expect([...b.w].every((x) => x === 0)).toBe(true);
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
    a.update(obs(), toAction(0), 9, obs(), true, []);
    a.setTrainedNumGhosts(2);
    a.reset();
    expect([...a.w].every((x) => x === 0)).toBe(true);
    expect(a.trainedNumGhosts).toBeNull();
  });

  // D9: targetSyncSteps freezes the TD bootstrap target between syncs. With it
  // enabled, an update's bestNextQ is read off the (still zero-init) target
  // weights, not the live weights we just inflated — so the TD error, and the
  // resulting weight change, differ sharply from the same update with the
  // target network disabled (where bootstrapping reads the live weights).
  test('targetSyncSteps freezes the bootstrap target until sync (D9)', () => {
    const withTarget = new LinearQLearningAgent(hyper({ alpha: 0.1, gamma: 1, targetSyncSteps: 100 }));
    withTarget.w.fill(5); // live weights inflated; wTarget still zero-init
    withTarget.update(obs(), toAction(0), 0, obs(), false, ALL);

    const withoutTarget = new LinearQLearningAgent(hyper({ alpha: 0.1, gamma: 1 })); // targetSyncSteps unset (D8 behavior)
    withoutTarget.w.fill(5);
    withoutTarget.update(obs(), toAction(0), 0, obs(), false, ALL);

    // Target-off bootstraps off the same inflated live weights used for
    // currentQ, so the TD error (and weight movement) is small. Target-on
    // bootstraps off a zero target, producing a much larger negative TD error.
    expect(withTarget.w[0]).toBeLessThan(withoutTarget.w[0]);
  });

  // After exactly targetSyncSteps update() calls the target should have just
  // synced to the live weights. Verified via targetSyncSteps=1 (syncs every
  // call, so the target is always exactly one update stale): after update #1
  // the target equals the post-update live weights, so update #2 — bootstrap
  // off that frozen-but-current target — must produce the same result as a
  // fresh no-target agent seeded to that same starting point (which always
  // bootstraps off its live weights).
  test('target network syncs to the live weights after targetSyncSteps updates (D9)', () => {
    const o1 = obs({ wallMask: 3 });
    const o2 = obs({ wallMask: 7 });

    const synced = new LinearQLearningAgent(hyper({ alpha: 0.1, gamma: 1, targetSyncSteps: 1 }));
    synced.update(o1, toAction(0), 1, o2, false, ALL); // also triggers the sync

    const noTarget = new LinearQLearningAgent(hyper({ alpha: 0.1, gamma: 1 }));
    noTarget.w.set(synced.w); // seed to the point synced's target just synced to

    synced.update(o2, toAction(1), 1, o1, false, ALL);
    noTarget.update(o2, toAction(1), 1, o1, false, ALL);

    expect([...synced.w]).toEqual([...noTarget.w]);
  });

  // D5.8: features stay normalized so peekMaxQ reflects a bounded linear value.
  // With all-ones weights, Q(s,a) is the sum of that action's features; for the
  // default obs and action 0 that is bias 1 + towardPellet 1 (pelletDir 0) +
  // pelletDist 1.0 (none sentinel) + dangerDistAfter 1.0 (no ghosts) = 4.0,
  // and no action's sum exceeds it.
  test('features are normalized to ~[0,1]; peekMaxQ reads the linear value (D5.8/D8)', () => {
    const a = new LinearQLearningAgent(hyper());
    expect(a.peekMaxQ(obs())).toBe(0); // zero weights → zero value (not null)
    a.w.fill(1);
    expect(a.peekMaxQ(obs())).toBeCloseTo(4.0, 5);
  });

  // D5.9/D8: the distance features are continuous — distinct distances must map
  // to distinct feature values.
  test('distance features are continuous, not re-discretized (D5.9)', () => {
    const p3 = extractFeatures(obs({ nearestPelletDist: 3 }), toAction(1))[3];
    const p7 = extractFeatures(obs({ nearestPelletDist: 7 }), toAction(1))[3];
    expect(p3).toBeCloseTo(3 / (PELLET_SEARCH_RADIUS + 1), 5);
    expect(p7).toBeCloseTo(7 / (PELLET_SEARCH_RADIUS + 1), 5);

    const g = (dx: number): number =>
      extractFeatures(obs({ nearestGhostRel: [{ dx, dy: 0, edible: false }, null] }), toAction(0))[6];
    expect(g(3)).not.toBeCloseTo(g(9), 5);
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
      const legal = env.getLegalActionIndices();
      const action = a.act(o, legal, () => rng.next());
      const res = env.step(action);
      const nextLegal = res.done ? [] : env.getLegalActionIndices();
      a.update(o, action, res.reward, res.obs, res.done, nextLegal);
      steps += 1;
      if (res.done) { a.endEpisode(); env.reset(123 + steps); }
    }
    const all = [...a.w];
    expect(all.every((x) => Number.isFinite(x))).toBe(true);
    expect(all.every((x) => Math.abs(x) < 1e4)).toBe(true);
  }, 20_000);
});
