import { describe, expect, test } from 'vitest';
import { QLearningAgent } from './qlearning';
import { observationKey, observationKeyToString, OBSERVATION_KEY_VERSION, type Observation } from '../env/observation';
import { toAction } from '../engine/types';

const obs: Observation = {
  pac: { x: 1, y: 1 },
  pacRegion: 0,
  ghosts: [{ x: 2, y: 2 }],
  wallMask: 0,
  nearestPelletDir: 1,
  ghostRel: [{ dx: 1, dy: 1 }],
  ghostsEdible: false,
  ghostCodes: [1, 0],
  ghostHeadings: [0, 0],
  lastAction: -1,
  pelletsRemainingBucket: 4,
  powerPelletsLeftBucket: 2,
  nearestPelletDist: 1,
  nearestGhostDists: [2, Infinity],
  nearestGhostRel: [{ dx: 1, dy: 1, edible: false }, null],
};

describe('qlearning', () => {
  test('updates q value', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0, optimisticInit: -1 });
    agent.update(obs, toAction(0), 10, obs, true);
    const val = [...agent.q.values()][0][0];
    // init=-1, alpha=0.5, done, reward=10: -1 + 0.5*(10 - -1) = 4.5
    expect(val).toBe(4.5);
  });

  test('nStep accumulates terminal rewards and flushes short episode suffixes', () => {
    const agent = new QLearningAgent({
      alpha: 1, gamma: 0.5, epsilon: 0, epsilonDecay: 1, epsilonMin: 0,
      optimisticInit: 0, nStep: 3,
    });
    const first = { ...obs, wallMask: 1 };
    const second = { ...obs, wallMask: 2 };
    const third = { ...obs, wallMask: 4 };
    agent.update(first, toAction(0), 1, second, false, [0].map(toAction));
    agent.update(second, toAction(0), 2, third, false, [0].map(toAction));
    expect(agent.q.size).toBe(0); // wait until the three-step return is known
    agent.update(third, toAction(0), 3, third, true, []);

    expect(agent.q.get(observationKey(first))?.[0]).toBeCloseTo(2.75); // 1 + .5·2 + .5²·3
    expect(agent.q.get(observationKey(second))?.[0]).toBeCloseTo(3.5); // 2 + .5·3
    expect(agent.q.get(observationKey(third))?.[0]).toBeCloseTo(3);
  });

  test('nStep=1 is identical to the omitted one-step baseline', () => {
    const makeAgent = (nStep?: number) => new QLearningAgent({
      alpha: 0.4, gamma: 0.8, epsilon: 0, epsilonDecay: 1, epsilonMin: 0,
      optimisticInit: 0, ...(nStep === undefined ? {} : { nStep }),
    });
    const baseline = makeAgent();
    const explicit = makeAgent(1);
    const first = { ...obs, wallMask: 1 };
    const second = { ...obs, wallMask: 2 };
    const train = (agent: QLearningAgent) => {
      agent.update(first, toAction(0), 1, second, false, [0].map(toAction));
      agent.update(second, toAction(1), 3, second, true, []);
    };
    train(baseline);
    train(explicit);
    expect([...explicit.q.entries()]).toEqual([...baseline.q.entries()]);
    expect([...explicit.visits.entries()]).toEqual([...baseline.visits.entries()]);
  });

  test('default optimistic init is 50', () => {
    const agent = new QLearningAgent({ alpha: 1, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    // First action samples values; unseen state should return optimistic 50s.
    agent.act(obs, [0].map(toAction), () => 0);
    expect([...agent.q.values()][0]).toEqual(new Float32Array([50, 50, 50, 50]));
  });

  test('breaks greedy ties randomly among legal actions', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });

    expect(agent.act(obs, [1, 2].map(toAction), () => 0)).toBe(1);
    expect(agent.act(obs, [1, 2].map(toAction), () => 0.99)).toBe(2);
  });

  // ── T4: deterministic eval tie-breaks ────────────────────────────────────
  // Set up a state where two legal actions are tied at the top Q-value but
  // differ in visit counts / pellet alignment, then assert each mode resolves
  // it deterministically without consulting the RNG.
  const tieState = (visits: number[]): QLearningAgent => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    const key = observationKey(obs);
    agent.q.set(key, new Float32Array([50, 50, 50, 50])); // all tied
    agent.visits.set(key, new Uint32Array(visits));
    return agent;
  };
  const boom = (): number => { throw new Error('tie-break consulted RNG'); };

  test("'visits' tie-break picks the most-visited tied action, no RNG", () => {
    const agent = tieState([0, 5, 0, 2]); // action 1 most-visited among {1,3}
    expect(agent.act(obs, [1, 3].map(toAction), boom, 'visits')).toBe(1);
  });

  test("'pellet' tie-break steers toward nearestPelletDir when it's tied", () => {
    // obs.nearestPelletDir === 1; action 3 is the most-visited, but pellet wins.
    const agent = tieState([0, 0, 0, 9]);
    expect(agent.act(obs, [3, 1].map(toAction), boom, 'pellet')).toBe(1);
  });

  test("'pellet' falls back to most-visited when pellet dir isn't a candidate", () => {
    // pellet dir 1 not in {2,3}; falls back to visits → action 2.
    const agent = tieState([0, 0, 7, 2]);
    expect(agent.act(obs, [2, 3].map(toAction), boom, 'pellet')).toBe(2);
  });

  test("default tie-break is 'random' (uses the RNG)", () => {
    const agent = tieState([0, 9, 0, 0]); // would pick 1 under 'visits'
    // omitting the arg must preserve baseline random behavior, not go deterministic
    expect(agent.act(obs, [1, 3].map(toAction), () => 0.99)).toBe(3);
  });

  test('bootstraps only from legal next actions', () => {
    const agent = new QLearningAgent({ alpha: 1, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0, optimisticInit: -1 });
    agent.update(obs, toAction(3), 100, obs, true);

    // legal=[1,2]; both at init=-1, so bestNext=-1. target=1+1*(-1)=0.
    // Q[0] = -1 + 1*(0 - -1) = 0. Illegal action 3 (Q=100) must NOT be used.
    agent.update(obs, toAction(0), 1, obs, false, [1, 2].map(toAction));

    expect([...agent.q.values()][0][0]).toBe(0);
  });

  test('loads serialized current-version observation keys without collisions', () => {
    const testObs: Observation = {
      pac: { x: 0, y: 0 },
      pacRegion: 0,
      ghosts: [],
      wallMask: 0,
      nearestPelletDir: 0,
      ghostsEdible: false,
      ghostRel: [],
      ghostCodes: [7, 3],
      ghostHeadings: [1, 2],
      lastAction: 2,
      pelletsRemainingBucket: 3,
      powerPelletsLeftBucket: 1,
      nearestPelletDist: 5,
      nearestGhostDists: [4, 9],
      nearestGhostRel: [{ dx: 2, dy: 2, edible: false }, { dx: -4, dy: 5, edible: false }],
    };
    const key = observationKey(testObs);
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0, optimisticInit: -1 });

    const loaded = agent.load({
      algorithm: 'qlearning',
      mazeId: 'classic',
      timestamp: '2026-05-11T00:00:00.000Z',
      numGhostsEncoded: 2,
      observationKeyVersion: OBSERVATION_KEY_VERSION,
      hyper: agent.hyper,
      qTable: { [observationKeyToString(key)]: [1, 2, 3, 4] }, // key includes lastAction=2
    });

    expect(loaded).toBe(true);
    expect(agent.q.get(key)).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  test('load discards Q-table when policy key version differs', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    const loaded = agent.load({
      algorithm: 'qlearning',
      mazeId: 'classic',
      timestamp: '2026-05-11T00:00:00.000Z',
      numGhostsEncoded: 2,
      observationKeyVersion: 1, // old version
      hyper: agent.hyper,
      qTable: { 'v1:some:old:key': [1, 2, 3, 4] },
    });
    expect(loaded).toBe(false);
    expect(agent.q.size).toBe(0);
  });

  // C2 regression
  test('load preserves the live agent epsilon (does not adopt saved decayed value)', () => {
    const agent = new QLearningAgent({ alpha: 0.2, gamma: 0.99, epsilon: 0.5, epsilonDecay: 0.999, epsilonMin: 0.05 });
    agent.load({
      algorithm: 'qlearning',
      mazeId: 'classic',
      timestamp: '2026-05-11T00:00:00.000Z',
      numGhostsEncoded: 2,
      observationKeyVersion: OBSERVATION_KEY_VERSION,
      hyper: { alpha: 0.1, gamma: 0.95, epsilon: 0.05, epsilonDecay: 0.999, epsilonMin: 0.05 }, // decayed
      qTable: {},
    });
    // ε kept live, but non-exploration hypers should follow the saved values.
    expect(agent.hyper.epsilon).toBe(0.5);
    expect(agent.hyper.alpha).toBe(0.1);
    expect(agent.hyper.gamma).toBe(0.95);
  });

  // D10 (root cause #3, 2026-07-01 investigation): epsilonMinDecay disabled
  // by default — epsilonMin stays fixed even after ε has decayed down to it.
  test('endEpisode: epsilonMin is fixed forever when epsilonMinDecay is unset (default, D10)', () => {
    const agent = new QLearningAgent({ alpha: 0.1, gamma: 0.99, epsilon: 0.2, epsilonDecay: 0.5, epsilonMin: 0.2 });
    for (let i = 0; i < 10; i++) agent.endEpisode();
    expect(agent.hyper.epsilon).toBe(0.2);
    expect(agent.hyper.epsilonMin).toBe(0.2);
  });

  // D10: once ε has reached epsilonMin, epsilonMinDecay shrinks the floor
  // itself each episode, down to epsilonMinFloor — letting exploration keep
  // shrinking over a long tail of training instead of exploring randomly at a
  // fixed rate forever.
  test('endEpisode: epsilonMinDecay shrinks the floor once ε reaches it (D10)', () => {
    const agent = new QLearningAgent({
      alpha: 0.1, gamma: 0.99, epsilon: 0.2, epsilonDecay: 1, epsilonMin: 0.2,
      epsilonMinDecay: 0.5, epsilonMinFloor: 0.05,
    });
    // ε is already at epsilonMin (epsilonDecay=1 is a no-op), so the
    // second-stage decay engages on the very first call. With decay=1, ε
    // itself doesn't separately shrink — Math.max only clamps ε UP to the
    // floor, never down — so after this ε (0.2) sits ABOVE the new, lower
    // floor (0.1) and the condition `ε <= epsilonMin` no longer holds; further
    // calls are inert. A real run uses epsilonDecay<1 so ε keeps tracking the
    // shrinking floor instead of stalling above it (see the next test).
    agent.endEpisode();
    expect(agent.hyper.epsilonMin).toBeCloseTo(0.1);
    agent.endEpisode();
    expect(agent.hyper.epsilonMin).toBeCloseTo(0.1); // stalled: ε(0.2) > epsilonMin(0.1)
  });

  // D10: with a real epsilonDecay<1, ε keeps tracking the shrinking floor
  // instead of getting stuck at the old (higher) one — Math.max re-anchors to
  // whichever is current each call.
  test('endEpisode: with epsilonDecay<1, ε tracks the shrinking floor down (D10)', () => {
    const agent = new QLearningAgent({
      alpha: 0.1, gamma: 0.99, epsilon: 0.2, epsilonDecay: 0.5, epsilonMin: 0.2,
      epsilonMinDecay: 0.5, epsilonMinFloor: 0.01,
    });
    for (let i = 0; i < 6; i++) agent.endEpisode();
    expect(agent.hyper.epsilon).toBeLessThan(0.1);
    expect(agent.hyper.epsilon).toBeCloseTo(agent.hyper.epsilonMin);
  });

  // D10: epsilon still decays normally toward epsilonMin first; the
  // second-stage floor decay must not engage early and cut exploration short.
  test('endEpisode: epsilonMinDecay does not engage before ε reaches epsilonMin (D10)', () => {
    const agent = new QLearningAgent({
      alpha: 0.1, gamma: 0.99, epsilon: 1.0, epsilonDecay: 0.9, epsilonMin: 0.2,
      epsilonMinDecay: 0.5, epsilonMinFloor: 0.05,
    });
    agent.endEpisode(); // ε: 1.0 → 0.9, still well above epsilonMin
    expect(agent.hyper.epsilon).toBeCloseTo(0.9);
    expect(agent.hyper.epsilonMin).toBe(0.2); // untouched — ε hasn't reached the floor yet
  });

  // H9 regression
  test('load discards Q-table when numGhosts mismatches', () => {
    const agent = new QLearningAgent({ alpha: 0.2, gamma: 0.99, epsilon: 0.5, epsilonDecay: 0.999, epsilonMin: 0.05 });
    const key = observationKey(obs);
    const loaded = agent.load(
      {
        algorithm: 'qlearning',
        mazeId: 'classic',
        timestamp: '2026-05-11T00:00:00.000Z',
        numGhostsEncoded: 3,
        observationKeyVersion: OBSERVATION_KEY_VERSION,
        hyper: agent.hyper,
        qTable: { [observationKeyToString(key)]: [1, 2, 3, 4] },
      },
      2, // current env has 2 ghosts → mismatch
    );
    expect(loaded).toBe(false);
    expect(agent.q.size).toBe(0);
  });

  // C1 regression: visit-weighted serialization
  test('serialize emits visitTable; update() increments only the touched slot', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.update(obs, toAction(2), 1, obs, true);
    const ser = agent.serialize('classic', 1);
    expect(ser.visitTable).toBeDefined();
    const visits = Object.values(ser.visitTable!)[0];
    expect(visits).toEqual([0, 0, 1, 0]); // only action 2 was updated
  });

  // N7 regression: trainedNumGhosts tracks what the Q-table was actually
  // trained against so serialize() emits the truthful count even if the
  // caller passes a different numGhosts arg.
  test('setTrainedNumGhosts pins trainedNumGhosts (N7)', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    expect(agent.trainedNumGhosts).toBeNull();
    agent.setTrainedNumGhosts(2);
    expect(agent.trainedNumGhosts).toBe(2);
  });

  test('setTrainedNumGhosts is idempotent for the same value (N7)', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.setTrainedNumGhosts(3);
    agent.setTrainedNumGhosts(3); // same value — must not warn or throw
    expect(agent.trainedNumGhosts).toBe(3);
  });

  test('setTrainedNumGhosts rejects different value without changing it (N7)', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.setTrainedNumGhosts(2);
    agent.setTrainedNumGhosts(4); // different — should be silently rejected
    expect(agent.trainedNumGhosts).toBe(2); // unchanged
  });

  test('reset() clears trainedNumGhosts (N7)', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.setTrainedNumGhosts(2);
    agent.reset();
    expect(agent.trainedNumGhosts).toBeNull();
  });

  test('serialize() uses trainedNumGhosts over the caller arg when pinned (N7)', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.setTrainedNumGhosts(2);
    // Caller passes 5 — serialize must use the pinned 2, not 5.
    const ser = agent.serialize('classic', 5);
    expect(ser.numGhostsEncoded).toBe(2);
  });

  test('serialize() uses caller arg when trainedNumGhosts is null (N7)', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    // trainedNumGhosts is null (never trained) — fall back to the passed arg.
    const ser = agent.serialize('classic', 3);
    expect(ser.numGhostsEncoded).toBe(3);
  });

  // M6: update() bootstraps from optimisticInit=50 when the next state is unseen
  test('update() bootstraps from optimisticInit=50 when next state is unseen (M6)', () => {
    const agent = new QLearningAgent({ alpha: 1, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    const nextObs = { ...obs, wallMask: 1 }; // distinct unseen state
    agent.update(obs, toAction(0), 10, nextObs, false, [0, 1, 2, 3].map(toAction));
    // alpha=1, gamma=1, reward=10, bestNext=50(unseen optimisticInit) → target=60 → Q[0]=60
    expect([...agent.q.values()][0][0]).toBe(60);
  });

  // D7.4: peekMaxQ backs the Q-value overlay — null for unseen states (no
  // phantom insert), max-over-actions for visited ones.
  test('peekMaxQ: null for unseen, max for visited (D7.4)', () => {
    const agent = new QLearningAgent({ alpha: 1, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    expect(agent.peekMaxQ(obs)).toBeNull();
    expect(agent.q.size).toBe(0); // peek must not insert a phantom entry
    // Terminal update with reward above optimisticInit(50) so action 0 is the
    // clear max (the three untouched slots remain at 50).
    agent.update(obs, toAction(0), 100, { ...obs, wallMask: 7 }, true, []); // target=100 → Q[0]=100
    expect(agent.peekMaxQ(obs)).toBe(100);
  });
});
