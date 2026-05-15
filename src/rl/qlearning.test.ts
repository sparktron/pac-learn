import { describe, expect, test } from 'vitest';
import { QLearningAgent } from './qlearning';
import { observationKey, observationKeyToString, OBSERVATION_KEY_VERSION, type Observation } from '../env/observation';

const obs: Observation = {
  pac: { x: 1, y: 1 },
  ghosts: [{ x: 2, y: 2 }],
  wallMask: 0,
  nearestPelletDir: 1,
  ghostRel: [{ dx: 1, dy: 1 }],
  ghostsEdible: false,
  ghostCodes: [1, 0],
  numEdibleBucket: 0,
};

describe('qlearning', () => {
  test('updates q value', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.update(obs, 0, 10, obs, true);
    const val = [...agent.q.values()][0][0];
    // init=-1, alpha=0.5, done, reward=10: -1 + 0.5*(10 - -1) = 4.5
    expect(val).toBe(4.5);
  });

  test('breaks greedy ties randomly among legal actions', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });

    expect(agent.act(obs, [1, 2], () => 0)).toBe(1);
    expect(agent.act(obs, [1, 2], () => 0.99)).toBe(2);
  });

  test('bootstraps only from legal next actions', () => {
    const agent = new QLearningAgent({ alpha: 1, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.update(obs, 3, 100, obs, true);

    // legal=[1,2]; both at init=-1, so bestNext=-1. target=1+1*(-1)=0.
    // Q[0] = -1 + 1*(0 - -1) = 0. Illegal action 3 (Q=100) must NOT be used.
    agent.update(obs, 0, 1, obs, false, [1, 2]);

    expect([...agent.q.values()][0][0]).toBe(0);
  });

  test('loads serialized v2 observation keys without collisions', () => {
    const testObs: Observation = {
      pac: { x: 0, y: 0 },
      ghosts: [],
      wallMask: 0,
      nearestPelletDir: 0,
      ghostsEdible: false,
      ghostRel: [],
      ghostCodes: [7, 3],
      numEdibleBucket: 1,
    };
    const key = observationKey(testObs);
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });

    agent.load({
      algorithm: 'qlearning',
      mazeId: 'classic',
      timestamp: '2026-05-11T00:00:00.000Z',
      numGhostsEncoded: 2,
      observationKeyVersion: OBSERVATION_KEY_VERSION,
      hyper: agent.hyper,
      qTable: { [observationKeyToString(key)]: [1, 2, 3, 4] },
    });

    expect(agent.q.get(key)).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  test('load discards Q-table when policy key version differs', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.load({
      algorithm: 'qlearning',
      mazeId: 'classic',
      timestamp: '2026-05-11T00:00:00.000Z',
      numGhostsEncoded: 2,
      observationKeyVersion: 1, // old version
      hyper: agent.hyper,
      qTable: { 'v1:some:old:key': [1, 2, 3, 4] },
    });
    expect(agent.q.size).toBe(0);
  });
});
