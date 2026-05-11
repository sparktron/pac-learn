import { describe, expect, test } from 'vitest';
import { QLearningAgent } from './qlearning';
import { observationKey, observationKeyToString, type Observation } from '../env/observation';

const obs: Observation = { pac: { x: 1, y: 1 }, ghosts: [{ x: 2, y: 2 }], wallMask: 0, nearestPelletDir: 1, ghostRel: [{ dx: 1, dy: 1 }] };

describe('qlearning', () => {
  test('updates q value', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.update(obs, 0, 10, obs, true);
    const val = [...agent.q.values()][0][0];
    expect(val).toBe(5);
  });

  test('loads serialized high-bit observation keys without collisions', () => {
    const highGhostObs: Observation = {
      pac: { x: 0, y: 0 },
      ghosts: [],
      wallMask: 0,
      nearestPelletDir: 0,
      ghostRel: [
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
        { dx: 1, dy: 0 },
      ],
    };
    const key = observationKey(highGhostObs);
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });

    agent.load({
      algorithm: 'qlearning',
      mazeId: 'classic',
      timestamp: '2026-05-11T00:00:00.000Z',
      hyper: agent.hyper,
      qTable: { [observationKeyToString(key)]: [1, 2, 3, 4] },
    });

    expect(agent.q.get(key)).toEqual(new Float32Array([1, 2, 3, 4]));
  });
});
