import { describe, expect, test } from 'vitest';
import { QLearningAgent } from './qlearning';
import type { Observation } from '../env/observation';

const obs: Observation = { pac: { x: 1, y: 1 }, ghosts: [{ x: 2, y: 2 }], wallMask: 0, nearestPelletDir: 1, ghostRel: [{ dx: 1, dy: 1 }] };

describe('qlearning', () => {
  test('updates q value', () => {
    const agent = new QLearningAgent({ alpha: 0.5, gamma: 1, epsilon: 0, epsilonDecay: 1, epsilonMin: 0 });
    agent.update(obs, 0, 10, obs, true);
    const val = [...agent.q.values()][0][0];
    expect(val).toBe(5);
  });
});
