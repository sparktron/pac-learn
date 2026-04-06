import { describe, expect, test } from 'vitest';
import { createDefaultEnv } from './environment';

describe('environment scoring', () => {
  test('collecting a power pellet updates score and reward consistently', () => {
    const env = createDefaultEnv();
    env.setParams({
      pelletDensity: 0,
      enablePowerPellets: true,
      reward: { ...env.params.reward, powerPelletReward: 37 },
    });
    env.reset(5);

    const beforeScore = env.getPacmen()[0].score;
    const result = env.step(0); // up into wall from start, remains on power pellet tile

    const expectedReward = 37 + env.params.reward.stepPenalty + env.params.reward.survivalReward;
    expect(result.reward).toBeCloseTo(expectedReward);
    expect(env.getPacmen()[0].score - beforeScore).toBe(37);
    expect(result.info.score).toBe(env.getPacmen()[0].score);
  });
});
