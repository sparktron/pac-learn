import { describe, expect, test } from 'vitest';
import { LINEAR_HYPER_DEFAULTS, TABULAR_HYPER_DEFAULTS } from './hyperDefaults';

describe('shared production hyperparameter defaults', () => {
  test('keeps the tuned tabular baseline', () => {
    expect(TABULAR_HYPER_DEFAULTS).toEqual({
      alpha: 0.1,
      gamma: 0.99,
      epsilon: 0.5,
      epsilonDecay: 0.999997,
      epsilonMin: 0.20,
      endgameEpsilon: 0.25,
      endgameBucketThreshold: 1,
    });
  });

  test('keeps the tuned linear baseline separate from tabular exploration', () => {
    expect(LINEAR_HYPER_DEFAULTS).toEqual({
      alpha: 0.02,
      gamma: 0.99,
      epsilon: 0.3,
      epsilonDecay: 0.9995,
      epsilonMin: 0.05,
      targetSyncSteps: 2000,
    });
    expect(LINEAR_HYPER_DEFAULTS.epsilonMin).toBeLessThan(TABULAR_HYPER_DEFAULTS.epsilonMin);
  });
});
