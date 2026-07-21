import type { LinearQHyperParams } from './linearQlearning';
import type { QHyperParams } from './qlearning';

/**
 * Production training defaults shared by the GUI and headless bench.
 *
 * Keep these centralized: training results are only comparable when both
 * surfaces instantiate each algorithm with the same baseline configuration.
 * CLI/UI edits may override these values after construction.
 */
export const TABULAR_HYPER_DEFAULTS = {
  alpha: 0.1,
  gamma: 0.99,
  epsilon: 0.5,
  epsilonDecay: 0.999997,
  epsilonMin: 0.20,
  endgameEpsilon: 0.25,
  endgameBucketThreshold: 1,
} satisfies QHyperParams;

export const LINEAR_HYPER_DEFAULTS = {
  alpha: 0.02,
  gamma: 0.99,
  epsilon: 0.3,
  epsilonDecay: 0.9995,
  epsilonMin: 0.05,
  targetSyncSteps: 2000,
} satisfies LinearQHyperParams;
