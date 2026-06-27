/**
 * Linear Approximation Q-Learning Agent
 *
 * Instead of a tabular Q(s,a) lookup, uses: Q(s,a) = w_a · f(s)
 * where w_a is a weight vector for action a, and f(s) is a feature vector.
 *
 * Features are derived from the Observation and include:
 * - Bias term (constant 1)
 * - Distance to nearest pellet (raw pixels)
 * - Tunnel-aware distance to nearest ghost
 * - Tunnel-aware distance to second-nearest ghost
 * - Count of ghosts within 1 step (0, 1, or 2)
 * - Power pellets available (0 or 1)
 * - Pellet progression (0-4 bucket as float)
 * - Wall configuration (4-bit mask as float)
 * - Last action taken (0-3 or -1 for start)
 *
 * This avoids the curse of dimensionality: instead of learning 120k+ discrete
 * states, we learn ~9 continuous weights per action, which converges 2-3× faster
 * and handles generalization across similar states much better.
 */

import { type Observation, PELLET_SEARCH_RADIUS } from '../env/observation';
import { type Action, ACTIONS } from '../engine/types';

export interface LinearQHyperParams {
  alpha: number;        // Learning rate for weight updates
  gamma: number;        // Discount factor
  epsilon: number;      // Current exploration rate
  epsilonDecay: number; // Per-episode decay multiplier
  epsilonMin: number;   // Minimum epsilon floor
  endgameEpsilon?: number;
  endgameBucketThreshold?: number;
  /**
   * L2 regularization parameter. Prevents weights from growing unbounded.
   * lambda * ||w||² is added to the loss. Common values: 0 (off), 0.0001.
   */
  lambda?: number;
}

export interface SerializedLinearPolicy {
  algorithm: 'linear-qlearning';
  mazeId: string;
  timestamp: string;
  numGhostsEncoded: number;
  version: number; // Feature schema version, bump if features change
  hyper: LinearQHyperParams;
  /**
   * Weight vectors: one per action (0-3 for up/down/left/right, the DIRECTIONS
   * action-space order). Each is a Float32Array of length NUM_FEATURES.
   */
  weights: number[][];
}

const NUM_FEATURES = 9; // [bias, dist_pellet, dist_ghost_1, dist_ghost_2, ghosts_nearby, power_available, pellet_bucket, wall_mask, last_action]
// Bumped 1→2 (D5.8): all features normalized to ~[0,1].
// Bumped 2→3 (D5.9): the distance features now use the *continuous* distances
// (BFS depth for pellets, tunnel-aware Manhattan for ghosts) carried on the
// observation, instead of re-deriving 0.5/1.0 and 1/3/8 from the already-
// discretized buckets. v2 policies are discarded on load (their weights were
// fit against the coarse 2-/3-valued features and don't transfer).
const FEATURE_SCHEMA_VERSION = 3;

// D5.8: normalization constants. Linear FA + bootstrapping + off-policy is the
// "deadly triad" — it has no convergence guarantee, and a mix of raw-magnitude
// features with normalized ones lets the big features dominate updates so a
// stable tabular α can diverge here. Keeping every feature in ~[0,1] bounds each
// gradient term by α·tdError, the standard precondition for stable linear TD.
// PELLET_DIST_MAX = radius+1 so the "no pellet in radius" sentinel maps to 1.0.
const PELLET_DIST_MAX = PELLET_SEARCH_RADIUS + 1;
const GHOST_DIST_MAX = 20;  // distances at/over this (incl. absent = ∞) clamp to 1.0

/**
 * Extract numerical features from an observation. All features are normalized
 * to ~[0,1] (the bias is exactly 1) so no single feature dominates the update.
 * Returns a vector of length NUM_FEATURES.
 */
function extractFeatures(obs: Observation): Float32Array {
  const features = new Float32Array(NUM_FEATURES);
  let idx = 0;

  // 0: Bias term
  features[idx++] = 1.0;

  // 1: Continuous distance to the nearest pellet (D5.9). nearestPelletDist is the
  // BFS depth 1..radius, or radius+1 when none is reachable → normalizes to 1.0.
  features[idx++] = Math.min(obs.nearestPelletDist, PELLET_DIST_MAX) / PELLET_DIST_MAX;

  // 2: Continuous distance to the nearest ghost (D5.9), tunnel-aware Manhattan.
  // Absent slot is +Infinity → clamps to GHOST_DIST_MAX → 1.0 (farthest).
  const distGhost1 = Math.min(obs.nearestGhostDists[0], GHOST_DIST_MAX);
  features[idx++] = distGhost1 / GHOST_DIST_MAX;

  // 3: Continuous distance to the second-nearest ghost (D5.9).
  const distGhost2 = Math.min(obs.nearestGhostDists[1], GHOST_DIST_MAX);
  features[idx++] = distGhost2 / GHOST_DIST_MAX;

  // 4: Count of ghosts within 1 step (dist ≤ 1), normalized to [0,1].
  let ghostsNearby = 0;
  if (obs.nearestGhostDists[0] <= 1) ghostsNearby++;
  if (obs.nearestGhostDists[1] <= 1) ghostsNearby++;
  features[idx++] = ghostsNearby / 2.0;

  // 5: Power pellets available (0 or 1)
  const powerAvailable = obs.powerPelletsLeftBucket > 0 ? 1.0 : 0.0;
  features[idx++] = powerAvailable;

  // 6: Pellet progression bucket (0-4) normalized to 0-1
  features[idx++] = obs.pelletsRemainingBucket / 4.0;

  // 7: Wall mask as float (0-15 → 0-1)
  features[idx++] = obs.wallMask / 15.0;

  // 8: Last action (shift -1→0, 0-3→1-4, then normalize)
  const actionNorm = (obs.lastAction + 1) / 4.0;
  features[idx++] = actionNorm;

  // D5.6: dropped the per-call `console.assert(idx === NUM_FEATURES)` — it ran on
  // every act()/update() (millions of times per run). The feature count/order is
  // pinned by linearQlearning.test.ts instead.
  return features;
}

export class LinearQLearningAgent {
  /**
   * Weight vectors: one per action (0-3 for up/down/left/right, the DIRECTIONS
   * action-space order). weights[a][f] = weight for feature f in action a.
   */
  readonly weights: Float32Array[] = [
    new Float32Array(NUM_FEATURES),
    new Float32Array(NUM_FEATURES),
    new Float32Array(NUM_FEATURES),
    new Float32Array(NUM_FEATURES),
  ];

  hyper: LinearQHyperParams;
  loadedNumGhosts: number | null = null;
  trainedNumGhosts: number | null = null;

  constructor(hyper: LinearQHyperParams) {
    this.hyper = { ...hyper };
    // D5.1: weights start at zero (Float32Array is zero-filled), for full
    // reproducibility. Linear Q-learning needs no symmetry-breaking — each action
    // has its own weight vector and greedy ties are broken by the seeded RNG in
    // act() — so the previous Math.random() init only made same-seed training
    // runs diverge, breaking the determinism the rest of the system relies on
    // (and the bench's algorithm-compare / hyperparam-sweep reproducibility).
  }

  /**
   * Max Q-value over the four actions for the current observation. Backs the
   * Q-value overlay, mirroring the tabular agent's peekMaxQ (D5.9). Unlike the
   * tabular version this never returns null for an "unseen" state — linear FA
   * generalizes, so every observation has a defined value (0 at zero-init).
   */
  peekMaxQ(obs: Observation): number | null {
    const features = extractFeatures(obs);
    let mx = -Infinity;
    for (const a of ACTIONS) {
      const q = this.qValue(features, a);
      if (q > mx) mx = q;
    }
    return Number.isFinite(mx) ? mx : null;
  }

  /**
   * Compute Q(s, a) = w_a · f(s)
   */
  private qValue(features: Float32Array, action: Action): number {
    let q = 0;
    const w = this.weights[action];
    for (let i = 0; i < NUM_FEATURES; i++) {
      q += w[i] * features[i];
    }
    return q;
  }

  // `_tieBreak` is accepted for interface parity with QLearningAgent (the
  // trainer is generic over both) but unused here: the linear agent's Q-values
  // are continuous, so exact ties between actions effectively never occur.
  act(obs: Observation, legalActions: Action[], random: () => number, _tieBreak?: 'random' | 'visits' | 'pellet'): Action {
    if (legalActions.length === 0) return ACTIONS[0];

    // State-conditional ε floor for endgame
    let effectiveEps = this.hyper.epsilon;
    const endgameEps = this.hyper.endgameEpsilon ?? 0;
    const threshold = this.hyper.endgameBucketThreshold ?? 0;
    if (endgameEps > effectiveEps && obs.pelletsRemainingBucket <= threshold) {
      effectiveEps = endgameEps;
    }

    // ε-greedy action selection
    if (effectiveEps > 0 && random() < effectiveEps) {
      return legalActions[Math.floor(random() * legalActions.length)] ?? legalActions[0];
    }

    // Greedy: pick action with highest Q-value. D5.7: compute each legal action's
    // Q once (was computed twice — max-scan then tie-filter).
    const features = extractFeatures(obs);
    const qByAction = new Map<number, number>();
    let bestValue = -Infinity;
    for (const a of legalActions) {
      const q = this.qValue(features, a);
      qByAction.set(a, q);
      if (q > bestValue) bestValue = q;
    }

    const bestActions = legalActions.filter((a) => qByAction.get(a) === bestValue);
    if (bestActions.length === 1) return bestActions[0];
    return bestActions[Math.floor(random() * bestActions.length)] ?? legalActions[0];
  }

  update(
    obs: Observation,
    action: Action,
    reward: number,
    nextObs: Observation,
    done: boolean,
    nextLegalActions: Action[] = [...ACTIONS],
  ): void {
    const features = extractFeatures(obs);
    const currentQ = this.qValue(features, action);

    // Compute best next Q-value
    let bestNextQ = 0;
    if (!done && nextLegalActions.length > 0) {
      const nextFeatures = extractFeatures(nextObs);
      let bestValue = -Infinity;
      for (const a of nextLegalActions) {
        bestValue = Math.max(bestValue, this.qValue(nextFeatures, a));
      }
      bestNextQ = bestValue;
    }

    // TD target: r + γ · max_a Q(s', a)
    const target = reward + (done ? 0 : this.hyper.gamma * bestNextQ);
    const tdError = target - currentQ;

    // Update weights: w_a := w_a + α · (δ · f(s) − λ · w_a)
    // D5.2: the L2 decay is scaled by α (standard SGD weight decay). Previously
    // the `− λ·w` term was applied raw, so weight decay ran independent of the
    // learning rate. Inert at the default λ=0; matters once a user sets λ.
    const alpha = this.hyper.alpha;
    const lambda = this.hyper.lambda ?? 0;
    const weights = this.weights[action];

    for (let i = 0; i < NUM_FEATURES; i++) {
      weights[i] = weights[i] + alpha * (tdError * features[i] - lambda * weights[i]);
    }
  }

  endEpisode(): void {
    this.hyper.epsilon = Math.max(this.hyper.epsilonMin, this.hyper.epsilon * this.hyper.epsilonDecay);
  }

  reset(): void {
    // D5.1: zero-init (see constructor) for deterministic restarts.
    for (const w of this.weights) w.fill(0);
    this.trainedNumGhosts = null;
    this.loadedNumGhosts = null;
  }

  setTrainedNumGhosts(n: number): void {
    if (this.trainedNumGhosts !== null && this.trainedNumGhosts !== n) {
      console.warn(
        `[LinearQLearningAgent] setTrainedNumGhosts(${n}) ignored: weights already pinned to ${this.trainedNumGhosts}. ` +
        'Call reset() first to retrain against a different ghost count.',
      );
      return;
    }
    this.trainedNumGhosts = n;
  }

  serialize(mazeId: string, numGhostsEncoded: number): SerializedLinearPolicy {
    return {
      algorithm: 'linear-qlearning',
      mazeId,
      timestamp: new Date().toISOString(),
      numGhostsEncoded: this.trainedNumGhosts ?? numGhostsEncoded,
      version: FEATURE_SCHEMA_VERSION,
      hyper: this.hyper,
      weights: Array.from(this.weights).map((w) => Array.from(w)),
    };
  }

  load(data: SerializedLinearPolicy, currentNumGhosts?: number): void {
    const liveExploration = {
      epsilon: this.hyper.epsilon,
      epsilonDecay: this.hyper.epsilonDecay,
      epsilonMin: this.hyper.epsilonMin,
      endgameEpsilon: this.hyper.endgameEpsilon,
      endgameBucketThreshold: this.hyper.endgameBucketThreshold,
    };
    this.hyper = { ...data.hyper, ...liveExploration };
    this.loadedNumGhosts = data.numGhostsEncoded ?? null;

    if (data.version !== FEATURE_SCHEMA_VERSION) {
      console.warn(
        `[LinearQLearningAgent] feature schema version ${data.version} != current ${FEATURE_SCHEMA_VERSION}. ` +
        'Weights discarded — training from scratch.',
      );
      this.reset();
      return;
    }

    if (
      currentNumGhosts !== undefined &&
      data.numGhostsEncoded !== undefined &&
      data.numGhostsEncoded !== currentNumGhosts
    ) {
      console.warn(
        `[LinearQLearningAgent] numGhosts mismatch: policy was trained with ${data.numGhostsEncoded} ghost(s) ` +
        `but env has ${currentNumGhosts}. Weights discarded — training from scratch.`,
      );
      this.reset();
      return;
    }

    // Load weights
    this.trainedNumGhosts = data.numGhostsEncoded ?? null;
    if (data.weights && data.weights.length === 4) {
      for (let a = 0; a < 4; a++) {
        if (data.weights[a] && data.weights[a].length === NUM_FEATURES) {
          this.weights[a].set(data.weights[a]);
        }
      }
    }
  }
}
