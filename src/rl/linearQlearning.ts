/**
 * Linear Approximation Q-Learning Agent
 *
 * Q(s, a) = w · f(s, a) with a SINGLE weight vector shared across actions and
 * ACTION-CONDITIONED features (D8).
 *
 * The previous design was Q(s, a) = w_a · f(s): four per-action weight vectors
 * over state-only features. That is structurally incapable of navigation — every
 * action saw the *same* feature vector, so no weight setting could express
 * "move toward the pellet" or "don't step into the ghost"; the agent could only
 * learn fixed action priors (e.g. "prefer up"). Its features didn't include
 * nearestPelletDir at all. This is why the linear agent never won.
 *
 * The fix follows the classic approximate-Q Pac-Man formulation (Berkeley
 * CS188): each feature describes what the world looks like AFTER taking the
 * candidate action — does it move toward the pellet, does it put a dangerous
 * ghost within a step, does it walk into a wall. A single shared weight vector
 * means "a ghost one step away is bad" is learned once, from every action's
 * experience, instead of four times independently.
 *
 * Features (all in ~[0,1]; see extractFeatures for the pinned order):
 * - Bias term (constant 1)
 * - Moves into a wall (from wallMask)
 * - Moves toward the nearest pellet (action === BFS first-step direction)
 * - Distance to nearest pellet (state; shifts all actions equally)
 * - Dangerous ghost within 1 tile after the move
 * - Dangerous ghost within 2 tiles after the move
 * - Distance to the nearest dangerous ghost after the move
 * - Moves toward a nearby edible ghost (chase opportunity)
 * - Reverses the previous action (oscillation signal)
 *
 * D9: bootstrapping max_a Q(s', a) off the SAME weights being updated is one
 * leg of the "deadly triad" (linear FA + bootstrapping + off-policy) — every
 * update nudges the very estimate used as next update's target, so the online
 * weights can oscillate indefinitely instead of settling (observed: an 8-min
 * bench's per-checkpoint win rate swinging 0%↔27%, not a monotone climb). The
 * fix (target network, DQN-style) is a second weight vector, wTarget, that
 * only gets synced from the live w every targetSyncSteps update() calls; the
 * TD target is computed off wTarget while the live w keeps updating every
 * step. Decouples "what we're chasing" from "what we're updating" long enough
 * for the chase to actually converge. act()/peekMaxQ() always use the live w
 * — only the bootstrap target is delayed.
 */

import { type Observation, PELLET_SEARCH_RADIUS } from '../env/observation';
import { type Action, ACTIONS, DIRECTIONS, DIR_VEC, reverseAction } from '../engine/types';

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
  /**
   * D9: sync interval (in update() calls) for the target weight vector used
   * to bootstrap the TD target. undefined or 0 disables it — bootstraps off
   * the live online weights (D8 behavior). See LinearQLearningAgent header
   * for why this exists.
   */
  targetSyncSteps?: number;
}

export interface SerializedLinearPolicy {
  algorithm: 'linear-qlearning';
  mazeId: string;
  timestamp: string;
  numGhostsEncoded: number;
  version: number; // Feature schema version, bump if features change
  hyper: LinearQHyperParams;
  /**
   * v4+: a single shared weight vector, serialized as weights[0] (the outer
   * array is kept for shape compatibility with pre-v4 files, which stored one
   * vector per action; those are discarded on load via the version check).
   */
  weights: number[][];
}

export const NUM_FEATURES = 9;
// Bumped 1→2 (D5.8): all features normalized to ~[0,1].
// Bumped 2→3 (D5.9): distance features use the continuous distances carried on
// the observation instead of re-discretized buckets.
// Bumped 3→4 (D8): action-conditioned features + single shared weight vector
// (see the file header). v3 policies stored four state-only-feature vectors;
// their weights are meaningless under the new features and are discarded on load.
const FEATURE_SCHEMA_VERSION = 4;

// D5.8: normalization constants. Linear FA + bootstrapping + off-policy is the
// "deadly triad" — it has no convergence guarantee, and a mix of raw-magnitude
// features with normalized ones lets the big features dominate updates so a
// stable tabular α can diverge here. Keeping every feature in ~[0,1] bounds each
// gradient term by α·tdError, the standard precondition for stable linear TD.
// PELLET_DIST_MAX = radius+1 so the "no pellet in radius" sentinel maps to 1.0.
const PELLET_DIST_MAX = PELLET_SEARCH_RADIUS + 1;
const GHOST_DIST_MAX = 20;  // distances at/over this (incl. absent = ∞) clamp to 1.0
// Edible ghosts beyond this Manhattan distance aren't worth steering toward —
// the frightened timer (default 20 steps) would expire before we got there.
const EDIBLE_CHASE_RADIUS = 8;

// wallMask bit index per action. The mask is built in encodeObservation's CARD
// order (N/E/S/W → bits 0-3) while actions are DIRECTIONS order (up/down/left/
// right), so the mapping is not the identity: up→bit0, down→bit2, left→bit3,
// right→bit1.
const WALL_BIT_FOR_ACTION = [0, 2, 3, 1];

/**
 * Extract the action-conditioned feature vector f(s, a). All features are in
 * ~[0,1] (the bias is exactly 1) so no single feature dominates the update.
 *
 * Ghost "after the move" distances are computed from nearestGhostRel: if the
 * move is blocked by a wall the env keeps Pac-Man in place, so the post-move
 * offsets equal the current ones in that case.
 */
export function extractFeatures(obs: Observation, action: Action): Float32Array {
  const features = new Float32Array(NUM_FEATURES);

  // 0: Bias term
  features[0] = 1.0;

  // 1: Moves into a wall (the env turns this into a no-op under 'stay').
  const blocked = (obs.wallMask & (1 << WALL_BIT_FOR_ACTION[action])) !== 0;
  features[1] = blocked ? 1.0 : 0.0;

  // 2: Moves toward the nearest pellet (BFS first-step direction; the v9 key
  // aligned nearestPelletDir with the action space, so equality is the test).
  features[2] = action === obs.nearestPelletDir ? 1.0 : 0.0;

  // 3: Distance to the nearest pellet (state feature — identical for all
  // actions, so it never changes the argmax; it carries the value baseline).
  features[3] = Math.min(obs.nearestPelletDist, PELLET_DIST_MAX) / PELLET_DIST_MAX;

  // Post-move ghost geometry. Pac-Man's move shifts every ghost offset by
  // −DIR_VEC[action]; a wall-blocked move leaves the offsets unchanged.
  const v = DIR_VEC[DIRECTIONS[action]];
  let dangerDistAfter = Infinity;
  let edibleApproach = 0;
  for (const slot of obs.nearestGhostRel) {
    if (!slot) continue;
    const distNow = Math.abs(slot.dx) + Math.abs(slot.dy);
    const distAfter = blocked
      ? distNow
      : Math.abs(slot.dx - v.x) + Math.abs(slot.dy - v.y);
    if (slot.edible) {
      if (distNow <= EDIBLE_CHASE_RADIUS && distAfter < distNow) edibleApproach = 1;
    } else if (distAfter < dangerDistAfter) {
      dangerDistAfter = distAfter;
    }
  }

  // 4: Dangerous ghost within 1 tile after the move — the "this step can kill
  // me next tick" indicator (distAfter 0 also covers stepping onto the ghost).
  features[4] = dangerDistAfter <= 1 ? 1.0 : 0.0;

  // 5: Dangerous ghost within 2 tiles after the move (the ghost moves too).
  features[5] = dangerDistAfter <= 2 ? 1.0 : 0.0;

  // 6: Distance to the nearest dangerous ghost after the move (absent → 1.0,
  // i.e. maximally safe).
  features[6] = Math.min(dangerDistAfter, GHOST_DIST_MAX) / GHOST_DIST_MAX;

  // 7: Moves toward a nearby edible ghost (chase opportunity).
  features[7] = edibleApproach;

  // 8: Reverses the previous action (soft oscillation signal, mirrors the
  // env's reversePenalty).
  features[8] = obs.lastAction >= 0 && action === reverseAction(obs.lastAction as Action) ? 1.0 : 0.0;

  return features;
}

export class LinearQLearningAgent {
  /**
   * The shared weight vector (D8): Q(s, a) = w · f(s, a). One vector for all
   * four actions — action differences come from the action-conditioned
   * features, so every experience improves every action's estimate.
   */
  readonly w = new Float32Array(NUM_FEATURES);
  // D9: target network for the TD bootstrap — see the file header. Starts
  // equal to w (zero-init) so the first targetSyncSteps updates behave
  // identically to no target network.
  private readonly wTarget = new Float32Array(NUM_FEATURES);
  private stepsSinceSync = 0;

  hyper: LinearQHyperParams;
  loadedNumGhosts: number | null = null;
  trainedNumGhosts: number | null = null;

  constructor(hyper: LinearQHyperParams) {
    this.hyper = { ...hyper };
    // D5.1: weights start at zero (Float32Array is zero-filled), for full
    // reproducibility. Linear Q-learning needs no symmetry-breaking — greedy
    // ties are broken by the seeded RNG in act() — so a Math.random() init
    // would only make same-seed training runs diverge.
  }

  /**
   * Max Q-value over the four actions for the current observation. Backs the
   * Q-value overlay, mirroring the tabular agent's peekMaxQ (D5.9). Unlike the
   * tabular version this never returns null for an "unseen" state — linear FA
   * generalizes, so every observation has a defined value (0 at zero-init).
   */
  peekMaxQ(obs: Observation): number | null {
    let mx = -Infinity;
    for (const a of ACTIONS) {
      const q = this.qValue(obs, a);
      if (q > mx) mx = q;
    }
    return Number.isFinite(mx) ? mx : null;
  }

  /**
   * Compute Q(s, a) = w · f(s, a) against the live online weights.
   */
  private qValue(obs: Observation, action: Action): number {
    return this.qValueWith(this.w, obs, action);
  }

  private qValueWith(weights: Float32Array, obs: Observation, action: Action): number {
    const features = extractFeatures(obs, action);
    let q = 0;
    for (let i = 0; i < NUM_FEATURES; i++) {
      q += weights[i] * features[i];
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

    // Greedy: pick the action with the highest Q-value over its OWN features.
    const qByAction = new Map<number, number>();
    let bestValue = -Infinity;
    for (const a of legalActions) {
      const q = this.qValue(obs, a);
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
    const features = extractFeatures(obs, action);
    const currentQ = this.qValue(obs, action);

    // D9: bootstrap off the target weights (frozen between syncs) when
    // enabled, else off the live weights (D8 behavior, targetSyncSteps unset).
    const targetSyncSteps = this.hyper.targetSyncSteps ?? 0;
    const bootstrapWeights = targetSyncSteps > 0 ? this.wTarget : this.w;

    // Compute best next Q-value over the next state's own per-action features.
    let bestNextQ = 0;
    if (!done && nextLegalActions.length > 0) {
      let bestValue = -Infinity;
      for (const a of nextLegalActions) {
        bestValue = Math.max(bestValue, this.qValueWith(bootstrapWeights, nextObs, a));
      }
      bestNextQ = bestValue;
    }

    // TD target: r + γ · max_a Q(s', a)
    const target = reward + (done ? 0 : this.hyper.gamma * bestNextQ);
    const tdError = target - currentQ;

    // Update the shared weights: w := w + α · (δ · f(s,a) − λ · w)
    // D5.2: the L2 decay is scaled by α (standard SGD weight decay). Inert at
    // the default λ=0; matters once a user sets λ.
    const alpha = this.hyper.alpha;
    const lambda = this.hyper.lambda ?? 0;
    for (let i = 0; i < NUM_FEATURES; i++) {
      this.w[i] = this.w[i] + alpha * (tdError * features[i] - lambda * this.w[i]);
    }

    // D9: periodically snap the target to the (now-updated) live weights.
    if (targetSyncSteps > 0) {
      this.stepsSinceSync++;
      if (this.stepsSinceSync >= targetSyncSteps) {
        this.wTarget.set(this.w);
        this.stepsSinceSync = 0;
      }
    }
  }

  endEpisode(): void {
    this.hyper.epsilon = Math.max(this.hyper.epsilonMin, this.hyper.epsilon * this.hyper.epsilonDecay);
  }

  reset(): void {
    // D5.1: zero-init (see constructor) for deterministic restarts.
    this.w.fill(0);
    this.wTarget.fill(0);
    this.stepsSinceSync = 0;
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
      weights: [Array.from(this.w)],
      hyper: this.hyper,
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

    // Load the shared weight vector (v4+ stores it as weights[0]).
    this.trainedNumGhosts = data.numGhostsEncoded ?? null;
    if (data.weights?.[0]?.length === NUM_FEATURES) {
      this.w.set(data.weights[0]);
    }
    // D9: the target isn't serialized — resync it to the loaded weights so
    // resumed training starts from "just synced" instead of a stale/zero
    // target fighting the freshly-loaded online weights.
    this.wTarget.set(this.w);
    this.stepsSinceSync = 0;
  }
}
