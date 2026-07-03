import { observationKey, observationKeyToString, stringToObservationKey, OBSERVATION_KEY_VERSION, type Observation } from '../env/observation';
import { type Action, ACTIONS } from '../engine/types';

/**
 * How greedy `act()` breaks ties between equal-max Q-values (T4).
 *   'random'  — uniform over tied actions (default; training-baseline-preserving).
 *   'visits'  — deterministic: most-visited tied action, then lowest index.
 *   'pellet'  — deterministic: the tied action toward nearestPelletDir if any,
 *               else most-visited, then lowest index.
 *
 * Optimistic init leaves unvisited slots at 50, so early/aliased states have
 * many ties; under ε=0 eval 'random' degrades the greedy policy to a partial
 * random walk. The deterministic modes give eval a sensible, RNG-free default.
 */
export type GreedyTieBreak = 'random' | 'visits' | 'pellet';

export interface QHyperParams {
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  /**
   * Initial Q-value for unseen state-actions. Optimistic init (a value larger
   * than typical observed returns) drives systematic exploration: any unvisited
   * action looks more attractive than a visited-but-mediocre one, so the agent
   * walks toward novelty until each state-action has been updated downward.
   *
   * Default: 50 — chosen as a fraction of the typical "good" episode return
   * (~600) so that updates pull values down within a few visits while still
   * out-competing confirmed bad actions (which fall to ≈ deathPenalty).
   */
  optimisticInit?: number;
  /**
   * State-conditional exploration floor for endgame states (Priority 3b).
   *
   * When the observation's `pelletsRemainingBucket` is ≤ endgameBucketThreshold,
   * ε is clamped UP to at least `endgameEpsilon` (regardless of the decayed
   * value). This concentrates exploration in late-game states where the agent
   * has historically had no learned policy. Most exploration happens early in
   * training; with this clamp, the agent keeps exploring endgame states even
   * after the global ε has decayed to its floor.
   *
   * Defaults: undefined / 0 = disabled (use plain ε). Suggested live values:
   *   endgameEpsilon: 0.4, endgameBucketThreshold: 1.
   */
  endgameEpsilon?: number;
  endgameBucketThreshold?: number;
  /**
   * Second-stage decay for `epsilonMin` itself (root cause #3, 2026-07-01
   * win-rate investigation): a fixed 0.20 floor means the agent explores
   * randomly on 20% of steps *forever*, even after tens of millions of
   * episodes — every rare, hard-won trajectory into a near-winning endgame
   * state has a standing 20% chance of being knocked off-policy by a random
   * move before it can be reinforced. Finding #1 (test_history.md) showed a
   * *high* floor is necessary early (removing it regresses to 0% wins), so
   * this doesn't lower epsilonMin outright — it only starts shrinking it,
   * multiplicatively, once ε has actually decayed down to the floor (i.e.
   * after the bulk of state-space discovery has already happened).
   *
   * undefined or 1 = disabled (epsilonMin stays fixed forever — today's
   * behavior, the safe default per the roadmap's flag-gating convention).
   */
  epsilonMinDecay?: number;
  /** Floor for the epsilonMin decay above. Defaults to epsilonMin itself
   *  (i.e. no-op) if epsilonMinDecay is set without an explicit floor. */
  epsilonMinFloor?: number;
}

export interface SerializedPolicy {
  algorithm: 'qlearning';
  mazeId: string;
  timestamp: string;
  /** Number of ghosts the env had during training. */
  numGhostsEncoded: number;
  /** observationKey() layout version. Policies with a different version have
   *  incompatible keys and their Q-tables are discarded on load. */
  observationKeyVersion: number;
  hyper: QHyperParams;
  qTable: Record<string, number[]>; // Serialized with string keys for readability
  /**
   * Per-slot visit counts parallel to qTable. visitTable[key][a] === 0 means
   * action `a` from `key` was never updated and its Q-value is still at
   * optimisticInit. Used by federated merge to weight learned slots and
   * ignore untouched ones (mixing init values with learned ones biases the
   * merge toward the prior). Optional for backward compatibility — legacy
   * policies without visitTable fall back to "skip values == optimisticInit".
   */
  visitTable?: Record<string, number[]>;
}

export class QLearningAgent {
  readonly q = new Map<number, Float32Array>();
  /** Per-slot update counters parallel to q. Slot is 0 ⇒ Q[slot] is still at
   *  optimisticInit (never updated). See SerializedPolicy.visitTable. */
  readonly visits = new Map<number, Uint32Array>();
  hyper: QHyperParams;
  /** Set by load(). Reflects the numGhostsEncoded from the last loaded policy. */
  loadedNumGhosts: number | null = null;
  /**
   * The numGhosts value the Q-table was actually trained against. Pinned
   * on first update() after a load/reset (via setTrainedNumGhosts from the
   * UI or the training-start hook). Used by serialize() so saved policies
   * carry the truthful trained-with value even if the UI's params.numGhosts
   * drifted after training. Reset to null by reset().
   */
  trainedNumGhosts: number | null = null;

  constructor(hyper: QHyperParams) {
    this.hyper = { ...hyper };
  }

  private values(state: number): Float32Array {
    const existing = this.q.get(state);
    if (existing) return existing;
    // Optimistic init: untried actions look attractive, driving the agent to
    // visit them at least once before settling on the greedy choice. The
    // previous pessimistic init (-1) caused premature commitment to the first
    // positively-rewarded action in each state and contributed to 0% win rate.
    const init = this.hyper.optimisticInit ?? 50;
    const arr = new Float32Array([init, init, init, init]);
    this.q.set(state, arr);
    this.visits.set(state, new Uint32Array(4));
    return arr;
  }

  /**
   * Max Q-value over the four actions for an already-visited state, or null if
   * the state was never seen (still at optimisticInit). Read-only: does NOT
   * insert a phantom entry like values() does. Used by the Q-value overlay.
   */
  peekMaxQ(obs: Observation): number | null {
    const vals = this.q.get(observationKey(obs));
    if (!vals) return null;
    let mx = -Infinity;
    for (let a = 0; a < 4; a += 1) if (vals[a] > mx) mx = vals[a];
    return Number.isFinite(mx) ? mx : null;
  }

  act(obs: Observation, legalActions: Action[], random: () => number, tieBreak: GreedyTieBreak = 'random'): Action {
    if (legalActions.length === 0) return ACTIONS[0];

    // State-conditional ε floor for endgame states. When obs indicates we're
    // in the late-game pellet buckets, use max(decayed ε, endgameEpsilon) so
    // the agent keeps exploring those rarely-visited states even after global
    // ε has decayed.
    let effectiveEps = this.hyper.epsilon;
    const endgameEps = this.hyper.endgameEpsilon ?? 0;
    const threshold = this.hyper.endgameBucketThreshold ?? 0;
    if (endgameEps > effectiveEps && obs.pelletsRemainingBucket <= threshold) {
      effectiveEps = endgameEps;
    }

    if (effectiveEps > 0 && random() < effectiveEps) {
      return legalActions[Math.floor(random() * legalActions.length)] ?? legalActions[0];
    }

    const state = observationKey(obs);
    const vals = this.values(state);
    let bestValue = -Infinity;
    for (const a of legalActions) if (vals[a] > bestValue) bestValue = vals[a];
    const bestActions = legalActions.filter((a) => vals[a] === bestValue);
    if (bestActions.length === 1) return bestActions[0];
    return this.breakTie(bestActions, obs, state, random, tieBreak);
  }

  /** Resolve a greedy tie among equal-max actions per the requested strategy. */
  private breakTie(
    cands: Action[],
    obs: Observation,
    state: number,
    random: () => number,
    mode: GreedyTieBreak,
  ): Action {
    if (mode === 'random') {
      return cands[Math.floor(random() * cands.length)] ?? cands[0];
    }
    // 'pellet': steer toward the BFS pellet direction when it's a tied option.
    if (mode === 'pellet') {
      const d = obs.nearestPelletDir;
      if (d >= 0 && d <= 3 && cands.includes(d as Action)) return d as Action;
    }
    // 'pellet' fallthrough + 'visits': prefer the most-updated action (a learned
    // slot beats one still sitting at optimisticInit), breaking remaining ties by
    // lowest action index so eval stays fully deterministic (no RNG draw).
    const v = this.visits.get(state);
    if (!v) return cands[0];
    let best = cands[0];
    for (const a of cands) if (v[a] > v[best]) best = a;
    return best;
  }

  update(
    obs: Observation,
    action: Action,
    reward: number,
    nextObs: Observation,
    done: boolean,
    nextLegalActions: Action[] = [...ACTIONS],
  ): void {
    const s = observationKey(obs);
    const qS = this.values(s);
    // Read next-state values without inserting a phantom entry for terminal states.
    // Use optimisticInit as the fallback so a missing next-state looks as attractive
    // as any other unvisited state — consistent with values() above.
    const init = this.hyper.optimisticInit ?? 50;
    let bestNext = 0;
    if (!done && nextLegalActions.length > 0) {
      const qN = this.q.get(observationKey(nextObs));
      bestNext = qN ? Math.max(...nextLegalActions.map((a) => qN[a])) : init;
    }
    const target = reward + (done ? 0 : this.hyper.gamma * bestNext);
    qS[action] = qS[action] + this.hyper.alpha * (target - qS[action]);
    const v = this.visits.get(s);
    if (v && v[action] < 0xffffffff) v[action] += 1;
  }

  endEpisode(): void {
    this.hyper.epsilon = Math.max(this.hyper.epsilonMin, this.hyper.epsilon * this.hyper.epsilonDecay);

    // Second-stage floor decay (see epsilonMinDecay doc). Only engages once ε
    // has actually reached the floor, so early training keeps its full
    // epsilonMin exploration rate exactly as before.
    const epsilonMinDecay = this.hyper.epsilonMinDecay ?? 1;
    if (epsilonMinDecay < 1 && this.hyper.epsilon <= this.hyper.epsilonMin) {
      const epsilonMinFloor = this.hyper.epsilonMinFloor ?? this.hyper.epsilonMin;
      this.hyper.epsilonMin = Math.max(epsilonMinFloor, this.hyper.epsilonMin * epsilonMinDecay);
    }
  }

  reset(): void {
    this.q.clear();
    this.visits.clear();
    this.trainedNumGhosts = null;
    this.loadedNumGhosts = null;
  }

  /** Pin the numGhosts the Q-table is being trained against. Idempotent if
   *  called with the same value; logs a warning if a different value is
   *  attempted (caller should reset the Q-table first). */
  setTrainedNumGhosts(n: number): void {
    if (this.trainedNumGhosts !== null && this.trainedNumGhosts !== n) {
      console.warn(
        `[QLearningAgent] setTrainedNumGhosts(${n}) ignored: Q-table already pinned to ${this.trainedNumGhosts}. ` +
        'Call reset() first to retrain against a different ghost count.',
      );
      return;
    }
    this.trainedNumGhosts = n;
  }

  serialize(mazeId: string, numGhostsEncoded: number): SerializedPolicy {
    const qTable: Record<string, number[]> = {};
    const visitTable: Record<string, number[]> = {};
    for (const [key, values] of this.q.entries()) {
      const keyStr = observationKeyToString(key);
      qTable[keyStr] = Array.from(values);
      const v = this.visits.get(key);
      visitTable[keyStr] = v ? Array.from(v) : [0, 0, 0, 0];
    }
    return {
      algorithm: 'qlearning',
      mazeId,
      timestamp: new Date().toISOString(),
      // Prefer the pinned trained-with value over the caller's argument so
      // a save after the user fiddles with the numGhosts input records the
      // value the Q-table actually trained against.
      numGhostsEncoded: this.trainedNumGhosts ?? numGhostsEncoded,
      observationKeyVersion: OBSERVATION_KEY_VERSION,
      hyper: this.hyper,
      qTable,
      visitTable,
    };
  }

  load(data: SerializedPolicy, currentNumGhosts?: number): void {
    // Preserve exploration hyperparams across load(). A serialized policy
    // carries its end-of-training (decayed) ε; copying it wholesale would
    // pin a freshly-warmstarted worker at near-greedy and silently kill
    // federated exploration. Keep the live agent's epsilon* and
    // endgameEpsilon* fields; take everything else from disk.
    const liveExploration = {
      epsilon: this.hyper.epsilon,
      epsilonDecay: this.hyper.epsilonDecay,
      epsilonMin: this.hyper.epsilonMin,
      endgameEpsilon: this.hyper.endgameEpsilon,
      endgameBucketThreshold: this.hyper.endgameBucketThreshold,
      epsilonMinDecay: this.hyper.epsilonMinDecay,
      epsilonMinFloor: this.hyper.epsilonMinFloor,
    };
    this.hyper = { ...data.hyper, ...liveExploration };
    this.loadedNumGhosts = data.numGhostsEncoded ?? null;

    const policyVersion = data.observationKeyVersion ?? 1;
    if (policyVersion !== OBSERVATION_KEY_VERSION) {
      console.warn(
        `[QLearningAgent] policy key version ${policyVersion} != current ${OBSERVATION_KEY_VERSION}. ` +
        'Q-table discarded — training from scratch with the updated encoder.',
      );
      this.q.clear();
      this.visits.clear();
      this.trainedNumGhosts = null;
      return;
    }

    if (
      currentNumGhosts !== undefined &&
      data.numGhostsEncoded !== undefined &&
      data.numGhostsEncoded !== currentNumGhosts
    ) {
      // Don't half-load: coincidentally-matching keys across different
      // ghost counts encode different geometric situations, so the loaded
      // Q-values would silently alias unrelated states. Clean miss is
      // strictly better than stale contamination.
      console.warn(
        `[QLearningAgent] numGhosts mismatch: policy was trained with ${data.numGhostsEncoded} ghost(s) ` +
        `but env has ${currentNumGhosts}. Q-table discarded — training from scratch.`,
      );
      this.q.clear();
      this.visits.clear();
      this.trainedNumGhosts = null;
      return;
    }

    // Q-table accepted: pin trainedNumGhosts to the loaded value so a
    // later serialize() records the truthful trained-with count.
    this.trainedNumGhosts = data.numGhostsEncoded ?? null;
    this.q.clear();
    this.visits.clear();
    // D5.10: decode each serialized key via the shared inverse in observation.ts
    // (same base constants as observationKey) instead of re-packing here with
    // hardcoded bases. Skips entries whose version/format doesn't match.
    for (const [keyStr, values] of Object.entries(data.qTable)) {
      const key = stringToObservationKey(keyStr);
      if (key === null) continue;
      this.q.set(key, new Float32Array(values));
      const v = data.visitTable?.[keyStr];
      this.visits.set(key, v ? new Uint32Array(v) : new Uint32Array(4));
    }
  }
}
