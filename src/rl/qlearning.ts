import { observationKey, observationKeyToString, OBSERVATION_KEY_VERSION, type Observation } from '../env/observation';

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
}

export class QLearningAgent {
  readonly q = new Map<number, Float32Array>();
  hyper: QHyperParams;
  /** Set by load(). Reflects the numGhostsEncoded from the last loaded policy. */
  loadedNumGhosts: number | null = null;

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
    return arr;
  }

  act(obs: Observation, legalActions: number[], random: () => number): number {
    if (legalActions.length === 0) return 0;

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

    if (random() < effectiveEps) {
      return legalActions[Math.floor(random() * legalActions.length)] ?? legalActions[0];
    }

    const vals = this.values(observationKey(obs));
    let bestValue = -Infinity;
    for (const a of legalActions) if (vals[a] > bestValue) bestValue = vals[a];
    const bestActions = legalActions.filter((a) => vals[a] === bestValue);
    if (bestActions.length === 1) return bestActions[0];
    return bestActions[Math.floor(random() * bestActions.length)] ?? legalActions[0];
  }

  update(
    obs: Observation,
    action: number,
    reward: number,
    nextObs: Observation,
    done: boolean,
    nextLegalActions: number[] = [0, 1, 2, 3],
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
  }

  endEpisode(): void {
    this.hyper.epsilon = Math.max(this.hyper.epsilonMin, this.hyper.epsilon * this.hyper.epsilonDecay);
  }

  reset(): void {
    this.q.clear();
  }

  serialize(mazeId: string, numGhostsEncoded: number): SerializedPolicy {
    const qTable: Record<string, number[]> = {};
    for (const [key, values] of this.q.entries()) {
      qTable[observationKeyToString(key)] = Array.from(values);
    }
    return {
      algorithm: 'qlearning',
      mazeId,
      timestamp: new Date().toISOString(),
      numGhostsEncoded,
      observationKeyVersion: OBSERVATION_KEY_VERSION,
      hyper: this.hyper,
      qTable,
    };
  }

  load(data: SerializedPolicy, currentNumGhosts?: number): void {
    this.hyper = { ...data.hyper };
    this.loadedNumGhosts = data.numGhostsEncoded ?? null;

    const policyVersion = data.observationKeyVersion ?? 1;
    if (policyVersion !== OBSERVATION_KEY_VERSION) {
      console.warn(
        `[QLearningAgent] policy key version ${policyVersion} != current ${OBSERVATION_KEY_VERSION}. ` +
        'Q-table discarded — training from scratch with the updated encoder.',
      );
      this.q.clear();
      return;
    }

    if (
      currentNumGhosts !== undefined &&
      data.numGhostsEncoded !== undefined &&
      data.numGhostsEncoded !== currentNumGhosts
    ) {
      console.warn(
        `[QLearningAgent] numGhosts mismatch: policy was trained with ${data.numGhostsEncoded} ghost(s) ` +
        `but env has ${currentNumGhosts}. Nearly every observation will be a Q-table miss.`,
      );
    }

    this.q.clear();
    // v8 key string format:
    //   "v8:wallMask:pelletDir:gc0:gh0:gc1:gh1:lastAction:pelletsBucket:powerBucket"
    for (const [keyStr, values] of Object.entries(data.qTable)) {
      const parts = keyStr.split(':');
      if (parts[0] !== 'v8' || parts.length !== 10) continue;
      const wallMask      = parseInt(parts[1], 10);
      const pelletDir     = parseInt(parts[2], 10);
      const gc0           = parseInt(parts[3], 10);
      const gh0           = parseInt(parts[4], 10);
      const gc1           = parseInt(parts[5], 10);
      const gh1           = parseInt(parts[6], 10);
      const lastAction    = parseInt(parts[7], 10); // raw: -1 to 3
      const pelletsBucket = parseInt(parts[8], 10); // 0-4
      const powerBucket   = parseInt(parts[9], 10); // 0-2

      let key = wallMask;
      let place = 16;  // WALL_MASK_BASE
      key += pelletDir          * place; place *= 5;   // PELLET_DIR_BASE
      key += gc0                * place; place *= 19;  // GHOST_ZONE_BASE
      key += gh0                * place; place *= 3;   // GHOST_HEADING_BASE
      key += gc1                * place; place *= 19;
      key += gh1                * place; place *= 3;
      key += (lastAction + 1)   * place; place *= 5;   // LAST_ACTION_BASE
      key += pelletsBucket      * place; place *= 5;   // PELLETS_REMAINING_BUCKET_BASE
      key += powerBucket        * place;               // POWER_PELLETS_BUCKET_BASE=3

      this.q.set(key, new Float32Array(values));
    }
  }
}
