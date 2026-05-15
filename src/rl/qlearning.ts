import { observationKey, observationKeyToString, OBSERVATION_KEY_VERSION, type Observation } from '../env/observation';

export interface QHyperParams {
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
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
    // Pessimistic init: -1 ensures any positively-rewarded explored action beats
    // an untried one, while confirmed-bad actions (-100s) still lose to unexplored.
    const arr = new Float32Array([-1, -1, -1, -1]);
    this.q.set(state, arr);
    return arr;
  }

  act(obs: Observation, legalActions: number[], random: () => number): number {
    if (legalActions.length === 0) return 0;

    if (random() < this.hyper.epsilon) {
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
    let bestNext = 0;
    if (!done && nextLegalActions.length > 0) {
      const qN = this.q.get(observationKey(nextObs));
      bestNext = qN ? Math.max(...nextLegalActions.map((a) => qN[a])) : -1;
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
    // v3 key string format: "v3:wallMask:pelletDir:edibleBucket:gc0:gc1"
    for (const [keyStr, values] of Object.entries(data.qTable)) {
      const parts = keyStr.split(':');
      if (parts[0] !== 'v3' || parts.length !== 6) continue;
      const wallMask     = parseInt(parts[1], 10);
      const pelletDir    = parseInt(parts[2], 10);
      const edibleBucket = parseInt(parts[3], 10);
      const gc0          = parseInt(parts[4], 10);
      const gc1          = parseInt(parts[5], 10);

      let key = wallMask;
      let place = 16; // WALL_MASK_BASE: 4-bit cardinal mask
      key += pelletDir    * place; place *= 5;
      key += edibleBucket * place; place *= 3;
      key += gc0          * place; place *= 10;
      key += gc1          * place;

      this.q.set(key, new Float32Array(values));
    }
  }
}
