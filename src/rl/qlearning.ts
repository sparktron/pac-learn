import { observationKey, observationKeyToString, type Observation } from '../env/observation';

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
  hyper: QHyperParams;
  qTable: Record<string, number[]>; // Serialized with string keys for readability
}

export class QLearningAgent {
  readonly q = new Map<number, Float32Array>();
  hyper: QHyperParams;

  constructor(hyper: QHyperParams) {
    this.hyper = { ...hyper };
  }

  private values(state: number): Float32Array {
    const existing = this.q.get(state);
    if (existing) return existing;
    const arr = new Float32Array([0, 0, 0, 0]);
    this.q.set(state, arr);
    return arr;
  }

  act(obs: Observation, legalActions: number[], random: () => number): number {
    const state = observationKey(obs);
    if (random() < this.hyper.epsilon) return legalActions[Math.floor(random() * legalActions.length)] ?? 0;
    const vals = this.values(state);
    return legalActions.reduce((best, a) => (vals[a] > vals[best] ? a : best), legalActions[0] ?? 0);
  }

  update(obs: Observation, action: number, reward: number, nextObs: Observation, done: boolean): void {
    const s = observationKey(obs);
    const ns = observationKey(nextObs);
    const qS = this.values(s);
    const qN = this.values(ns);
    const target = reward + (done ? 0 : this.hyper.gamma * Math.max(...qN));
    qS[action] = qS[action] + this.hyper.alpha * (target - qS[action]);
  }

  endEpisode(): void {
    this.hyper.epsilon = Math.max(this.hyper.epsilonMin, this.hyper.epsilon * this.hyper.epsilonDecay);
  }

  reset(): void {
    this.q.clear();
  }

  serialize(mazeId: string): SerializedPolicy {
    const qTable: Record<string, number[]> = {};
    for (const [key, values] of this.q.entries()) {
      qTable[observationKeyToString(key)] = Array.from(values);
    }
    return {
      algorithm: 'qlearning',
      mazeId,
      timestamp: new Date().toISOString(),
      hyper: this.hyper,
      qTable,
    };
  }

  load(data: SerializedPolicy): void {
    this.hyper = { ...data.hyper };
    this.q.clear();
    Object.entries(data.qTable).forEach(([keyStr, values]) => {
      // Parse string key format: wallMask:pelletDir:dx1,dy1:dx2,dy2:...
      const parts = keyStr.split(':');
      const wallMask = parseInt(parts[0], 10);
      const pelletDir = parseInt(parts[1], 10);

      let key = wallMask;
      let place = 2 ** 25;
      key += pelletDir * place;
      place *= 4;

      // Parse ghost offsets (up to 4 ghosts)
      for (let i = 0; i < 4; i++) {
        const [dxStr = '0', dyStr = '0'] = (parts[2 + i] ?? '0,0').split(',');
        const dx = Math.max(0, Math.min(6, parseInt(dxStr, 10) + 3));
        const dy = Math.max(0, Math.min(6, parseInt(dyStr, 10) + 3));
        key += (dx * 7 + dy) * place;
        place *= 49;
      }

      this.q.set(key, new Float32Array(values));
    });
  }
}
