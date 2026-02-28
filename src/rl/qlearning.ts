import { observationKey, type Observation } from '../env/observation';

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
  qTable: Record<string, number[]>;
}

export class QLearningAgent {
  readonly q = new Map<string, number[]>();
  hyper: QHyperParams;

  constructor(hyper: QHyperParams) {
    this.hyper = { ...hyper };
  }

  private values(state: string): number[] {
    const existing = this.q.get(state);
    if (existing) return existing;
    const arr = [0, 0, 0, 0];
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
    return {
      algorithm: 'qlearning',
      mazeId,
      timestamp: new Date().toISOString(),
      hyper: this.hyper,
      qTable: Object.fromEntries([...this.q.entries()]),
    };
  }

  load(data: SerializedPolicy): void {
    this.hyper = { ...data.hyper };
    this.q.clear();
    Object.entries(data.qTable).forEach(([k, v]) => this.q.set(k, v));
  }
}
