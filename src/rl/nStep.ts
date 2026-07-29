import type { Action } from '../engine/types';
import type { Observation } from '../env/observation';

export interface NStepTransition {
  obs: Observation;
  action: Action;
  reward: number;
  nextObs: Observation;
  done: boolean;
  nextLegalActions: Action[];
}

/**
 * Delays one-step updates until an n-step return is available. A terminal
 * transition flushes every remaining prefix so short episodes still learn from
 * their final reward. The owning agent computes the algorithm-specific
 * bootstrap and applies the resulting update.
 */
export class NStepReturnBuffer {
  private pending: NStepTransition[] = [];

  constructor(private readonly apply: (transitions: readonly NStepTransition[]) => void) {}

  push(transition: NStepTransition, nStep: number): void {
    if (!Number.isInteger(nStep) || nStep < 1) {
      throw new Error(`nStep must be a positive integer; received ${nStep}`);
    }
    this.pending.push(transition);
    if (transition.done) {
      this.flush();
      return;
    }
    if (this.pending.length >= nStep) {
      this.apply(this.pending.slice(0, nStep));
      this.pending.shift();
    }
  }

  clear(): void {
    this.pending = [];
  }

  private flush(): void {
    while (this.pending.length > 0) {
      this.apply(this.pending);
      this.pending.shift();
    }
  }
}
