import { describe, expect, test } from 'vitest';
import { mergePolicies } from './policyMerge';
import type { SerializedPolicy } from './qlearning';

const policy = (over: Partial<SerializedPolicy>): SerializedPolicy => ({
  algorithm: 'qlearning',
  mazeId: 'pacman-classic',
  timestamp: '',
  numGhostsEncoded: 2,
  observationKeyVersion: 9,
  hyper: { alpha: 0.1, gamma: 0.99, epsilon: 0.2, epsilonDecay: 1, epsilonMin: 0.05, optimisticInit: 50 },
  qTable: {},
  visitTable: {},
  ...over,
});

describe('mergePolicies', () => {
  test('visit-weighted (√-tapered) average over visited slots', () => {
    const p1 = policy({ qTable: { K: [10, 50, 50, 50] }, visitTable: { K: [4, 0, 0, 0] } }); // w=√4=2
    const p2 = policy({ qTable: { K: [20, 50, 50, 50] }, visitTable: { K: [1, 0, 0, 0] } }); // w=√1=1
    const { merged } = mergePolicies([p1, p2]);
    // slot 0: (10·2 + 20·1) / (2+1) = 40/3
    expect(merged.qTable.K[0]).toBeCloseTo(40 / 3, 5);
    // raw visits summed
    expect(merged.visitTable!.K).toEqual([5, 0, 0, 0]);
  });

  test('slots no worker visited stay at optimisticInit (C1)', () => {
    // Slot present in qTable with a learned-looking value but ZERO visits → init,
    // not averaged in. This is the catastrophic-merge guard.
    const p = policy({ qTable: { K: [-90, 1, 2, 3] }, visitTable: { K: [0, 0, 0, 0] } });
    const { merged } = mergePolicies([p]);
    expect(merged.qTable.K).toEqual([50, 50, 50, 50]);
  });

  test('a learned death signal is not masked by an unvisited init peer (C1)', () => {
    const learned = policy({ qTable: { K: [-90, 50, 50, 50] }, visitTable: { K: [9, 0, 0, 0] } });
    const untried = policy({ qTable: { K: [50, 50, 50, 50] }, visitTable: { K: [0, 0, 0, 0] } });
    const { merged } = mergePolicies([learned, untried]);
    expect(merged.qTable.K[0]).toBeCloseTo(-90, 5); // only the visited worker counts
  });

  test('ε reset to the MAX across inputs', () => {
    const p1 = policy({ hyper: { ...policy({}).hyper, epsilon: 0.2 } });
    const p2 = policy({ hyper: { ...policy({}).hyper, epsilon: 0.5 } });
    expect(mergePolicies([p1, p2]).merged.hyper.epsilon).toBe(0.5);
  });

  test('stats report shared/total states', () => {
    const p1 = policy({ qTable: { A: [1, 1, 1, 1], B: [1, 1, 1, 1] }, visitTable: { A: [1, 0, 0, 0], B: [1, 0, 0, 0] } });
    const p2 = policy({ qTable: { A: [2, 2, 2, 2] }, visitTable: { A: [1, 0, 0, 0] } });
    const { stats } = mergePolicies([p1, p2]);
    expect(stats.totalStates).toBe(2); // A, B
    expect(stats.sharedStates).toBe(1); // A in both
  });

  test('throws on observationKeyVersion mismatch', () => {
    const p1 = policy({ observationKeyVersion: 9 });
    const p2 = policy({ observationKeyVersion: 8 });
    expect(() => mergePolicies([p1, p2])).toThrow(/observationKeyVersion mismatch/);
  });

  test('throws on legacy policy without visitTable', () => {
    const p = policy({ qTable: { K: [1, 2, 3, 4] } });
    delete (p as { visitTable?: unknown }).visitTable;
    expect(() => mergePolicies([p])).toThrow(/visitTable/);
  });

  test('throws on empty input', () => {
    expect(() => mergePolicies([])).toThrow(/no policies/);
  });
});
