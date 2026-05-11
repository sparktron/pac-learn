import { describe, expect, test } from 'vitest';
import { createDefaultEnv } from './environment';
import { observationKey, observationKeyToString, type Observation } from './observation';

describe('observation encoding', () => {
  test('is deterministic with same seed', () => {
    const a = createDefaultEnv();
    const b = createDefaultEnv();
    const oa = a.reset(123);
    const ob = b.reset(123);
    expect(observationKey(oa)).toBe(observationKey(ob));
  });

  test('keeps ghost offsets in high key positions distinct', () => {
    const base: Observation = {
      pac: { x: 0, y: 0 },
      ghosts: [],
      wallMask: 0,
      nearestPelletDir: 0,
      ghostRel: [
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
      ],
    };
    const changedFourthGhost: Observation = {
      ...base,
      ghostRel: [
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
        { dx: 1, dy: 0 },
      ],
    };

    expect(observationKey(base)).not.toBe(observationKey(changedFourthGhost));
    expect(observationKeyToString(observationKey(changedFourthGhost))).toBe('0:0:0,0:0,0:0,0:1,0');
  });
});
