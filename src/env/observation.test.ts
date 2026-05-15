import { describe, expect, test } from 'vitest';
import { createDefaultEnv } from './environment';
import { observationKey, observationKeyToString, encodeGhostZone, type Observation } from './observation';

const baseObs = (): Observation => ({
  pac: { x: 0, y: 0 },
  ghosts: [],
  wallMask: 0,
  nearestPelletDir: 0,
  ghostsEdible: false,
  ghostRel: [],
  ghostCodes: [0, 0],
  numEdibleBucket: 0,
});

describe('observation encoding', () => {
  test('is deterministic with same seed', () => {
    const a = createDefaultEnv();
    const b = createDefaultEnv();
    const oa = a.reset(123);
    const ob = b.reset(123);
    expect(observationKey(oa)).toBe(observationKey(ob));
  });

  test('ghost codes produce distinct keys for different nearest ghost positions', () => {
    const obs1: Observation = { ...baseObs(), ghostCodes: [2, 0] }; // nearest ghost: mid-up
    const obs2: Observation = { ...baseObs(), ghostCodes: [6, 0] }; // nearest ghost: far-up
    expect(observationKey(obs1)).not.toBe(observationKey(obs2));
  });

  test('second ghost slot produces distinct key from first', () => {
    const obs1: Observation = { ...baseObs(), ghostCodes: [2, 0] };
    const obs2: Observation = { ...baseObs(), ghostCodes: [0, 2] };
    expect(observationKey(obs1)).not.toBe(observationKey(obs2));
  });

  test('edibility bucket changes the key', () => {
    const none: Observation  = { ...baseObs(), numEdibleBucket: 0 };
    const some: Observation  = { ...baseObs(), numEdibleBucket: 1 };
    const all: Observation   = { ...baseObs(), numEdibleBucket: 2 };
    expect(observationKey(none)).not.toBe(observationKey(some));
    expect(observationKey(some)).not.toBe(observationKey(all));
    expect(observationKey(none)).not.toBe(observationKey(all));
  });

  test('pelletDir "none" sentinel is distinct from "up"', () => {
    const up: Observation   = { ...baseObs(), nearestPelletDir: 0 };
    const none: Observation = { ...baseObs(), nearestPelletDir: 4 };
    expect(observationKey(up)).not.toBe(observationKey(none));
  });

  test('absent ghost (code 0) is distinct from ghost-on-same-tile (code 1)', () => {
    const absent: Observation  = { ...baseObs(), ghostCodes: [0, 0] };
    const onTile: Observation  = { ...baseObs(), ghostCodes: [1, 0] };
    expect(observationKey(absent)).not.toBe(observationKey(onTile));
  });

  test('observationKeyToString round-trips v3 format', () => {
    const obs: Observation = { ...baseObs(), nearestPelletDir: 2, numEdibleBucket: 1, ghostCodes: [3, 7] };
    const str = observationKeyToString(observationKey(obs));
    expect(str).toMatch(/^v3:/);
    // wallMask=0, pelletDir=2, edibleBucket=1, gc0=3, gc1=7
    expect(str).toBe('v3:0:2:1:3:7');
  });

  test('encodeGhostZone: absent returns 0', () => {
    expect(encodeGhostZone(undefined, { x: 5, y: 5 }, 28)).toBe(0);
  });

  test('encodeGhostZone: ghost on same tile returns 1', () => {
    expect(encodeGhostZone({ x: 5, y: 5 }, { x: 5, y: 5 }, 28)).toBe(1);
  });

  test('encodeGhostZone: adjacent ghost returns 1', () => {
    expect(encodeGhostZone({ x: 6, y: 5 }, { x: 5, y: 5 }, 28)).toBe(1);
  });

  test('encodeGhostZone: mid-range ghost above returns 2 (up)', () => {
    // dist=3, dy<0 => up => 2+0=2
    expect(encodeGhostZone({ x: 5, y: 2 }, { x: 5, y: 5 }, 28)).toBe(2);
  });

  test('encodeGhostZone: far ghost to the right returns 7', () => {
    // dist=8, dx>0 => right => 6+1=7
    expect(encodeGhostZone({ x: 13, y: 5 }, { x: 5, y: 5 }, 28)).toBe(7);
  });

  test('encodeGhostZone: tunnel-wrapped ghost uses shortest path', () => {
    // Ghost at x=1, pac at x=26, width=28. Raw dx=-25 but wrapped dx=+3 (right, dist=3)
    expect(encodeGhostZone({ x: 1, y: 5 }, { x: 26, y: 5 }, 28)).toBe(3); // mid-right
  });
});
