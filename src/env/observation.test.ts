import { describe, expect, test } from 'vitest';
import { createDefaultEnv, type WorldState } from './environment';
import {
  observationKey,
  observationKeyToString,
  stringToObservationKey,
  encodeObservation,
  encodeGhostZone,
  encodeGhostHeading,
  pelletsRemainingBucket,
  powerPelletsLeftBucket,
  type Observation,
} from './observation';

const baseObs = (): Observation => ({
  pac: { x: 0, y: 0 },
  ghosts: [],
  wallMask: 0,
  nearestPelletDir: 0,
  ghostsEdible: false,
  ghostRel: [],
  ghostCodes: [0, 0],
  ghostHeadings: [0, 0],
  lastAction: -1,
  pelletsRemainingBucket: 4, // "opening" — most tests assume game start
  powerPelletsLeftBucket: 2, // "many" — game start
  nearestPelletDist: 1,
  nearestGhostDists: [Infinity, Infinity],
  nearestGhostRel: [null, null],
});

describe('observation encoding', () => {
  test('is deterministic with same seed', () => {
    const a = createDefaultEnv();
    const b = createDefaultEnv();
    const oa = a.reset(123);
    const ob = b.reset(123);
    expect(observationKey(oa)).toBe(observationKey(ob));
  });

  // v10 regression guard. wallMask used to probe raw `pac.x + dx` while the
  // env's canMove() wraps through nextPosition() first, so at a tunnel mouth
  // the observation reported a wall for a move the env offers as legal and
  // executes. The invariant: a legal action is never encoded as blocked.
  // (The converse does not hold — getLegalActions also excludes the ghost
  // house, so an unblocked action can still be illegal.)
  test('legal actions are never encoded as walls, including at tunnel mouths (v10)', () => {
    const env = createDefaultEnv();
    env.reset(11);
    const w = env.world.width;

    const checkHere = (): void => {
      const o = env.observe();
      for (const a of env.getLegalActionIndices()) {
        // wallMask bits are CARD order (N/E/S/W); actions are up/down/left/right.
        const bit = [0, 2, 3, 1][a];
        expect((o.wallMask >> bit) & 1).toBe(0);
      }
    };

    // Walk both tunnel mouths explicitly — a random rollout does not reliably
    // reach them, and this is the exact geometry the v9 mask got wrong.
    let tunnelRows = 0;
    for (let y = 0; y < env.world.height; y += 1) {
      if (env.world.isWall(0, y) || env.world.isWall(w - 1, y)) continue;
      tunnelRows += 1;
      for (const x of [0, w - 1]) {
        env.getPacmen()[0].pos = { x, y };
        checkHere();
      }
    }
    // Guard the guard: on a maze with no wrap row this test would prove nothing.
    expect(tunnelRows).toBeGreaterThan(0);

    // Then a normal rollout for the ordinary (non-tunnel) tiles.
    env.reset(11);
    for (let step = 0; step < 2000; step += 1) {
      checkHere();
      const legal = env.getLegalActionIndices();
      const r = env.step(legal[step % legal.length]);
      if (r.done) env.reset(11 + step);
    }
  });

  test('different ghost zone codes produce distinct keys', () => {
    const obs1: Observation = { ...baseObs(), ghostCodes: [3, 0] }; // mid-up dangerous
    const obs2: Observation = { ...baseObs(), ghostCodes: [11, 0] }; // far-up dangerous
    expect(observationKey(obs1)).not.toBe(observationKey(obs2));
  });

  test('second ghost slot produces distinct key from first', () => {
    const obs1: Observation = { ...baseObs(), ghostCodes: [3, 0] };
    const obs2: Observation = { ...baseObs(), ghostCodes: [0, 3] };
    expect(observationKey(obs1)).not.toBe(observationKey(obs2));
  });

  test('edible vs dangerous ghost at same position produce distinct keys', () => {
    // zone 1 (here), dangerous = code 1; edible = code 2
    const dangerous: Observation = { ...baseObs(), ghostCodes: [1, 0] };
    const edible: Observation    = { ...baseObs(), ghostCodes: [2, 0] };
    expect(observationKey(dangerous)).not.toBe(observationKey(edible));
  });

  test('pelletDir "none" sentinel is distinct from "up"', () => {
    const up: Observation   = { ...baseObs(), nearestPelletDir: 0 };
    const none: Observation = { ...baseObs(), nearestPelletDir: 4 };
    expect(observationKey(up)).not.toBe(observationKey(none));
  });

  test('absent ghost (code 0) is distinct from ghost-on-same-tile (code 1)', () => {
    const absent: Observation = { ...baseObs(), ghostCodes: [0, 0] };
    const onTile: Observation = { ...baseObs(), ghostCodes: [1, 0] };
    expect(observationKey(absent)).not.toBe(observationKey(onTile));
  });

  test('observationKeyToString round-trips v11 format', () => {
    const obs: Observation = {
      ...baseObs(),
      nearestPelletDir: 2,
      ghostCodes: [3, 14],
      ghostHeadings: [1, 2],
      lastAction: 1,
      pelletsRemainingBucket: 2,
      powerPelletsLeftBucket: 1,
    };
    const str = observationKeyToString(observationKey(obs));
    expect(str).toMatch(/^v11:/);
    // wallMask=0, pelletDir=2, gc0=3, gh0=1, gc1=14, gh1=2, lastAction=1, pelletsBucket=2, powerBucket=1
    expect(str).toBe('v11:0:2:3:1:14:2:1:2:1');
  });

  // D5.10: stringToObservationKey is the exact inverse of the numeric key path,
  // so observationKey → string → stringToObservationKey round-trips. This is the
  // shared decode that qlearning.load() now uses (no hardcoded base constants).
  test('stringToObservationKey round-trips observationKey (D5.10)', () => {
    const cases: Observation[] = [
      baseObs(),
      { ...baseObs(), nearestPelletDir: 3, ghostCodes: [3, 14], ghostHeadings: [1, 2], lastAction: 2, pelletsRemainingBucket: 1, powerPelletsLeftBucket: 1 },
      { ...baseObs(), wallMask: 15, ghostCodes: [18, 18], ghostHeadings: [2, 2], lastAction: 3, pelletsRemainingBucket: 0, powerPelletsLeftBucket: 0 },
    ];
    for (const o of cases) {
      const key = observationKey(o);
      expect(stringToObservationKey(observationKeyToString(key))).toBe(key);
    }
  });

  test('stringToObservationKey rejects wrong version / malformed strings (D5.10)', () => {
    expect(stringToObservationKey('v8:0:0:0:0:0:0:0:0:0')).toBeNull(); // wrong version
    expect(stringToObservationKey('v11:0:0:0')).toBeNull(); // too few fields
    expect(stringToObservationKey('v11:0:x:0:0:0:0:0:0:0')).toBeNull(); // non-numeric
  });

  test('different ghostHeadings produce distinct keys', () => {
    const approaching: Observation = { ...baseObs(), ghostCodes: [3, 0], ghostHeadings: [1, 0] };
    const receding:    Observation = { ...baseObs(), ghostCodes: [3, 0], ghostHeadings: [2, 0] };
    expect(observationKey(approaching)).not.toBe(observationKey(receding));
  });

  test('different pelletsRemainingBucket values produce distinct keys', () => {
    const opening: Observation = { ...baseObs(), pelletsRemainingBucket: 4 };
    const endgame: Observation = { ...baseObs(), pelletsRemainingBucket: 0 };
    expect(observationKey(opening)).not.toBe(observationKey(endgame));
  });

  test('different powerPelletsLeftBucket values produce distinct keys', () => {
    const many: Observation = { ...baseObs(), powerPelletsLeftBucket: 2 };
    const none: Observation = { ...baseObs(), powerPelletsLeftBucket: 0 };
    expect(observationKey(many)).not.toBe(observationKey(none));
  });

  test('different lastAction values produce distinct keys', () => {
    const moving = { ...baseObs(), lastAction: 1 }; // moved right
    const start  = { ...baseObs(), lastAction: -1 }; // episode start
    expect(observationKey(moving)).not.toBe(observationKey(start));
  });

  test('lastAction does not collide with ghost zone encoding', () => {
    // lastAction=0 (up) with gc1=0 vs lastAction=0 (up) with gc1=1 — should differ
    const a = { ...baseObs(), ghostCodes: [0, 0] as [number, number], lastAction: 0 };
    const b = { ...baseObs(), ghostCodes: [0, 1] as [number, number], lastAction: 0 };
    expect(observationKey(a)).not.toBe(observationKey(b));
  });

  // ── encodeGhostZone ──────────────────────────────────────────────────────

  test('encodeGhostZone: absent returns 0', () => {
    expect(encodeGhostZone(undefined, { x: 5, y: 5 }, 28)).toBe(0);
  });

  test('encodeGhostZone: same-tile dangerous → 1', () => {
    // zone=1, not edible: (1-1)*2+0+1 = 1
    expect(encodeGhostZone({ x: 5, y: 5 }, { x: 5, y: 5 }, 28, false)).toBe(1);
  });

  test('encodeGhostZone: same-tile edible → 2', () => {
    // zone=1, edible: (1-1)*2+1+1 = 2
    expect(encodeGhostZone({ x: 5, y: 5 }, { x: 5, y: 5 }, 28, true)).toBe(2);
  });

  test('encodeGhostZone: adjacent ghost dangerous → 1', () => {
    // dist=1, zone=1, not edible → 1
    expect(encodeGhostZone({ x: 6, y: 5 }, { x: 5, y: 5 }, 28, false)).toBe(1);
  });

  test('encodeGhostZone: mid-range ghost above dangerous → 3', () => {
    // dist=3, dy<0 → up → zone=2+0=2, not edible: (2-1)*2+0+1=3
    expect(encodeGhostZone({ x: 5, y: 2 }, { x: 5, y: 5 }, 28, false)).toBe(3);
  });

  test('encodeGhostZone: mid-range ghost above edible → 4', () => {
    // zone=2, edible: (2-1)*2+1+1=4
    expect(encodeGhostZone({ x: 5, y: 2 }, { x: 5, y: 5 }, 28, true)).toBe(4);
  });

  test('encodeGhostZone: far ghost right dangerous → 13', () => {
    // dist=8, dx>0 → right → zone=6+1=7, not edible: (7-1)*2+0+1=13
    expect(encodeGhostZone({ x: 13, y: 5 }, { x: 5, y: 5 }, 28, false)).toBe(13);
  });

  test('encodeGhostZone: far ghost right edible → 14', () => {
    // zone=7, edible: (7-1)*2+1+1=14
    expect(encodeGhostZone({ x: 13, y: 5 }, { x: 5, y: 5 }, 28, true)).toBe(14);
  });

  test('encodeGhostZone: tunnel-wrapped ghost uses shortest path', () => {
    // Ghost at x=1, pac at x=26, width=28. Raw dx=-25, wrapped dx=+3 (right, dist=3).
    // zone=2+1=3 (mid-right), not edible: (3-1)*2+0+1=5
    expect(encodeGhostZone({ x: 1, y: 5 }, { x: 26, y: 5 }, 28, false)).toBe(5);
  });

  // ── encodeGhostHeading ────────────────────────────────────────────────────

  test('encodeGhostHeading: absent ghost → 0', () => {
    expect(encodeGhostHeading(undefined, { x: 5, y: 5 }, 28, 'up')).toBe(0);
  });

  test('encodeGhostHeading: ghost with no lastDir → 0', () => {
    expect(encodeGhostHeading({ x: 4, y: 5 }, { x: 5, y: 5 }, 28, null)).toBe(0);
  });

  test('encodeGhostHeading: ghost moving toward pac → 1 (approaching)', () => {
    // Pac at (5,5), ghost at (3,5) moving right (+1,0). dx=+2, dot=+2 → approaching.
    expect(encodeGhostHeading({ x: 3, y: 5 }, { x: 5, y: 5 }, 28, 'right')).toBe(1);
  });

  test('encodeGhostHeading: ghost moving away from pac → 2 (receding)', () => {
    // Pac at (5,5), ghost at (3,5) moving left (-1,0). dx=+2, dot=-2 → receding.
    expect(encodeGhostHeading({ x: 3, y: 5 }, { x: 5, y: 5 }, 28, 'left')).toBe(2);
  });

  test('encodeGhostHeading: perpendicular movement → 0', () => {
    // Pac at (5,5), ghost at (3,5) moving up (0,-1). dy=0 (same row) → dot=0.
    expect(encodeGhostHeading({ x: 3, y: 5 }, { x: 5, y: 5 }, 28, 'up')).toBe(0);
  });

  test('encodeGhostHeading: tunnel wrap — left through tunnel approaches pac', () => {
    // Ghost at x=0, pac at x=27, width=28. wrapped dx = -1.
    // Ghost moving left (-1,0): dot = (-1)*(-1) = +1 → approaching through the tunnel.
    expect(encodeGhostHeading({ x: 0, y: 5 }, { x: 27, y: 5 }, 28, 'left')).toBe(1);
  });

  // D4.4: bucket boundary correctness (the key-distinctness tests above only
  // prove different buckets differ, not that a given fraction maps correctly).
  test('pelletsRemainingBucket maps fraction boundaries correctly (D4.4)', () => {
    expect(pelletsRemainingBucket(0, 100)).toBe(0);   // all gone → endgame
    expect(pelletsRemainingBucket(10, 100)).toBe(0);  // 0.10 inclusive → endgame
    expect(pelletsRemainingBucket(11, 100)).toBe(1);  // just over → late
    expect(pelletsRemainingBucket(25, 100)).toBe(1);  // 0.25 inclusive → late
    expect(pelletsRemainingBucket(26, 100)).toBe(2);  // just over → mid
    expect(pelletsRemainingBucket(50, 100)).toBe(2);  // 0.50 inclusive → mid
    expect(pelletsRemainingBucket(75, 100)).toBe(3);  // 0.75 inclusive → early
    expect(pelletsRemainingBucket(76, 100)).toBe(4);  // just over → opening
    expect(pelletsRemainingBucket(100, 100)).toBe(4); // full board → opening
    expect(pelletsRemainingBucket(5, 0)).toBe(0);     // total=0 guard → endgame
  });

  test('powerPelletsLeftBucket maps count to none/one/many (D4.4)', () => {
    expect(powerPelletsLeftBucket(0)).toBe(0);
    expect(powerPelletsLeftBucket(1)).toBe(1);
    expect(powerPelletsLeftBucket(2)).toBe(2);
    expect(powerPelletsLeftBucket(5)).toBe(2);
  });

  // ── nearestPelletDir / bfsPelletDir (D4.3) ────────────────────────────────
  // This is the behavioral assertion that would have caught M2/D4.1: the
  // pellet-direction index must equal the action that walks toward the pellet.
  // v9 alignment: nearestPelletDir=k ⇔ DIRECTIONS action k (up=0, down=1,
  // left=2, right=3).

  // Fully-open room; only out-of-bounds tiles count as walls. A single pellet
  // is planted so the BFS first-step direction is unambiguous.
  const openWorld = (w: number, h: number): WorldState => ({
    width: w,
    height: h,
    pellets: Array.from({ length: h }, () => Array.from({ length: w }, () => false)),
    powerPellets: Array.from({ length: h }, () => Array.from({ length: w }, () => false)),
    heatmap: [],
    isWall: (x, y) => x < 0 || y < 0 || x >= w || y >= h,
    isGhostHouse: () => false,
  });

  const pelletDirToward = (px: number, py: number): number => {
    const world = openWorld(7, 7);
    world.pellets[py][px] = true;
    return encodeObservation(world, { x: 3, y: 3 }, []).nearestPelletDir;
  };

  test('nearestPelletDir points up=0 toward a pellet above (D4.3)', () => {
    expect(pelletDirToward(3, 2)).toBe(0); // pellet directly above pac → action 0 (up)
  });

  test('nearestPelletDir points down=1 toward a pellet below (D4.3)', () => {
    expect(pelletDirToward(3, 4)).toBe(1); // action 1 (down)
  });

  test('nearestPelletDir points left=2 toward a pellet to the left (D4.3)', () => {
    expect(pelletDirToward(2, 3)).toBe(2); // action 2 (left)
  });

  test('nearestPelletDir points right=3 toward a pellet to the right (D4.3)', () => {
    expect(pelletDirToward(4, 3)).toBe(3); // action 3 (right)
  });

  test('nearestPelletDir returns the "none" sentinel 4 when no pellet is reachable (D4.3)', () => {
    const world = openWorld(7, 7); // no pellets at all
    expect(encodeObservation(world, { x: 3, y: 3 }, []).nearestPelletDir).toBe(4);
  });

  test('nearestPelletDir resolves a pellet beyond the radius-12 fast path', () => {
    const world = openWorld(40, 3);
    world.pellets[1][16] = true;
    const observation = encodeObservation(world, { x: 1, y: 1 }, []);
    expect(observation.nearestPelletDir).toBe(3);
    expect(observation.nearestPelletDist).toBe(15);
  });

  test('nearestPelletDir uses the tunnel: pellet across the wrap is reached by going left (D4.3)', () => {
    const world = openWorld(7, 7);
    world.pellets[3][6] = true; // far-right column, same row as pac at x=0
    // From x=0, action 2 (left) wraps to x=6 in one step — the shortest path.
    expect(encodeObservation(world, { x: 0, y: 3 }, []).nearestPelletDir).toBe(2);
  });
});
