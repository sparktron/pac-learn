import { describe, expect, test, beforeEach } from 'vitest';
import { chooseGhostMove } from './ghostAi';
import type { GhostState, WorldState, PacmanEnvironment } from '../env/environment';
import type { Direction, Vec2 } from '../engine/types';

// ── Test helpers ────────────────────────────────────────────────────
// A fully-open NxN world with solid border walls.
const openWorld = (n: number, over: Partial<WorldState> = {}): WorldState => ({
  width: n,
  height: n,
  pellets: [],
  powerPellets: [],
  heatmap: Array.from({ length: n }, () => Array(n).fill(0)),
  isWall: (x, y) => x <= 0 || x >= n - 1 || y <= 0 || y >= n - 1,
  isGhostHouse: () => false,
  ...over,
});

// A single open horizontal corridor on row `row`; everything else is wall.
const corridorWorld = (n: number, row: number, over: Partial<WorldState> = {}): WorldState =>
  openWorld(n, { isWall: (x, y) => y !== row || x <= 0 || x >= n - 1, ...over });

const mkGhost = (over: Partial<GhostState> = {}): GhostState => ({
  id: 0, pos: { x: 2, y: 2 }, aiType: 'classic', edibleTimer: 0,
  releaseDelay: 0, inBox: false, lastDir: null, pendingReverse: false, ...over,
});

// Minimal env stub exposing only what chooseGhostMove consults.
const mkEnv = (over: Partial<Record<keyof PacmanEnvironment, unknown>> = {}): PacmanEnvironment =>
  ({
    getPacDesiredDir: () => 'right' as Direction,
    getBlinkyPos: () => ({ x: 0, y: 0 }),
    getScatterTarget: () => ({ x: 1, y: 1 }),
    isScatterPhase: () => false,
    nextRand: () => 0,
    ...over,
  } as unknown as PacmanEnvironment);

describe('ghost AI', () => {
  let world: WorldState;
  let ghost: GhostState;

  beforeEach(() => {
    // Simple 5x5 grid with walls on edges, open in middle
    const isWall = (x: number, y: number) => x <= 0 || x >= 4 || y <= 0 || y >= 4;
    world = {
      width: 5,
      height: 5,
      pellets: [],
      powerPellets: [],
      heatmap: Array.from({ length: 5 }, () => Array(5).fill(0)),
      isWall,
      isGhostHouse: () => false,
    };
    ghost = { id: 0, pos: { x: 2, y: 2 }, aiType: 'classic', edibleTimer: 0, releaseDelay: 0, inBox: false, lastDir: null, pendingReverse: false };
  });

  test('classic AI moves toward Pac-Man', () => {
    const pacPos: Vec2 = { x: 3, y: 2 };
    const move = chooseGhostMove(world, ghost, pacPos);
    // Should move right toward Pac-Man
    expect(move).toBe('right');
  });

  test('classic AI avoids walls', () => {
    const pacPos: Vec2 = { x: 0, y: 0 }; // Off the board (wall)
    const move = chooseGhostMove(world, ghost, pacPos);
    // Should return a valid move (not null)
    expect(['up', 'down', 'left', 'right']).toContain(move);
  });

  test('can choose horizontal tunnel wraparound moves', () => {
    const tunnelWorld: WorldState = {
      ...world,
      isWall: (x, y) => y < 0 || y >= 5 || x < 0 || x >= 5 || y !== 2,
    };
    ghost.pos = { x: 0, y: 2 };

    expect(chooseGhostMove(tunnelWorld, ghost, { x: 4, y: 2 })).toBe('left');
  });

  // C7 regression: chase↔scatter transitions set pendingReverse per-ghost.
  // A single env-wide consume()-flag was eaten by ghost 0, leaving ghosts
  // 1..N never reversing on mode change.
  test('pendingReverse fires independently for each ghost', () => {
    const tunnelWorld: WorldState = {
      ...world,
      isWall: (x, y) => y < 0 || y >= 5 || x < 0 || x >= 5 || y !== 2,
    };
    const g0: GhostState = { id: 0, pos: { x: 2, y: 2 }, aiType: 'classic', edibleTimer: 0, releaseDelay: 0, inBox: false, lastDir: 'right', pendingReverse: true };
    const g1: GhostState = { id: 1, pos: { x: 3, y: 2 }, aiType: 'classic', edibleTimer: 0, releaseDelay: 0, inBox: false, lastDir: 'right', pendingReverse: true };
    expect(chooseGhostMove(tunnelWorld, g0, { x: 4, y: 2 })).toBe('left');
    expect(g0.pendingReverse).toBe(false);
    // g1 must also reverse — the per-ghost flag is independent.
    expect(chooseGhostMove(tunnelWorld, g1, { x: 4, y: 2 })).toBe('left');
    expect(g1.pendingReverse).toBe(false);
  });

  test('returns null when completely surrounded', () => {
    // Create a ghost in an isolated 1x1 space (surrounded by walls)
    ghost.pos = { x: 2, y: 2 };
    const isolatedWorld: WorldState = {
      ...world,
      isWall: (x, y) => !(x === 2 && y === 2),
    };
    const move = chooseGhostMove(isolatedWorld, ghost, { x: 0, y: 0 });
    expect(move).toBeNull();
  });

  // ── D3.4: removeReverse ──────────────────────────────────────────
  test('does not reverse into its last direction when other moves exist', () => {
    const w = corridorWorld(5, 2); // open x∈[1,3] on row 2
    const g = mkGhost({ pos: { x: 2, y: 2 }, lastDir: 'right' });
    // Pac is behind (to the left); reversing would head toward it, but the
    // reverse of 'right' is filtered, so the ghost continues right (away).
    expect(chooseGhostMove(w, g, { x: 1, y: 2 })).toBe('right');
  });

  test('allows reverse at a dead-end (only the reverse is legal)', () => {
    const w = corridorWorld(5, 2);
    const g = mkGhost({ pos: { x: 3, y: 2 }, lastDir: 'right' }); // wall at x=4
    expect(chooseGhostMove(w, g, { x: 1, y: 2 })).toBe('left');
  });

  // ── D3.5 / D3.2: edible flee (may reverse) ───────────────────────
  test('edible ghost flees away from Pac-Man, reversing if needed', () => {
    const w = corridorWorld(5, 2);
    // Just moved right; now edible with Pac ahead on the right → must reverse
    // left to flee. (Pre-D3.2 this filtered the reverse and fled toward Pac.)
    const g = mkGhost({ pos: { x: 2, y: 2 }, lastDir: 'right', edibleTimer: 5 });
    expect(chooseGhostMove(w, g, { x: 3, y: 2 })).toBe('left');
  });

  // ── D3.6: in-box BFS toward the exit ─────────────────────────────
  test('in-box ghost steps toward ghostHouseExit', () => {
    // Vertical shaft: column x=2 open on rows 1..3; exit at the top (2,1).
    const w = openWorld(5, {
      isWall: (x, y) => !(x === 2 && y >= 1 && y <= 3),
      isGhostHouse: (x, y) => x === 2 && (y === 2 || y === 3),
      ghostHouseExit: { x: 2, y: 1 },
    });
    const g = mkGhost({ pos: { x: 2, y: 3 }, inBox: true });
    expect(chooseGhostMove(w, g, { x: 0, y: 0 })).toBe('up');
  });

  // ── D3.7: personality targeting ──────────────────────────────────
  test('Pinky (role 1) targets ahead of Pac, not Pac itself', () => {
    const w = openWorld(9);
    const env = mkEnv({ getPacDesiredDir: () => 'right' });
    // Ghost right of Pac; Pac faces right so Pinky's 4-ahead target is further
    // right → ghost moves right, whereas Blinky (target=Pac) would move left.
    const pinky = mkGhost({ id: 1, pos: { x: 4, y: 4 } });
    expect(chooseGhostMove(w, pinky, { x: 2, y: 4 }, env)).toBe('right');
    const blinky = mkGhost({ id: 0, pos: { x: 4, y: 4 } });
    expect(chooseGhostMove(w, blinky, { x: 2, y: 4 }, env)).toBe('left');
  });

  // A2: an explicit personality overrides the id-derived role (same geometry as
  // the Pinky test above, but the roles are forced via personality, not id).
  test('personality overrides the id-derived role (A2)', () => {
    const w = openWorld(9);
    const env = mkEnv({ getPacDesiredDir: () => 'right' });
    // id 0 defaults to Blinky (→ left), but personality 1 = Pinky (→ right).
    const asPinky = mkGhost({ id: 0, personality: 1, pos: { x: 4, y: 4 } });
    expect(chooseGhostMove(w, asPinky, { x: 2, y: 4 }, env)).toBe('right');
    // id 1 defaults to Pinky (→ right), but personality 0 = Blinky (→ left).
    const asBlinky = mkGhost({ id: 1, personality: 0, pos: { x: 4, y: 4 } });
    expect(chooseGhostMove(w, asBlinky, { x: 2, y: 4 }, env)).toBe('left');
  });

  test('Inky (role 2) consults Blinky position (vector targeting)', () => {
    const w = openWorld(9);
    const inky = (blinky: Vec2) =>
      chooseGhostMove(w, mkGhost({ id: 2, pos: { x: 2, y: 4 } }), { x: 4, y: 2 },
        mkEnv({ getPacDesiredDir: () => 'up', getBlinkyPos: () => blinky }));
    // pivot = 2-ahead of Pac(4,2) facing up = (4,0).
    //   blinky=(4,0) → target (4,0): ghost(2,4) moves up.
    //   blinky=(0,0) → target (8,0): ghost(2,4) moves right.
    expect(inky({ x: 4, y: 0 })).toBe('up');
    expect(inky({ x: 0, y: 0 })).toBe('right');
  });

  test('Clyde (role 3) chases when far, scatters when within 8 tiles', () => {
    const w = openWorld(15);
    const env = mkEnv({ getScatterTarget: () => ({ x: 1, y: 1 }) });
    // Far (>8): behaves like Blinky, heading toward Pac.
    const far = mkGhost({ id: 3, pos: { x: 12, y: 7 } });
    expect(chooseGhostMove(w, far, { x: 1, y: 7 }, env)).toBe('left');
    // Near (≤8): heads to its scatter corner (1,1) — up/left, not toward Pac.
    const near = mkGhost({ id: 3, pos: { x: 5, y: 7 } });
    expect(['up', 'left']).toContain(chooseGhostMove(w, near, { x: 4, y: 7 }, env));
  });

  // ── D3.8: scatter phase ──────────────────────────────────────────
  test('scatter phase routes ghosts to their corner', () => {
    const w = openWorld(9);
    const env = mkEnv({ isScatterPhase: () => true, getScatterTarget: () => ({ x: 1, y: 1 }) });
    const g = mkGhost({ id: 0, pos: { x: 4, y: 4 } });
    // Target corner (1,1) is up-left of the ghost.
    expect(['up', 'left']).toContain(chooseGhostMove(w, g, { x: 7, y: 7 }, env));
  });

  // ── D3.9: heatmap + hybrid ───────────────────────────────────────
  test('heatmap AI climbs the heat gradient', () => {
    const heatmap = Array.from({ length: 5 }, () => Array(5).fill(0));
    heatmap[2][3] = 10; // hottest tile is to the right
    const w = openWorld(5, { heatmap });
    const g = mkGhost({ pos: { x: 2, y: 2 }, aiType: 'heatmap' });
    expect(chooseGhostMove(w, g, { x: 0, y: 0 })).toBe('right');
  });

  test('hybrid AI uses the seeded env RNG and is reproducible', () => {
    const w = openWorld(9);
    const g = () => mkGhost({ id: 0, pos: { x: 4, y: 4 }, aiType: 'hybrid' });
    const env = () => mkEnv({ nextRand: () => 0 }); // <0.7 → classic branch
    const a = chooseGhostMove(w, g(), { x: 6, y: 4 }, env());
    const b = chooseGhostMove(w, g(), { x: 6, y: 4 }, env());
    expect(a).toBe(b);
    expect(a).toBe('right'); // classic targeting toward Pac on the right
  });

  // ── D3.10: free ghosts avoid the ghost house ─────────────────────
  test('free ghost will not step into a ghost-house tile', () => {
    const w = openWorld(5, { isGhostHouse: (x, y) => x === 3 && y === 2 });
    const g = mkGhost({ pos: { x: 2, y: 2 } });
    // Pac is on the ghost-house tile to the right; the ghost must not enter it.
    expect(chooseGhostMove(w, g, { x: 3, y: 2 })).not.toBe('right');
  });
});
