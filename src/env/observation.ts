import type { Vec2 } from '../engine/types';
import type { WorldState } from './environment';

export interface Observation {
  pac: Vec2;
  ghosts: Vec2[];
  wallMask: number;
  /** 0=up, 1=right, 2=down, 3=left, 4=no pellet reachable within search radius */
  nearestPelletDir: number;
  /** Raw tunnel-aware clamped offsets; kept for rendering. Not used in observationKey. */
  ghostRel: Array<{ dx: number; dy: number }>;
  /** True when at least one ghost is currently edible. Derived from numEdibleBucket. */
  ghostsEdible: boolean;
  /**
   * Compact zone code for the two nearest active ghosts (sorted by Manhattan distance).
   *   0        = absent (fewer than N active ghosts)
   *   1        = here or adjacent (dist ≤ 1)
   *   2–5      = mid range (dist 2–5): up / right / down / left
   *   6–9      = far range (dist 6+):  up / right / down / left
   *
   * Using (0,0) padding for "absent" was the old bug — it aliased the "ghost on same
   * tile" slot.  A dedicated absent=0 code eliminates that collision.
   */
  ghostCodes: [number, number];
  /** 0 = no edible ghosts, 1 = some edible, 2 = all edible. */
  numEdibleBucket: number;
}

// ─── Key version ─────────────────────────────────────────────────────────────
// Bump this any time the observationKey() layout changes so that load() can
// detect incompatible saved policies and discard their Q-tables rather than
// silently corrupting training with mismatched keys.
export const OBSERVATION_KEY_VERSION = 3;

// ─── Internal helpers ────────────────────────────────────────────────────────

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

// Pellet-direction encoding: up=0, right=1, down=2, left=3 (rotational order).
// This is NOT the same order as DIRECTIONS in engine/types (up, down, left, right).
// The agent treats nearestPelletDir as an opaque feature, so the encoding just
// needs to be internally consistent between bfsPelletDir and observationKey.
const DIRS: Array<{ dx: number; dy: number }> = [
  { dx: 0, dy: -1 }, // 0 = up
  { dx: 1, dy: 0 },  // 1 = right
  { dx: 0, dy: 1 },  // 2 = down
  { dx: -1, dy: 0 }, // 3 = left
];

const PELLET_SEARCH_RADIUS = 12;

/**
 * BFS from pac to find the first-step direction toward the nearest pellet
 * (regular or power) reachable within PELLET_SEARCH_RADIUS tiles. Returns 4
 * (the "none" sentinel) if no pellet is reachable in radius — this is what
 * lets the agent distinguish "no nearby pellet" from "pellet straight up",
 * which the previous raycast implementation conflated.
 */
const bfsPelletDir = (world: WorldState, pac: Vec2): number => {
  const w = world.width;
  const h = world.height;
  const visited = new Uint8Array(w * h);
  const key = (x: number, y: number): number => y * w + x;
  visited[key(pac.x, pac.y)] = 1;
  type Node = { x: number; y: number; firstDir: number; depth: number };
  const queue: Node[] = [];
  for (let i = 0; i < 4; i += 1) {
    let nx = pac.x + DIRS[i].dx;
    const ny = pac.y + DIRS[i].dy;
    if (nx < 0) nx = w - 1;
    if (nx >= w) nx = 0;
    if (ny < 0 || ny >= h || world.isWall(nx, ny)) continue;
    if (visited[key(nx, ny)]) continue;
    visited[key(nx, ny)] = 1;
    queue.push({ x: nx, y: ny, firstDir: i, depth: 1 });
  }
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    if (world.pellets[cur.y]?.[cur.x] || world.powerPellets[cur.y]?.[cur.x]) return cur.firstDir;
    if (cur.depth >= PELLET_SEARCH_RADIUS) continue;
    for (let i = 0; i < 4; i += 1) {
      let nx = cur.x + DIRS[i].dx;
      const ny = cur.y + DIRS[i].dy;
      if (nx < 0) nx = w - 1;
      if (nx >= w) nx = 0;
      if (ny < 0 || ny >= h || world.isWall(nx, ny)) continue;
      if (visited[key(nx, ny)]) continue;
      visited[key(nx, ny)] = 1;
      queue.push({ x: nx, y: ny, firstDir: cur.firstDir, depth: cur.depth + 1 });
    }
  }
  return 4;
};

/** Tunnel-aware ghost direction quadrant: 0=up, 1=right, 2=down, 3=left. */
const ghostQuadrant = (dx: number, dy: number): number =>
  Math.abs(dy) >= Math.abs(dx) ? (dy < 0 ? 0 : 2) : (dx > 0 ? 1 : 3);

/**
 * Encode one ghost slot into a compact zone code (0–9).
 * Pass `undefined` for absent/missing ghost slots to get code 0.
 */
export const encodeGhostZone = (g: Vec2 | undefined, pac: Vec2, worldWidth: number): number => {
  if (!g) return 0;
  let dx = g.x - pac.x;
  if (dx > worldWidth / 2) dx -= worldWidth;
  else if (dx < -worldWidth / 2) dx += worldWidth;
  const dy = g.y - pac.y;
  const dist = Math.abs(dx) + Math.abs(dy);
  if (dist <= 1) return 1;
  const dir = ghostQuadrant(dx, dy);
  return dist <= 5 ? 2 + dir : 6 + dir;
};

// ─── Public API ──────────────────────────────────────────────────────────────

export const encodeObservation = (
  world: WorldState,
  pac: Vec2,
  ghosts: Vec2[],
  ghostsEdible = false,
  numEdible = 0,
): Observation => {
  // Encode only the 4 immediate cardinal neighbors as a 4-bit mask (N/E/S/W → bits 0-3).
  // The old 5×5 window (25 bits = 33 M values) dwarfed every other feature and made
  // generalisation across corridors impossible.  4 bits = 16 values covers every
  // junction shape a Pac-Man maze can produce.
  const CARD = [{ dx: 0, dy: -1 }, { dx: 1, dy: 0 }, { dx: 0, dy: 1 }, { dx: -1, dy: 0 }];
  let mask = 0;
  CARD.forEach(({ dx, dy }, i) => {
    if (world.isWall(pac.x + dx, pac.y + dy)) mask |= (1 << i);
  });

  // Sort ghosts by tunnel-aware Manhattan distance; take nearest two.
  const sorted = ghosts
    .map((g) => {
      let dx = g.x - pac.x;
      if (dx > world.width / 2) dx -= world.width;
      else if (dx < -world.width / 2) dx += world.width;
      return { g, dist: Math.abs(dx) + Math.abs(g.y - pac.y) };
    })
    .sort((a, b) => a.dist - b.dist);

  const ghostCodes: [number, number] = [
    encodeGhostZone(sorted[0]?.g, pac, world.width),
    encodeGhostZone(sorted[1]?.g, pac, world.width),
  ];

  const numEdibleBucket = numEdible === 0 ? 0 : numEdible >= ghosts.length ? 2 : 1;

  return {
    pac,
    ghosts,
    wallMask: mask,
    nearestPelletDir: bfsPelletDir(world, pac),
    ghostRel: ghosts.map((g) => {
      let dx = g.x - pac.x;
      if (dx > world.width / 2) dx -= world.width;
      else if (dx < -world.width / 2) dx += world.width;
      return { dx: clamp(dx, -3, 3), dy: clamp(g.y - pac.y, -3, 3) };
    }),
    ghostsEdible,
    ghostCodes,
    numEdibleBucket,
  };
};

// ─── Key encoding ────────────────────────────────────────────────────────────

const GHOST_ZONE_BASE    = 10; // 10 zone codes per ghost slot
const WALL_MASK_BASE     = 16; // 4-bit cardinal wall mask (N/E/S/W) = 16 values
const PELLET_DIR_BASE    = 5;  // up/right/down/left/none
const EDIBLE_BUCKET_BASE = 3;  // none/some/all

/**
 * Hash observation to a numeric key (fits in 53-bit safe integer).
 * Uses arithmetic packing instead of bitwise shifts because JavaScript
 * bitwise operators truncate to 32 bits.
 *
 * Field order (low → high): wallMask, pelletDir, edibleBucket, ghost0, ghost1.
 *
 * Key version 3 changes vs v2:
 *   - wallMask: 25-bit 5×5 window (33 M values) → 4-bit cardinal neighbors (16 values)
 */
export const observationKey = (obs: Observation): number => {
  let key = obs.wallMask;
  let place = WALL_MASK_BASE;

  key += obs.nearestPelletDir * place;
  place *= PELLET_DIR_BASE;

  key += obs.numEdibleBucket * place;
  place *= EDIBLE_BUCKET_BASE;

  key += obs.ghostCodes[0] * place;
  place *= GHOST_ZONE_BASE;

  key += obs.ghostCodes[1] * place;

  return key;
};

/**
 * Reconstruct a string representation of the key for serialization.
 * Format: "v3:wallMask:pelletDir:edibleBucket:gc0:gc1"
 */
export const observationKeyToString = (key: number): string => {
  const wallMask = key % WALL_MASK_BASE;
  let rest = Math.floor(key / WALL_MASK_BASE);
  const pelletDir = rest % PELLET_DIR_BASE;
  rest = Math.floor(rest / PELLET_DIR_BASE);
  const edibleBucket = rest % EDIBLE_BUCKET_BASE;
  rest = Math.floor(rest / EDIBLE_BUCKET_BASE);
  const gc0 = rest % GHOST_ZONE_BASE;
  const gc1 = Math.floor(rest / GHOST_ZONE_BASE) % GHOST_ZONE_BASE;
  return `v3:${wallMask}:${pelletDir}:${edibleBucket}:${gc0}:${gc1}`;
};
