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
  /** True when at least one ghost is currently edible. Kept for rendering. */
  ghostsEdible: boolean;
  /**
   * Compact zone+edibility code for the two nearest active ghosts (sorted by Manhattan distance).
   *
   *   0           = absent (slot unused)
   *   odd  1,3,5… = dangerous ghost: zone 1–9, not edible  → code = (zone-1)*2 + 1
   *   even 2,4,6… = edible ghost:    zone 1–9, edible      → code = (zone-1)*2 + 2
   *
   * Zone mapping (tunnel-aware Manhattan distance):
   *   zone 1 = here/adjacent  (dist ≤ 1)
   *   zone 2 = mid-up         (dist 2–5, dy dominates, dy < 0)
   *   zone 3 = mid-right      (dist 2–5, dx dominates, dx > 0)
   *   zone 4 = mid-down       (dist 2–5, dy dominates, dy > 0)
   *   zone 5 = mid-left       (dist 2–5, dx dominates, dx < 0)
   *   zone 6 = far-up         (dist 6+,  dy dominates, dy < 0)
   *   zone 7 = far-right      (dist 6+,  dx dominates, dx > 0)
   *   zone 8 = far-down       (dist 6+,  dy dominates, dy > 0)
   *   zone 9 = far-left       (dist 6+,  dx dominates, dx < 0)
   *
   * Combining zone and edibility in a single slot means the agent can distinguish
   * "nearby ghost I should eat" from "nearby ghost I should flee" without needing
   * a separate global edibility feature.
   */
  ghostCodes: [number, number];
  /**
   * Action taken to arrive at the current state.
   * -1 = episode start (no previous action), 0–3 = up/right/down/left.
   * Breaks state aliasing between opposite movement directions so the agent can
   * learn "don't reverse toward the ghost" and avoid two-tile oscillation loops.
   */
  lastAction: number;
  /**
   * Coarse bucket of pellets remaining, encoded as fraction of starting pellets:
   *   0 = endgame   (0–10% pellets left — last pellets, often ghost-adjacent)
   *   1 = late      (10–25%)
   *   2 = mid       (25–50%)
   *   3 = early     (50–75%)
   *   4 = opening   (75–100%)
   *
   * Lets the agent learn different policies per game phase. With pellet-escalation
   * rewards, the agent needs to know it's near the end to take riskier paths.
   */
  pelletsRemainingBucket: number;
}

// ─── Key version ─────────────────────────────────────────────────────────────
// v6: adds pelletsRemainingBucket (5 buckets) to the key.
export const OBSERVATION_KEY_VERSION = 6;

/**
 * Convert (pelletsLeft, totalPellets) → bucket 0–4. Total=0 returns 0
 * (treats "all gone" as endgame, consistent with the win condition).
 */
export const pelletsRemainingBucket = (pelletsLeft: number, totalPellets: number): number => {
  if (totalPellets <= 0) return 0;
  const frac = pelletsLeft / totalPellets;
  if (frac <= 0.10) return 0;
  if (frac <= 0.25) return 1;
  if (frac <= 0.50) return 2;
  if (frac <= 0.75) return 3;
  return 4;
};
export const PELLETS_REMAINING_BUCKET_BASE = 5;

// ─── Internal helpers ────────────────────────────────────────────────────────

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

// Pellet-direction encoding: up=0, right=1, down=2, left=3 (rotational order).
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
 * (the "none" sentinel) if no pellet is reachable in radius.
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

/** Tunnel-aware direction quadrant: 0=up, 1=right, 2=down, 3=left. */
const ghostQuadrant = (dx: number, dy: number): number =>
  Math.abs(dy) >= Math.abs(dx) ? (dy < 0 ? 0 : 2) : (dx > 0 ? 1 : 3);

/**
 * Encode one ghost slot into a zone+edibility code (0–18).
 * Pass `undefined` for absent slots to get code 0.
 *
 *   0           = absent
 *   (zone-1)*2 + 1  = zone, not edible (dangerous)
 *   (zone-1)*2 + 2  = zone, edible (chase opportunity)
 */
export const encodeGhostZone = (g: Vec2 | undefined, pac: Vec2, worldWidth: number, edible = false): number => {
  if (!g) return 0;
  let dx = g.x - pac.x;
  if (dx > worldWidth / 2) dx -= worldWidth;
  else if (dx < -worldWidth / 2) dx += worldWidth;
  const dy = g.y - pac.y;
  const dist = Math.abs(dx) + Math.abs(dy);
  const zone = dist <= 1 ? 1 : dist <= 5 ? 2 + ghostQuadrant(dx, dy) : 6 + ghostQuadrant(dx, dy);
  return (zone - 1) * 2 + (edible ? 1 : 0) + 1;
};

// ─── Public API ──────────────────────────────────────────────────────────────

export const encodeObservation = (
  world: WorldState,
  pac: Vec2,
  ghosts: Vec2[],
  edibleFlags: boolean[] = [],
  lastAction: number = -1,
  pelletsLeft: number = 0,
  totalPellets: number = 0,
): Observation => {
  // 4-bit cardinal wall mask (N/E/S/W → bits 0-3). 16 values covers every
  // junction shape a Pac-Man maze can produce.
  const CARD = [{ dx: 0, dy: -1 }, { dx: 1, dy: 0 }, { dx: 0, dy: 1 }, { dx: -1, dy: 0 }];
  let mask = 0;
  CARD.forEach(({ dx, dy }, i) => {
    if (world.isWall(pac.x + dx, pac.y + dy)) mask |= (1 << i);
  });

  // Sort ghosts by tunnel-aware Manhattan distance; take nearest two.
  const sorted = ghosts
    .map((g, i) => {
      let dx = g.x - pac.x;
      if (dx > world.width / 2) dx -= world.width;
      else if (dx < -world.width / 2) dx += world.width;
      return { g, edible: edibleFlags[i] ?? false, dist: Math.abs(dx) + Math.abs(g.y - pac.y) };
    })
    .sort((a, b) => a.dist - b.dist);

  const ghostCodes: [number, number] = [
    encodeGhostZone(sorted[0]?.g, pac, world.width, sorted[0]?.edible ?? false),
    encodeGhostZone(sorted[1]?.g, pac, world.width, sorted[1]?.edible ?? false),
  ];

  const ghostsEdible = edibleFlags.some(Boolean);

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
    lastAction,
    pelletsRemainingBucket: pelletsRemainingBucket(pelletsLeft, totalPellets),
  };
};

// ─── Key encoding ────────────────────────────────────────────────────────────

const GHOST_ZONE_BASE  = 19; // 0=absent, 1–18 = zone 1–9 × 2 edibility states
const WALL_MASK_BASE   = 16; // 4-bit cardinal wall mask = 16 values
const PELLET_DIR_BASE  = 5;  // up/right/down/left/none
const LAST_ACTION_BASE = 5;  // -1=none (episode start) + 0-3 (up/right/down/left), encoded as +1 → 0-4

/**
 * Hash observation to a numeric key (fits in 53-bit safe integer).
 * Uses arithmetic packing — JS bitwise ops truncate to 32 bits.
 *
 * Field order (low → high): wallMask, pelletDir, ghost0, ghost1, lastAction, pelletsRemainingBucket.
 *
 * Key version 6 adds pelletsRemainingBucket (5 buckets) as the highest field.
 * This gives the agent endgame awareness — distinguishes "opening: lots of pellets,
 * play safe" from "endgame: 3 pellets left near ghosts, take the risk."
 *
 * State space: 16 × 5 × 19 × 19 × 5 × 5 = 722,000 theoretical maximum.
 */
export const observationKey = (obs: Observation): number => {
  let key = obs.wallMask;
  let place = WALL_MASK_BASE;

  key += obs.nearestPelletDir * place;
  place *= PELLET_DIR_BASE;

  key += obs.ghostCodes[0] * place;
  place *= GHOST_ZONE_BASE;

  key += obs.ghostCodes[1] * place;
  place *= GHOST_ZONE_BASE;

  key += (obs.lastAction + 1) * place; // shift -1→0, 0-3→1-4
  place *= LAST_ACTION_BASE;

  key += obs.pelletsRemainingBucket * place;

  return key;
};

/**
 * Reconstruct a string representation of the key for serialization.
 * Format: "v6:wallMask:pelletDir:gc0:gc1:lastAction:pelletsBucket"
 * lastAction is stored as the raw value (-1 to 3) for human readability.
 */
export const observationKeyToString = (key: number): string => {
  const wallMask = key % WALL_MASK_BASE;
  let rest = Math.floor(key / WALL_MASK_BASE);
  const pelletDir = rest % PELLET_DIR_BASE;
  rest = Math.floor(rest / PELLET_DIR_BASE);
  const gc0 = rest % GHOST_ZONE_BASE;
  rest = Math.floor(rest / GHOST_ZONE_BASE);
  const gc1 = rest % GHOST_ZONE_BASE;
  rest = Math.floor(rest / GHOST_ZONE_BASE);
  const lastAction = (rest % LAST_ACTION_BASE) - 1; // decode +1 shift
  const pelletsBucket = Math.floor(rest / LAST_ACTION_BASE) % PELLETS_REMAINING_BUCKET_BASE;
  return `v6:${wallMask}:${pelletDir}:${gc0}:${gc1}:${lastAction}:${pelletsBucket}`;
};
