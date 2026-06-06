import { DIR_VEC, wrapPosition, type Direction, type Vec2 } from '../engine/types';
import type { WorldState } from './environment';

export interface Observation {
  pac: Vec2;
  ghosts: Vec2[];
  wallMask: number;
  /** 0=up, 1=down, 2=left, 3=right, 4=no pellet reachable within search radius.
   *  Matches the DIRECTIONS action-space ordering so nearestPelletDir=k means "take action k". */
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
   * Per-slot heading of each nearest ghost relative to Pac-Man:
   *   0 = unknown / stationary / perpendicular  (no lastDir, or velocity ⊥ to displacement)
   *   1 = approaching                            (ghost's velocity has positive dot with ghost→pac)
   *   2 = receding                               (ghost's velocity has negative dot)
   *
   * Breaks state aliasing where "ghost in zone X" was identical regardless of
   * whether the ghost was closing in or walking away — the agent could not
   * learn to flee chasers vs. hold position against retreating ghosts.
   */
  ghostHeadings: [number, number];
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
  /**
   * Coarse bucket of power pellets still on the board:
   *   0 = none  (cannot make ghosts edible — must dodge)
   *   1 = one   (save it for the endgame cluster, or use to make a chase opportunity)
   *   2 = many  (can afford to consume one early)
   *
   * Critical for endgame survival: "1 power pellet left" tells the agent it has
   * one more chance to safely traverse a ghost-clustered zone.
   */
  powerPelletsLeftBucket: number;
}

// ─── Key version ─────────────────────────────────────────────────────────────
// v7: adds powerPelletsLeftBucket (3 buckets) to the key.
// v8: adds per-ghost heading codes (3 values each) for the two nearest ghosts,
//     so the agent can distinguish approaching from receding ghosts.
// v9: aligns nearestPelletDir indices with the DIRECTIONS action-space ordering
//     (up=0, down=1, left=2, right=3) so nearestPelletDir=k means "take action k".
//     Previously DIRS used rotational order (up=0, right=1, down=2, left=3), which
//     caused a mismatch between the observation feature and the action index.
export const OBSERVATION_KEY_VERSION = 9;

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

/** Convert powerPelletsLeft count → bucket 0–2. */
export const powerPelletsLeftBucket = (n: number): number => (n <= 0 ? 0 : n === 1 ? 1 : 2);
export const POWER_PELLETS_BUCKET_BASE = 3;

// ─── Internal helpers ────────────────────────────────────────────────────────

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

// Pellet-direction encoding matches DIRECTIONS action-space: up=0, down=1, left=2, right=3.
// nearestPelletDir=k means "take action k to reach the nearest pellet".
const DIRS: Array<{ dx: number; dy: number }> = [
  { dx: 0, dy: -1 }, // 0 = up
  { dx: 0, dy: 1 },  // 1 = down
  { dx: -1, dy: 0 }, // 2 = left
  { dx: 1, dy: 0 },  // 3 = right
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
    // D4.7: shared x-only tunnel wrap (y passes through, still bounds-checked).
    const { x: nx, y: ny } = wrapPosition(w, h, pac.x + DIRS[i].dx, pac.y + DIRS[i].dy);
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
      const { x: nx, y: ny } = wrapPosition(w, h, cur.x + DIRS[i].dx, cur.y + DIRS[i].dy);
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
/**
 * Encode one ghost slot's heading relative to Pac-Man:
 *   0 = unknown / stationary / perpendicular (no lastDir, or velocity ⊥ to displacement)
 *   1 = approaching (velocity · ghost→pac > 0)
 *   2 = receding    (velocity · ghost→pac < 0)
 *
 * Uses tunnel-wrapped x-displacement so a ghost stepping through a side tunnel
 * is correctly classified as approaching when that's the shortest path.
 */
export const encodeGhostHeading = (
  g: Vec2 | undefined,
  pac: Vec2,
  worldWidth: number,
  lastDir: Direction | null,
): number => {
  if (!g || !lastDir) return 0;
  let dx = pac.x - g.x;
  if (dx > worldWidth / 2) dx -= worldWidth;
  else if (dx < -worldWidth / 2) dx += worldWidth;
  const dy = pac.y - g.y;
  const v = DIR_VEC[lastDir];
  const dot = v.x * dx + v.y * dy;
  if (dot > 0) return 1;
  if (dot < 0) return 2;
  return 0;
};

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
  powerPelletsLeft: number = 0,
  ghostsLastDir: Array<Direction | null> = [],
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
      return {
        g,
        edible: edibleFlags[i] ?? false,
        lastDir: ghostsLastDir[i] ?? null,
        dist: Math.abs(dx) + Math.abs(g.y - pac.y),
      };
    })
    .sort((a, b) => a.dist - b.dist);

  const ghostCodes: [number, number] = [
    encodeGhostZone(sorted[0]?.g, pac, world.width, sorted[0]?.edible ?? false),
    encodeGhostZone(sorted[1]?.g, pac, world.width, sorted[1]?.edible ?? false),
  ];

  const ghostHeadings: [number, number] = [
    encodeGhostHeading(sorted[0]?.g, pac, world.width, sorted[0]?.lastDir ?? null),
    encodeGhostHeading(sorted[1]?.g, pac, world.width, sorted[1]?.lastDir ?? null),
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
    ghostHeadings,
    lastAction,
    pelletsRemainingBucket: pelletsRemainingBucket(pelletsLeft, totalPellets),
    powerPelletsLeftBucket: powerPelletsLeftBucket(powerPelletsLeft),
  };
};

// ─── Key encoding ────────────────────────────────────────────────────────────

const GHOST_ZONE_BASE    = 19; // 0=absent, 1–18 = zone 1–9 × 2 edibility states
const GHOST_HEADING_BASE = 3;  // 0=unknown/perpendicular, 1=approaching, 2=receding
const WALL_MASK_BASE     = 16; // 4-bit cardinal wall mask = 16 values
const PELLET_DIR_BASE    = 5;  // up/right/down/left/none
const LAST_ACTION_BASE   = 5;  // -1=none (episode start) + 0-3, encoded as +1 → 0-4

/**
 * Hash observation to a numeric key (fits in 53-bit safe integer).
 * Uses arithmetic packing — JS bitwise ops truncate to 32 bits.
 *
 * Field order (low → high): wallMask, pelletDir, gc0, gh0, gc1, gh1,
 *                            lastAction, pelletsRemainingBucket, powerPelletsLeftBucket.
 *
 * Key version 9 aligns nearestPelletDir indices with the DIRECTIONS action-space
 * (up=0, down=1, left=2, right=3) so pelletDir=k means "take action k toward the pellet".
 * Pairs each ghost's zone code with a heading code so the agent can tell a chaser
 * from a retreating ghost in the same zone.
 *
 * State space: 16 × 5 × 19 × 3 × 19 × 3 × 5 × 5 × 3 = 19,494,000 theoretical maximum.
 * Observed populated states will be far smaller — most heading combinations
 * never co-occur with most wall/zone combinations.
 */
export const observationKey = (obs: Observation): number => {
  let key = obs.wallMask;
  let place = WALL_MASK_BASE;

  key += obs.nearestPelletDir * place;
  place *= PELLET_DIR_BASE;

  key += obs.ghostCodes[0] * place;
  place *= GHOST_ZONE_BASE;

  key += obs.ghostHeadings[0] * place;
  place *= GHOST_HEADING_BASE;

  key += obs.ghostCodes[1] * place;
  place *= GHOST_ZONE_BASE;

  key += obs.ghostHeadings[1] * place;
  place *= GHOST_HEADING_BASE;

  key += (obs.lastAction + 1) * place; // shift -1→0, 0-3→1-4
  place *= LAST_ACTION_BASE;

  key += obs.pelletsRemainingBucket * place;
  place *= PELLETS_REMAINING_BUCKET_BASE;

  key += obs.powerPelletsLeftBucket * place;

  return key;
};

/**
 * Reconstruct a string representation of the key for serialization.
 * Format: "v9:wallMask:pelletDir:gc0:gh0:gc1:gh1:lastAction:pelletsBucket:powerBucket"
 * lastAction is stored as the raw value (-1 to 3) for human readability.
 */
export const observationKeyToString = (key: number): string => {
  const wallMask = key % WALL_MASK_BASE;
  let rest = Math.floor(key / WALL_MASK_BASE);
  const pelletDir = rest % PELLET_DIR_BASE;
  rest = Math.floor(rest / PELLET_DIR_BASE);
  const gc0 = rest % GHOST_ZONE_BASE;
  rest = Math.floor(rest / GHOST_ZONE_BASE);
  const gh0 = rest % GHOST_HEADING_BASE;
  rest = Math.floor(rest / GHOST_HEADING_BASE);
  const gc1 = rest % GHOST_ZONE_BASE;
  rest = Math.floor(rest / GHOST_ZONE_BASE);
  const gh1 = rest % GHOST_HEADING_BASE;
  rest = Math.floor(rest / GHOST_HEADING_BASE);
  const lastAction = (rest % LAST_ACTION_BASE) - 1; // decode +1 shift
  rest = Math.floor(rest / LAST_ACTION_BASE);
  const pelletsBucket = rest % PELLETS_REMAINING_BUCKET_BASE;
  const powerBucket = Math.floor(rest / PELLETS_REMAINING_BUCKET_BASE) % POWER_PELLETS_BUCKET_BASE;
  return `v${OBSERVATION_KEY_VERSION}:${wallMask}:${pelletDir}:${gc0}:${gh0}:${gc1}:${gh1}:${lastAction}:${pelletsBucket}:${powerBucket}`;
};

/**
 * Inverse of observationKeyToString: parse a serialized "v9:…" key string back
 * to the numeric key, using the SAME base constants as observationKey() so the
 * two can never drift apart (D5.10). Returns null for a wrong version, wrong
 * field count, or any non-numeric field — callers skip such entries on load.
 *
 * Previously qlearning.load() re-implemented this packing with the base
 * constants hardcoded inline; the v8→v9 bump had to be mirrored there by hand.
 * Centralizing it here makes the next key-layout change a single-site edit.
 */
export const stringToObservationKey = (keyStr: string): number | null => {
  const parts = keyStr.split(':');
  if (parts[0] !== `v${OBSERVATION_KEY_VERSION}` || parts.length !== 10) return null;
  const nums = parts.slice(1).map((p) => parseInt(p, 10));
  if (nums.some((n) => Number.isNaN(n))) return null;
  const [wallMask, pelletDir, gc0, gh0, gc1, gh1, lastAction, pelletsBucket, powerBucket] = nums;

  let key = wallMask;
  let place = WALL_MASK_BASE;
  key += pelletDir * place;        place *= PELLET_DIR_BASE;
  key += gc0 * place;              place *= GHOST_ZONE_BASE;
  key += gh0 * place;              place *= GHOST_HEADING_BASE;
  key += gc1 * place;              place *= GHOST_ZONE_BASE;
  key += gh1 * place;              place *= GHOST_HEADING_BASE;
  key += (lastAction + 1) * place; place *= LAST_ACTION_BASE; // mirror the +1 shift
  key += pelletsBucket * place;    place *= PELLETS_REMAINING_BUCKET_BASE;
  key += powerBucket * place;
  return key;
};
