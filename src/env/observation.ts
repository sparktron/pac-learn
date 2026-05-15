import type { Vec2 } from '../engine/types';
import type { WorldState } from './environment';

export interface Observation {
  pac: Vec2;
  ghosts: Vec2[];
  wallMask: number;
  /** 0=up, 1=right, 2=down, 3=left, 4=no pellet reachable within search radius */
  nearestPelletDir: number;
  ghostRel: Array<{ dx: number; dy: number }>;
  /** True when at least one ghost is currently edible (any power pellet active). */
  ghostsEdible: boolean;
}

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
  if (world.pellets[pac.y]?.[pac.x] || world.powerPellets[pac.y]?.[pac.x]) return 4;
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

export const encodeObservation = (
  world: WorldState,
  pac: Vec2,
  ghosts: Vec2[],
  ghostsEdible = false,
): Observation => {
  let bit = 0;
  let mask = 0;
  for (let dy = -2; dy <= 2; dy += 1) {
    for (let dx = -2; dx <= 2; dx += 1) {
      if (world.isWall(pac.x + dx, pac.y + dy)) mask |= (1 << bit);
      bit += 1;
    }
  }
  return {
    pac,
    ghosts,
    wallMask: mask,
    nearestPelletDir: bfsPelletDir(world, pac),
    ghostRel: ghosts.map((g) => {
      const w = world.width;
      let dx = g.x - pac.x;
      // Wrap dx through the tunnel if the short path crosses an edge.
      if (dx > w / 2) dx -= w;
      else if (dx < -w / 2) dx += w;
      return { dx: clamp(dx, -3, 3), dy: clamp(g.y - pac.y, -3, 3) };
    }),
    ghostsEdible,
  };
};

const GHOST_OFFSET_BASE = 7;
const GHOST_BITS_BASE = GHOST_OFFSET_BASE * GHOST_OFFSET_BASE;
const WALL_MASK_BASE = 2 ** 25;
const PELLET_DIR_BASE = 5; // up/right/down/left/none
const EDIBLE_BASE = 2;

/**
 * Hash observation to a numeric key (fits in 53-bit safe integer).
 * Uses arithmetic packing instead of bitwise shifts because JavaScript
 * bitwise operators truncate to 32 bits.
 *
 * Field order (low → high): wallMask, pelletDir, ghostsEdible, ghost0..ghost3.
 */
export const observationKey = (obs: Observation): number => {
  let key = obs.wallMask;
  let place = WALL_MASK_BASE;

  key += obs.nearestPelletDir * place;
  place *= PELLET_DIR_BASE;

  key += (obs.ghostsEdible ? 1 : 0) * place;
  place *= EDIBLE_BASE;

  for (let i = 0; i < 4; i++) {
    const g = obs.ghostRel[i] ?? { dx: 0, dy: 0 };
    const dx = Math.max(0, Math.min(6, g.dx + 3));
    const dy = Math.max(0, Math.min(6, g.dy + 3));
    key += (dx * GHOST_OFFSET_BASE + dy) * place;
    place *= GHOST_BITS_BASE;
  }

  return key;
};

/**
 * Reconstruct a string representation of the key for debugging/serialization.
 * Format: wallMask:pelletDir:edible:dx0,dy0:dx1,dy1:dx2,dy2:dx3,dy3
 */
export const observationKeyToString = (key: number): string => {
  const wallMask = key % WALL_MASK_BASE;
  let rest = Math.floor(key / WALL_MASK_BASE);
  const pelletDir = rest % PELLET_DIR_BASE;
  rest = Math.floor(rest / PELLET_DIR_BASE);
  const edible = rest % EDIBLE_BASE;
  rest = Math.floor(rest / EDIBLE_BASE);

  let s = `${wallMask}:${pelletDir}:${edible}`;
  for (let i = 0; i < 4; i++) {
    const bits = rest % GHOST_BITS_BASE;
    rest = Math.floor(rest / GHOST_BITS_BASE);
    const dx = Math.floor(bits / GHOST_OFFSET_BASE) - 3;
    const dy = (bits % GHOST_OFFSET_BASE) - 3;
    s += `:${dx},${dy}`;
  }

  return s;
};
