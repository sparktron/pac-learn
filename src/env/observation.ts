import type { Vec2 } from '../engine/types';
import type { WorldState } from './environment';

export interface Observation {
  pac: Vec2;
  ghosts: Vec2[];
  wallMask: number;
  nearestPelletDir: number;
  ghostRel: Array<{ dx: number; dy: number }>;
}

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

export const encodeObservation = (world: WorldState, pac: Vec2, ghosts: Vec2[]): Observation => {
  let bit = 0;
  let mask = 0;
  for (let dy = -2; dy <= 2; dy += 1) {
    for (let dx = -2; dx <= 2; dx += 1) {
      if (world.isWall(pac.x + dx, pac.y + dy)) mask |= (1 << bit);
      bit += 1;
    }
  }
  const dirs = [
    { dx: 0, dy: -1 },
    { dx: 1, dy: 0 },
    { dx: 0, dy: 1 },
    { dx: -1, dy: 0 },
  ];
  let bestDir = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  dirs.forEach((d, i) => {
    for (let k = 1; k < 8; k += 1) {
      const x = pac.x + d.dx * k;
      const y = pac.y + d.dy * k;
      if (world.isWall(x, y)) break;
      if (world.pellets[y]?.[x] || world.powerPellets[y]?.[x]) {
        if (k < bestDist) {
          bestDist = k;
          bestDir = i;
        }
        break;
      }
    }
  });

  return {
    pac,
    ghosts,
    wallMask: mask,
    nearestPelletDir: bestDir,
    ghostRel: ghosts.map((g) => ({ dx: clamp(g.x - pac.x, -4, 4), dy: clamp(g.y - pac.y, -4, 4) })),
  };
};

const GHOST_OFFSET_BASE = 7;
const GHOST_BITS_BASE = GHOST_OFFSET_BASE * GHOST_OFFSET_BASE;
const WALL_MASK_BASE = 2 ** 25;
const PELLET_DIR_BASE = 4;

/**
 * Hash observation to a numeric key (fits in 53-bit safe integer).
 * Uses arithmetic packing instead of bitwise shifts because JavaScript
 * bitwise operators truncate to 32 bits.
 */
export const observationKey = (obs: Observation): number => {
  let key = obs.wallMask;
  let place = WALL_MASK_BASE;

  key += obs.nearestPelletDir * place;
  place *= PELLET_DIR_BASE;

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
 */
export const observationKeyToString = (key: number): string => {
  const wallMask = key % WALL_MASK_BASE;
  let rest = Math.floor(key / WALL_MASK_BASE);
  const pelletDir = rest % PELLET_DIR_BASE;
  rest = Math.floor(rest / PELLET_DIR_BASE);

  let s = `${wallMask}:${pelletDir}`;
  for (let i = 0; i < 4; i++) {
    const bits = rest % GHOST_BITS_BASE;
    rest = Math.floor(rest / GHOST_BITS_BASE);
    const dx = Math.floor(bits / GHOST_OFFSET_BASE) - 3;
    const dy = (bits % GHOST_OFFSET_BASE) - 3;
    s += `:${dx},${dy}`;
  }

  return s;
};
