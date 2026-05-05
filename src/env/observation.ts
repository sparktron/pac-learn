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

/**
 * Hash observation to a numeric key (fits in 53-bit safe integer).
 * Bits 0-24: wallMask (25 bits)
 * Bits 25-26: nearestPelletDir (2 bits)
 * Bits 27-52: ghost offsets for up to 4 ghosts (6 bits per ghost, 3 bits per offset with clamp to ±3)
 */
export const observationKey = (obs: Observation): number => {
  let key = obs.wallMask;
  key |= (obs.nearestPelletDir << 25);

  // Encode up to 4 ghosts with ±3 clamp (3 bits per offset: 0-7 after adding 3)
  for (let i = 0; i < Math.min(4, obs.ghostRel.length); i++) {
    const g = obs.ghostRel[i];
    const dx = Math.max(0, Math.min(6, g.dx + 3));
    const dy = Math.max(0, Math.min(6, g.dy + 3));
    const bits = (dx << 3) | dy;
    key |= (bits << (27 + i * 6));
  }

  return key;
};

/**
 * Reconstruct a string representation of the key for debugging/serialization.
 */
export const observationKeyToString = (key: number): string => {
  const wallMask = key & 0x1FFFFFF;
  const pelletDir = (key >> 25) & 0x3;
  let s = `${wallMask}:${pelletDir}`;

  for (let i = 0; i < 4; i++) {
    const bits = (key >> (27 + i * 6)) & 0x3F;
    const dx = ((bits >> 3) & 0x7) - 3;
    const dy = (bits & 0x7) - 3;
    s += `:${dx},${dy}`;
  }

  return s;
};
