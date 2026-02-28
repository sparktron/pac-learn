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

export const observationKey = (obs: Observation): string => {
  const ghosts = obs.ghostRel.map((g) => `${g.dx + 4},${g.dy + 4}`).join('|');
  return `${obs.wallMask}:${obs.nearestPelletDir}:${ghosts}`;
};
