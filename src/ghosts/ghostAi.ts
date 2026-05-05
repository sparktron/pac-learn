import { DIR_VEC, DIRECTIONS, Direction, Vec2 } from '../engine/types';
import type { GhostState, WorldState } from '../env/environment';

export type GhostAIType = 'classic' | 'heatmap' | 'hybrid';

const manhattan = (a: Vec2, b: Vec2): number => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

// Pre-allocated buffer and count to avoid array allocations per move
const legalBuffer: Direction[] = []; // Reused array
const legalMoves = (world: WorldState, pos: Vec2): number => {
  // Returns count of legal moves; fills legalBuffer
  legalBuffer.length = 0;
  for (const d of DIRECTIONS) {
    if (!world.isWall(pos.x + DIR_VEC[d].x, pos.y + DIR_VEC[d].y)) {
      legalBuffer.push(d);
    }
  }
  return legalBuffer.length;
};

const safeHeat = (world: WorldState, x: number, y: number): number =>
  y >= 0 && y < world.height && x >= 0 && x < world.width ? world.heatmap[y][x] : 0;

export const chooseGhostMove = (world: WorldState, ghost: GhostState, pacPos: Vec2): Direction | null => {
  const legalCount = legalMoves(world, ghost.pos);
  if (legalCount === 0) return null;
  const legal = legalBuffer.slice(0, legalCount);

  if (ghost.aiType === 'classic') {
    return legal.reduce((best, d) => {
      const next = { x: ghost.pos.x + DIR_VEC[d].x, y: ghost.pos.y + DIR_VEC[d].y };
      const score = manhattan(next, pacPos);
      const bestNext = { x: ghost.pos.x + DIR_VEC[best].x, y: ghost.pos.y + DIR_VEC[best].y };
      return score < manhattan(bestNext, pacPos) ? d : best;
    }, legal[0]);
  }

  if (ghost.aiType === 'heatmap') {
    return legal.reduce((best, d) => {
      const a = safeHeat(world, ghost.pos.x + DIR_VEC[d].x, ghost.pos.y + DIR_VEC[d].y);
      const b = safeHeat(world, ghost.pos.x + DIR_VEC[best].x, ghost.pos.y + DIR_VEC[best].y);
      return a > b ? d : best;
    }, legal[0]);
  }

  return legal.reduce((best, d) => {
    const next = { x: ghost.pos.x + DIR_VEC[d].x, y: ghost.pos.y + DIR_VEC[d].y };
    const heat = safeHeat(world, next.x, next.y);
    const distScore = 1 / (1 + manhattan(next, pacPos));
    const score = distScore * 0.7 + heat * 0.3;
    const bestNext = { x: ghost.pos.x + DIR_VEC[best].x, y: ghost.pos.y + DIR_VEC[best].y };
    const bestScore = (1 / (1 + manhattan(bestNext, pacPos))) * 0.7 + safeHeat(world, bestNext.x, bestNext.y) * 0.3;
    return score > bestScore ? d : best;
  }, legal[0]);
};
