import { DIR_VEC, DIRECTIONS, Direction, Vec2 } from '../engine/types';
import type { GhostState, WorldState } from '../env/environment';

export type GhostAIType = 'classic' | 'heatmap' | 'hybrid';

const manhattan = (a: Vec2, b: Vec2): number => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

const legalMoves = (world: WorldState, pos: Vec2): Direction[] =>
  DIRECTIONS.filter((d) => !world.isWall(pos.x + DIR_VEC[d].x, pos.y + DIR_VEC[d].y));

export const chooseGhostMove = (world: WorldState, ghost: GhostState, pacPos: Vec2): Direction => {
  const legal = legalMoves(world, ghost.pos);
  if (legal.length === 0) return 'up';

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
      const a = world.heatmap[ghost.pos.y + DIR_VEC[d].y][ghost.pos.x + DIR_VEC[d].x];
      const b = world.heatmap[ghost.pos.y + DIR_VEC[best].y][ghost.pos.x + DIR_VEC[best].x];
      return a > b ? d : best;
    }, legal[0]);
  }

  return legal.reduce((best, d) => {
    const next = { x: ghost.pos.x + DIR_VEC[d].x, y: ghost.pos.y + DIR_VEC[d].y };
    const heat = world.heatmap[next.y][next.x];
    const distScore = 1 / (1 + manhattan(next, pacPos));
    const score = distScore * 0.7 + heat * 0.3;
    const bestNext = { x: ghost.pos.x + DIR_VEC[best].x, y: ghost.pos.y + DIR_VEC[best].y };
    const bestScore = (1 / (1 + manhattan(bestNext, pacPos))) * 0.7 + world.heatmap[bestNext.y][bestNext.x] * 0.3;
    return score > bestScore ? d : best;
  }, legal[0]);
};
