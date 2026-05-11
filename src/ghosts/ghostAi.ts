import { DIR_VEC, DIRECTIONS, Direction, Vec2 } from '../engine/types';
import type { GhostState, WorldState, PacmanEnvironment } from '../env/environment';

export type GhostAIType = 'classic' | 'heatmap' | 'hybrid';

const manhattan = (a: Vec2, b: Vec2): number => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

const nextPosition = (world: WorldState, pos: Vec2, d: Direction): Vec2 => {
  const next = { x: pos.x + DIR_VEC[d].x, y: pos.y + DIR_VEC[d].y };
  if (next.x < 0) next.x = world.width - 1;
  if (next.x >= world.width) next.x = 0;
  return next;
};

// Pre-allocated buffer and count to avoid array allocations per move
const legalBuffer: Direction[] = []; // Reused array
const legalMoves = (world: WorldState, pos: Vec2): number => {
  // Returns count of legal moves; fills legalBuffer
  legalBuffer.length = 0;
  for (const d of DIRECTIONS) {
    const next = nextPosition(world, pos, d);
    if (!world.isWall(next.x, next.y)) {
      legalBuffer.push(d);
    }
  }
  return legalBuffer.length;
};

const safeHeat = (world: WorldState, x: number, y: number): number =>
  y >= 0 && y < world.height && x >= 0 && x < world.width ? world.heatmap[y][x] : 0;

export const chooseGhostMove = (world: WorldState, ghost: GhostState, pacPos: Vec2, env?: PacmanEnvironment): Direction | null => {
  const legalCount = legalMoves(world, ghost.pos);
  if (legalCount === 0) return null;
  const legal = legalBuffer.slice(0, legalCount);

  if (ghost.aiType === 'classic') {
    // In scatter phase, chase the corner; in chase phase, chase Pac-Man
    const target = env && env.isScatterPhase() ? env.getScatterTarget(ghost.id, world.width, world.height) : pacPos;
    return legal.reduce((best, d) => {
      const next = nextPosition(world, ghost.pos, d);
      const score = manhattan(next, target);
      const bestNext = nextPosition(world, ghost.pos, best);
      return score < manhattan(bestNext, target) ? d : best;
    }, legal[0]);
  }

  if (ghost.aiType === 'heatmap') {
    return legal.reduce((best, d) => {
      const next = nextPosition(world, ghost.pos, d);
      const bestNext = nextPosition(world, ghost.pos, best);
      const a = safeHeat(world, next.x, next.y);
      const b = safeHeat(world, bestNext.x, bestNext.y);
      return a > b ? d : best;
    }, legal[0]);
  }

  // Hybrid mode (with scatter support)
  const target = env && env.isScatterPhase() ? env.getScatterTarget(ghost.id, world.width, world.height) : pacPos;
  return legal.reduce((best, d) => {
    const next = nextPosition(world, ghost.pos, d);
    const heat = safeHeat(world, next.x, next.y);
    const distScore = 1 / (1 + manhattan(next, target));
    const score = distScore * 0.7 + heat * 0.3;
    const bestNext = nextPosition(world, ghost.pos, best);
    const bestScore = (1 / (1 + manhattan(bestNext, target))) * 0.7 + safeHeat(world, bestNext.x, bestNext.y) * 0.3;
    return score > bestScore ? d : best;
  }, legal[0]);
};
