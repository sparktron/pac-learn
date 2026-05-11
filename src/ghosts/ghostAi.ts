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

const safeHeat = (world: WorldState, x: number, y: number): number =>
  y >= 0 && y < world.height && x >= 0 && x < world.width ? world.heatmap[y][x] : 0;

// BFS to find the first step toward a target.
// extraWall: additional tiles to treat as impassable (e.g. ghost house for free ghosts).
const bfsFirstStep = (
  world: WorldState,
  from: Vec2,
  to: Vec2,
  extraWall?: (x: number, y: number) => boolean,
): Direction | null => {
  if (from.x === to.x && from.y === to.y) return null;
  const key = (x: number, y: number) => y * world.width + x;
  const visited = new Uint8Array(world.width * world.height);
  visited[key(from.x, from.y)] = 1;
  const queue: Array<[number, number, Direction]> = [];
  for (const d of DIRECTIONS) {
    const next = nextPosition(world, from, d);
    if (!world.isWall(next.x, next.y) && !(extraWall?.(next.x, next.y)) && !visited[key(next.x, next.y)]) {
      visited[key(next.x, next.y)] = 1;
      queue.push([next.x, next.y, d]);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const [x, y, firstDir] = queue[head++];
    if (x === to.x && y === to.y) return firstDir;
    for (const d of DIRECTIONS) {
      const next = nextPosition(world, { x, y }, d);
      if (!world.isWall(next.x, next.y) && !(extraWall?.(next.x, next.y)) && !visited[key(next.x, next.y)]) {
        visited[key(next.x, next.y)] = 1;
        queue.push([next.x, next.y, firstDir]);
      }
    }
  }
  return null;
};

const getLegal = (world: WorldState, pos: Vec2, extraWall?: (x: number, y: number) => boolean): Direction[] =>
  DIRECTIONS.filter((d) => {
    const next = nextPosition(world, pos, d);
    return !world.isWall(next.x, next.y) && !(extraWall?.(next.x, next.y));
  });

export const chooseGhostMove = (world: WorldState, ghost: GhostState, pacPos: Vec2, env?: PacmanEnvironment): Direction | null => {
  // In-box ghosts navigate toward the ghost house exit (no ghost-house avoidance — they must pass through it).
  if (ghost.inBox) {
    const exit = world.ghostHouseExit;
    if (exit) {
      const legal = getLegal(world, ghost.pos);
      return bfsFirstStep(world, ghost.pos, exit) ?? legal[0] ?? null;
    }
    return null;
  }

  // Free ghosts avoid re-entering the ghost house.
  const avoidBox = world.isGhostHouse.bind(world);
  const legal = getLegal(world, ghost.pos, avoidBox);
  if (legal.length === 0) return null;

  if (ghost.aiType === 'classic') {
    const target = env && env.isScatterPhase() ? env.getScatterTarget(ghost.id, world.width, world.height) : pacPos;
    return bfsFirstStep(world, ghost.pos, target, avoidBox) ?? legal[0];
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

  // Hybrid: BFS toward target blended with heatmap
  const target = env && env.isScatterPhase() ? env.getScatterTarget(ghost.id, world.width, world.height) : pacPos;
  const bfsDir = bfsFirstStep(world, ghost.pos, target, avoidBox);
  if (bfsDir !== null && Math.random() < 0.7) return bfsDir;
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
