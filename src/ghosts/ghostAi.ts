import { DIR_VEC, DIRECTIONS, Direction, Vec2 } from '../engine/types';
import type { GhostState, WorldState, PacmanEnvironment } from '../env/environment';

export type GhostAIType = 'classic' | 'heatmap' | 'hybrid';

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
    const nx = from.x + DIR_VEC[d].x;
    const ny = from.y + DIR_VEC[d].y;
    if (!world.isWall(nx, ny) && !(extraWall?.(nx, ny)) && !visited[key(nx, ny)]) {
      visited[key(nx, ny)] = 1;
      queue.push([nx, ny, d]);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const [x, y, firstDir] = queue[head++];
    if (x === to.x && y === to.y) return firstDir;
    for (const d of DIRECTIONS) {
      const nx = x + DIR_VEC[d].x;
      const ny = y + DIR_VEC[d].y;
      if (!world.isWall(nx, ny) && !(extraWall?.(nx, ny)) && !visited[key(nx, ny)]) {
        visited[key(nx, ny)] = 1;
        queue.push([nx, ny, firstDir]);
      }
    }
  }
  return null;
};

const getLegal = (world: WorldState, pos: Vec2, extraWall?: (x: number, y: number) => boolean): Direction[] =>
  DIRECTIONS.filter((d) => {
    const nx = pos.x + DIR_VEC[d].x;
    const ny = pos.y + DIR_VEC[d].y;
    return !world.isWall(nx, ny) && !(extraWall?.(nx, ny));
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
      const a = safeHeat(world, ghost.pos.x + DIR_VEC[d].x, ghost.pos.y + DIR_VEC[d].y);
      const b = safeHeat(world, ghost.pos.x + DIR_VEC[best].x, ghost.pos.y + DIR_VEC[best].y);
      return a > b ? d : best;
    }, legal[0]);
  }

  // Hybrid: BFS toward target blended with heatmap
  const target = env && env.isScatterPhase() ? env.getScatterTarget(ghost.id, world.width, world.height) : pacPos;
  const bfsDir = bfsFirstStep(world, ghost.pos, target, avoidBox);
  if (bfsDir !== null && Math.random() < 0.7) return bfsDir;
  return legal.reduce((best, d) => {
    const a = safeHeat(world, ghost.pos.x + DIR_VEC[d].x, ghost.pos.y + DIR_VEC[d].y);
    const b = safeHeat(world, ghost.pos.x + DIR_VEC[best].x, ghost.pos.y + DIR_VEC[best].y);
    return a > b ? d : best;
  }, legal[0]);
};
