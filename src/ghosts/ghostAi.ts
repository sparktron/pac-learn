import { DIR_VEC, DIRECTIONS, Direction, Vec2, wrapPosition } from '../engine/types';
import type { GhostState, WorldState, PacmanEnvironment } from '../env/environment';

export type GhostAIType = 'classic' | 'heatmap' | 'hybrid';

const REVERSE: Record<Direction, Direction> = { up: 'down', down: 'up', left: 'right', right: 'left' };

// Classic Pac-Man tie-break order when multiple legal directions are equidistant
// from the target: up > left > down > right.
const TIE_PRIORITY: Direction[] = ['up', 'left', 'down', 'right'];

const manhattan = (a: Vec2, b: Vec2): number => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
const sqDist = (a: Vec2, b: Vec2): number => (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);

const nextPosition = (world: WorldState, pos: Vec2, d: Direction): Vec2 =>
  wrapPosition(world.width, world.height, pos.x + DIR_VEC[d].x, pos.y + DIR_VEC[d].y);

const safeHeat = (world: WorldState, x: number, y: number): number =>
  y >= 0 && y < world.height && x >= 0 && x < world.width ? world.heatmap[y][x] : 0;

const getLegal = (world: WorldState, pos: Vec2, extraWall?: (x: number, y: number) => boolean): Direction[] =>
  DIRECTIONS.filter((d) => {
    const next = nextPosition(world, pos, d);
    return !world.isWall(next.x, next.y) && !(extraWall?.(next.x, next.y));
  });

// BFS to find first step toward a target — used only inside the ghost house
// where the no-reverse, target-tile heuristic doesn't apply.
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

// Filter out the reverse of the ghost's last direction. If that leaves no
// options (dead end), allow the reverse rather than freezing.
const removeReverse = (legal: Direction[], lastDir: Direction | null): Direction[] => {
  if (!lastDir) return legal;
  const filtered = legal.filter((d) => d !== REVERSE[lastDir]);
  return filtered.length > 0 ? filtered : legal;
};

// Classic Pac-Man "look ahead one tile" target selection: pick the legal
// direction whose resulting tile minimizes squared distance to target,
// with a fixed up>left>down>right tie-break.
const chooseClassic = (
  world: WorldState,
  pos: Vec2,
  target: Vec2,
  candidates: Direction[],
): Direction => {
  let best = candidates[0];
  let bestDist = Infinity;
  let bestPrio = Infinity;
  for (const d of candidates) {
    const next = nextPosition(world, pos, d);
    const dist = sqDist(next, target);
    const prio = TIE_PRIORITY.indexOf(d);
    if (dist < bestDist || (dist === bestDist && prio < bestPrio)) {
      best = d;
      bestDist = dist;
      bestPrio = prio;
    }
  }
  return best;
};

// Per-personality target tile (Blinky/Pinky/Inky/Clyde, repeating for >4 ghosts).
const getChaseTarget = (
  ghost: GhostState,
  pacPos: Vec2,
  env: PacmanEnvironment | undefined,
  world: WorldState,
): Vec2 => {
  const role = ghost.id % 4;
  // Use the *desired* direction so Pinky/Inky still aim ahead of intent
  // when Pac-Man is wall-bumped (getPacLastDir freezes on illegal moves
  // and would otherwise leave them locked on a stale heading).
  const pacDir = env?.getPacDesiredDir() ?? 'left';
  const ahead = (n: number): Vec2 => ({
    x: pacPos.x + DIR_VEC[pacDir].x * n,
    y: pacPos.y + DIR_VEC[pacDir].y * n,
  });

  if (role === 0) {
    // Blinky: target Pac-Man directly.
    return pacPos;
  }
  if (role === 1) {
    // Pinky: 4 tiles ahead of Pac-Man.
    return ahead(4);
  }
  if (role === 2) {
    // Inky: vector from Blinky to (2 tiles ahead of Pac-Man), doubled.
    const pivot = ahead(2);
    const blinky = env?.getBlinkyPos() ?? pacPos;
    return { x: pivot.x + (pivot.x - blinky.x), y: pivot.y + (pivot.y - blinky.y) };
  }
  // Clyde: chase when far, scatter to corner when within 8 tiles.
  if (manhattan(ghost.pos, pacPos) > 8) return pacPos;
  return env?.getScatterTarget(ghost.id, world.width, world.height) ?? pacPos;
};

export const chooseGhostMove = (world: WorldState, ghost: GhostState, pacPos: Vec2, env?: PacmanEnvironment): Direction | null => {
  // In-box ghosts navigate toward the ghost house exit (BFS through the house tiles).
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

  // Mode change (chase<->scatter) forces an immediate reversal in classic Pac-Man.
  // pendingReverse is per-ghost so every ghost honors the mode flip; a prior
  // env-wide flag was consumed by ghost 0 only.
  let lastDir = ghost.lastDir;
  if (ghost.pendingReverse) {
    ghost.pendingReverse = false;
    if (lastDir) {
      const reverse = REVERSE[lastDir];
      if (legal.includes(reverse)) return reverse;
      lastDir = null;
    }
  }

  const candidates = removeReverse(legal, lastDir);

  // Edible ghosts flee Pac-Man. Kept active (rather than classic random) so the
  // ghostEatReward signal stays learnable for RL — without it, ghosts close the
  // gap themselves and Pac-Man can't reliably benefit from a power pellet.
  if (ghost.edibleTimer > 0) {
    // D3.2: flee over the full legal set, not the reverse-filtered candidates.
    // Frightened ghosts may double back to escape (arcade-accurate) — and a
    // cornered ghost that can only reverse must still flee rather than stall.
    // This widens Pac's ghost-eat window after a power pellet, keeping the
    // ghostEatReward signal learnable. (Behavior change — see test_history.md.)
    return legal.reduce((best, d) => {
      const next = nextPosition(world, ghost.pos, d);
      const bestNext = nextPosition(world, ghost.pos, best);
      return manhattan(next, pacPos) > manhattan(bestNext, pacPos) ? d : best;
    }, legal[0]);
  }

  if (ghost.aiType === 'classic') {
    const target = env && env.isScatterPhase()
      ? env.getScatterTarget(ghost.id, world.width, world.height)
      : getChaseTarget(ghost, pacPos, env, world);
    return chooseClassic(world, ghost.pos, target, candidates);
  }

  if (ghost.aiType === 'heatmap') {
    return candidates.reduce((best, d) => {
      const next = nextPosition(world, ghost.pos, d);
      const bestNext = nextPosition(world, ghost.pos, best);
      return safeHeat(world, next.x, next.y) > safeHeat(world, bestNext.x, bestNext.y) ? d : best;
    }, candidates[0]);
  }

  // Hybrid: classic targeting blended with heatmap.
  const target = env && env.isScatterPhase()
    ? env.getScatterTarget(ghost.id, world.width, world.height)
    : getChaseTarget(ghost, pacPos, env, world);
  // N4: use the env's seeded RNG instead of Math.random() so hybrid ghost
  // decisions are reproducible across runs with the same seed. Falls back
  // to Math.random only if env wasn't passed in (test scaffolding paths).
  const rand = env ? env.nextRand() : Math.random();
  if (rand < 0.7) return chooseClassic(world, ghost.pos, target, candidates);
  return candidates.reduce((best, d) => {
    const next = nextPosition(world, ghost.pos, d);
    const heat = safeHeat(world, next.x, next.y);
    const distScore = 1 / (1 + manhattan(next, target));
    const score = distScore * 0.7 + heat * 0.3;
    const bestNext = nextPosition(world, ghost.pos, best);
    const bestScore = (1 / (1 + manhattan(bestNext, target))) * 0.7 + safeHeat(world, bestNext.x, bestNext.y) * 0.3;
    return score > bestScore ? d : best;
  }, candidates[0]);
};
