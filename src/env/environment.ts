import { chooseGhostMove, GhostAIType } from '../ghosts/ghostAi';
import { DIR_VEC, DIRECTIONS, Direction, actionToDirection, Vec2 } from '../engine/types';
import { SeededRng } from '../engine/prng';
import { MAZES } from '../mazes/mazes';
import { encodeObservation, type Observation } from './observation';

export interface EnvParams {
  mazeId: string;
  pelletDensity: number;
  numGhosts: number;
  ghostSpeed: number;
  pacmanSpeed: number;
  enablePowerPellets: boolean;
  powerPelletDuration: number;
  captureRules: 'touch' | 'tile';
  maxEpisodeSteps: number;
  reward: {
    pelletReward: number;
    powerPelletReward: number;
    deathPenalty: number;
    stepPenalty: number;
    survivalReward: number;
    ghostEatReward: number;
    winBonus: number;
  };
  heatmapDecayRate: number;
  heatmapLearningRate: number;
  illegalMoveMode: 'noop' | 'stay';
  cooperativePacmen: boolean;
  numPacmen: number;
  ghostReleaseInterval: number;
}

export interface GhostState { id: number; pos: { x: number; y: number }; aiType: GhostAIType; edibleTimer: number; releaseDelay: number; inBox: boolean; }
interface PacState { id: number; pos: { x: number; y: number }; score: number; lifetimeScore: number; }

export interface WorldState {
  width: number;
  height: number;
  pellets: boolean[][];
  powerPellets: boolean[][];
  heatmap: number[][];
  isWall(x: number, y: number): boolean;
  isGhostHouse(x: number, y: number): boolean;
  ghostHouseExit?: { x: number; y: number };
}

export interface StepResult { obs: Observation; reward: number; done: boolean; info: { score: number; lifetimeScore: number; pelletsLeft: number; step: number }; }

const defaultParams: EnvParams = {
  mazeId: 'pacman-classic', pelletDensity: 1, numGhosts: 2, ghostSpeed: 0.95, pacmanSpeed: 1,
  enablePowerPellets: true, powerPelletDuration: 20, captureRules: 'tile', maxEpisodeSteps: 400,
  reward: { pelletReward: 5, powerPelletReward: 20, deathPenalty: -100, stepPenalty: -0.1, survivalReward: 0.02, ghostEatReward: 30, winBonus: 200 },
  heatmapDecayRate: 0.997, heatmapLearningRate: 0.03, illegalMoveMode: 'stay', cooperativePacmen: true, numPacmen: 1,
  ghostReleaseInterval: 60,
};

export class PacmanEnvironment {
  params: EnvParams = structuredClone(defaultParams);
  private rng = new SeededRng(42);
  private maze = MAZES[0];
  private pacmen: PacState[] = [];
  ghosts: GhostState[] = [];
  pelletsLeft = 0;
  stepCount = 0;
  ghostsEatenCombo = 0;
  private scatterChaseCycle = 0; // 0 = chase, 1 = scatter
  private phaseDuration = 0;
  private phaseTimer = 0;
  world: WorldState = { width: 0, height: 0, pellets: [], powerPellets: [], heatmap: [], isWall: () => true, isGhostHouse: () => false };

  setParams(params: Partial<EnvParams>): void {
    this.params = { ...this.params, ...params, reward: { ...this.params.reward, ...(params.reward ?? {}) } };
  }

  setGhostType(index: number, type: GhostAIType): void {
    if (this.ghosts[index]) this.ghosts[index].aiType = type;
  }

  reset(seed = 42): Observation {
    this.rng = new SeededRng(seed);
    this.maze = MAZES.find((m) => m.id === this.params.mazeId) ?? MAZES[0];
    const { grid } = this.maze;
    const h = grid.length;
    const w = grid[0].length;
    // Use maze-defined power pellet positions (avoids placing them on walls).
    const powerPositions = this.params.enablePowerPellets
      ? (this.maze.powerPelletPositions ?? []).filter((p) => grid[p.y]?.[p.x] === 0)
      : [];
    const power = Array.from({ length: h }, () => Array.from({ length: w }, () => false));
    powerPositions.forEach((p) => { power[p.y][p.x] = true; });
    const pellets = Array.from({ length: h }, (_, y) =>
      Array.from({ length: w }, (_, x) => {
        if (power[y][x]) return false;
        return grid[y][x] === 0 && this.rng.next() < this.params.pelletDensity;
      }),
    );
    const hasGhostHouse = this.maze.ghostHouseExit !== undefined;
    const ghostHouseTiles = new Set<string>();
    const addGhostHouseTile = (x: number, y: number): void => {
      if (y < 0 || x < 0 || y >= h || x >= w || grid[y][x] !== 2) return;
      const key = `${x},${y}`;
      if (ghostHouseTiles.has(key)) return;
      ghostHouseTiles.add(key);
      addGhostHouseTile(x + 1, y);
      addGhostHouseTile(x - 1, y);
      addGhostHouseTile(x, y + 1);
      addGhostHouseTile(x, y - 1);
    };
    this.maze.ghostStarts.forEach((p) => addGhostHouseTile(p.x, p.y));
    this.world = {
      width: w,
      height: h,
      pellets,
      powerPellets: power,
      heatmap: Array.from({ length: h }, () => Array.from({ length: w }, () => 0)),
      isWall: (x, y) => y < 0 || x < 0 || y >= h || x >= w || grid[y][x] === 1,
      isGhostHouse: (x, y) => ghostHouseTiles.has(`${x},${y}`),
      ghostHouseExit: this.maze.ghostHouseExit,
    };
    this.pacmen = Array.from({ length: this.params.numPacmen }, (_, i) => ({ id: i, pos: { ...this.maze.pacStart }, score: 0, lifetimeScore: 0 }));
    this.ghosts = Array.from({ length: this.params.numGhosts }, (_, i) => ({
      id: i,
      pos: { ...this.maze.ghostStarts[i % this.maze.ghostStarts.length] },
      aiType: 'classic',
      edibleTimer: 0,
      inBox: hasGhostHouse,
      releaseDelay: i * this.params.ghostReleaseInterval,
    }));
    this.pelletsLeft = pellets.flat().filter(Boolean).length + power.flat().filter(Boolean).length;
    // Pac-Man starts on this tile, so consume any pellet/power-pellet there immediately
    // to avoid awarding free points and a misleading "no nearby pellet" first observation.
    const ps = this.maze.pacStart;
    if (pellets[ps.y]?.[ps.x]) { pellets[ps.y][ps.x] = false; this.pelletsLeft -= 1; }
    if (power[ps.y]?.[ps.x])   { power[ps.y][ps.x] = false;   this.pelletsLeft -= 1; }
    this.stepCount = 0;
    this.ghostsEatenCombo = 0;
    // Initialize scatter/chase phases: start with 7 second chase, alternate with 5 second scatter
    this.scatterChaseCycle = 0;
    this.phaseDuration = 420; // 7 seconds at ~60 steps/sec
    this.phaseTimer = 0;
    return this.observe();
  }


  getPacmen(): ReadonlyArray<{ id: number; pos: { x: number; y: number }; score: number; lifetimeScore: number }> {
    return this.pacmen;
  }

  isScatterPhase(): boolean {
    return this.scatterChaseCycle === 1;
  }

  getScatterTarget(ghostId: number, gridWidth: number, gridHeight: number): Vec2 {
    // Corner targets for scatter phase: NE, SE, SW, NW
    const corners = [
      { x: gridWidth - 2, y: 1 },
      { x: gridWidth - 2, y: gridHeight - 2 },
      { x: 1, y: gridHeight - 2 },
      { x: 1, y: 1 },
    ];
    return corners[ghostId % 4];
  }

  private nextPosition(pos: { x: number; y: number }, d: Direction): { x: number; y: number } {
    const next = { x: pos.x + DIR_VEC[d].x, y: pos.y + DIR_VEC[d].y };
    // Handle tunnel wraparound on left/right edges.
    if (next.x < 0) next.x = this.world.width - 1;
    if (next.x >= this.world.width) next.x = 0;
    return next;
  }

  private canMove(pos: { x: number; y: number }, d: Direction, avoidGhostHouse = false): boolean {
    const next = this.nextPosition(pos, d);
    return !this.world.isWall(next.x, next.y) && !(avoidGhostHouse && this.world.isGhostHouse(next.x, next.y));
  }

  getLegalActions(): Direction[] {
    const p = this.pacmen[0];
    return DIRECTIONS.filter((d) => this.canMove(p.pos, d, true));
  }

  private moveEntity(pos: { x: number; y: number }, d: Direction): void {
    const next = this.nextPosition(pos, d);
    if (!this.world.isWall(next.x, next.y)) {
      pos.x = next.x;
      pos.y = next.y;
    }
  }

  observe(): Observation {
    // Only ghosts that can actually catch Pac-Man enter the observation.
    // In-box ghosts are skipped in the collision loop, so exposing their positions
    // would only bloat the Q-table state space without affecting gameplay.
    const activeGhosts = this.ghosts.filter((g) => !g.inBox);
    const numEdible = activeGhosts.filter((g) => g.edibleTimer > 0).length;
    return encodeObservation(this.world, this.pacmen[0].pos, activeGhosts.map((g) => g.pos), numEdible > 0, numEdible);
  }


  private movementIterations(speed: number): number {
    const whole = Math.floor(speed);
    const frac = speed - whole;
    return whole + (this.rng.next() < frac ? 1 : 0);
  }

  step(action: number): StepResult {
    this.stepCount += 1;
    // Update scatter/chase phase timer
    this.phaseTimer += 1;
    if (this.phaseTimer >= this.phaseDuration) {
      this.phaseTimer = 0;
      this.scatterChaseCycle = 1 - this.scatterChaseCycle;
      // Scatter phases are shorter (5 sec) than chase (7 sec)
      this.phaseDuration = this.scatterChaseCycle === 0 ? 420 : 300;
    }

    let reward = this.params.reward.stepPenalty + this.params.reward.survivalReward;
    const pac = this.pacmen[0];
    const desired = actionToDirection(action);

    this.world.heatmap = this.world.heatmap.map((row) => row.map((v) => v * this.params.heatmapDecayRate));
    this.world.heatmap[pac.pos.y][pac.pos.x] += this.params.heatmapLearningRate;

    // Snapshot positions before any movement so cross-over collisions can be detected.
    const pacPrevPositions = new Map<number, { x: number; y: number }>(this.pacmen.map((p) => [p.id, { ...p.pos }]));

    // movementIterations handles fractional speed; don't clamp to 1 or slow speeds have no effect.
    for (let m = 0; m < this.movementIterations(this.params.pacmanSpeed); m += 1) {
      if (this.getLegalActions().includes(desired)) {
        this.moveEntity(pac.pos, desired);
      } else if (this.params.illegalMoveMode === 'noop') {
        const legal = this.getLegalActions();
        if (legal.length) this.moveEntity(pac.pos, legal[this.rng.int(legal.length)]);
      }
    }

    for (let i = 1; i < this.pacmen.length; i += 1) {
      const legal = DIRECTIONS.filter((d) => this.canMove(this.pacmen[i].pos, d, true));
      if (legal.length) this.moveEntity(this.pacmen[i].pos, legal[this.rng.int(legal.length)]);
    }

    // Pellet collection for all Pac-Men
    if (this.world.pellets[pac.pos.y][pac.pos.x]) {
      this.world.pellets[pac.pos.y][pac.pos.x] = false;
      this.pelletsLeft -= 1;
      reward += this.params.reward.pelletReward;
      pac.score += this.params.reward.pelletReward;
      pac.lifetimeScore += this.params.reward.pelletReward;
    }
    if (this.world.powerPellets[pac.pos.y][pac.pos.x]) {
      this.world.powerPellets[pac.pos.y][pac.pos.x] = false;
      this.pelletsLeft -= 1;
      reward += this.params.reward.powerPelletReward;
      pac.score += this.params.reward.powerPelletReward;
      pac.lifetimeScore += this.params.reward.powerPelletReward;
      this.ghosts.forEach((g) => { g.edibleTimer = this.params.powerPelletDuration; });
      this.ghostsEatenCombo = 0; // Reset combo for new power pellet
    }

    // Extra Pac-Men also collect pellets
    for (let i = 1; i < this.pacmen.length; i += 1) {
      const p = this.pacmen[i];
      if (this.world.pellets[p.pos.y]?.[p.pos.x]) {
        this.world.pellets[p.pos.y][p.pos.x] = false;
        this.pelletsLeft -= 1;
        p.score += this.params.reward.pelletReward;
        p.lifetimeScore += this.params.reward.pelletReward;
      }
      if (this.world.powerPellets[p.pos.y]?.[p.pos.x]) {
        this.world.powerPellets[p.pos.y][p.pos.x] = false;
        this.pelletsLeft -= 1;
        p.score += this.params.reward.powerPelletReward;
        p.lifetimeScore += this.params.reward.powerPelletReward;
        this.ghosts.forEach((g) => { g.edibleTimer = this.params.powerPelletDuration; });
        this.ghostsEatenCombo = 0;
      }
    }

    const ghostPrevPositions = new Map<number, { x: number; y: number }>();
    for (const ghost of this.ghosts) {
      if (ghost.releaseDelay > 0) {
        ghost.releaseDelay -= 1;
      } else {
        if (ghost.edibleTimer > 0) ghost.edibleTimer -= 1;
        ghostPrevPositions.set(ghost.id, { ...ghost.pos });
        const iters = this.movementIterations(this.params.ghostSpeed);
        for (let m = 0; m < iters; m += 1) {
          const move = chooseGhostMove(this.world, ghost, pac.pos, this);
          if (move !== null) this.moveEntity(ghost.pos, move);
        }
        // Transition out of box once ghost steps onto a non-ghost-house tile
        if (ghost.inBox && !this.world.isGhostHouse(ghost.pos.x, ghost.pos.y)) {
          ghost.inBox = false;
        }
      }
    }

    let done = false;
    // Check collisions for all Pac-Men
    for (const pacman of this.pacmen) {
      const pacPrev = pacPrevPositions.get(pacman.id) ?? pacman.pos;
      for (const ghost of this.ghosts) {
        if (ghost.inBox) continue; // ghosts in the pen cannot catch Pac-Man
        const dx = Math.abs(ghost.pos.x - pacman.pos.x);
        const dy = Math.abs(ghost.pos.y - pacman.pos.y);
        const sameTile = dx === 0 && dy === 0;
        const adjacentTile = (dx <= 1 && dy === 0) || (dx === 0 && dy <= 1);
        // Cross-over: pac and ghost swapped tiles in this step (they pass through each other).
        // 'tile' mode must detect this or captures are silently missed.
        const ghostPrev = ghostPrevPositions.get(ghost.id);
        const crossOver = ghostPrev !== undefined
          && ghost.pos.x === pacPrev.x && ghost.pos.y === pacPrev.y
          && pacman.pos.x === ghostPrev.x && pacman.pos.y === ghostPrev.y;
        const collided = this.params.captureRules === 'touch' ? adjacentTile : (sameTile || crossOver);
        if (!collided) continue;
        if (ghost.edibleTimer > 0) {
          this.ghostsEatenCombo += 1;
          const comboReward = this.params.reward.ghostEatReward * this.ghostsEatenCombo;
          reward += comboReward;
          pacman.score += comboReward;
          pacman.lifetimeScore += comboReward;
          ghost.pos = { ...this.maze.ghostStarts[ghost.id % this.maze.ghostStarts.length] };
          ghost.edibleTimer = 0;
          ghost.inBox = this.maze.ghostHouseExit !== undefined;
          ghost.releaseDelay = 0;
        } else {
          reward += this.params.reward.deathPenalty;
          done = true;
        }
      }
    }

    // Win: all pellets cleared
    if (this.pelletsLeft <= 0) {
      reward += this.params.reward.winBonus;
      pac.score += this.params.reward.winBonus;
      pac.lifetimeScore += this.params.reward.winBonus;
      done = true;
    }
    if (this.stepCount >= this.params.maxEpisodeSteps) done = true;
    return { obs: this.observe(), reward, done, info: { score: pac.score, lifetimeScore: pac.lifetimeScore, pelletsLeft: this.pelletsLeft, step: this.stepCount } };
  }
}

export const createDefaultEnv = (): PacmanEnvironment => {
  const env = new PacmanEnvironment();
  env.reset(42);
  return env;
};
