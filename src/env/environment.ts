import { chooseGhostMove, GhostAIType } from '../ghosts/ghostAi';
import { DIR_VEC, DIRECTIONS, Direction, actionToDirection } from '../engine/types';
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
}

export interface GhostState { id: number; pos: { x: number; y: number }; aiType: GhostAIType; edibleTimer: number; }
interface PacState { id: number; pos: { x: number; y: number }; score: number; }

export interface WorldState {
  width: number;
  height: number;
  pellets: boolean[][];
  powerPellets: boolean[][];
  heatmap: number[][];
  isWall(x: number, y: number): boolean;
}

export interface StepResult { obs: Observation; reward: number; done: boolean; info: { score: number; pelletsLeft: number; step: number }; }

const defaultParams: EnvParams = {
  mazeId: 'classic', pelletDensity: 1, numGhosts: 1, ghostSpeed: 0.95, pacmanSpeed: 1,
  enablePowerPellets: true, powerPelletDuration: 20, captureRules: 'tile', maxEpisodeSteps: 400,
  reward: { pelletReward: 5, powerPelletReward: 20, deathPenalty: -100, stepPenalty: -0.1, survivalReward: 0.02, ghostEatReward: 30, winBonus: 200 },
  heatmapDecayRate: 0.997, heatmapLearningRate: 0.03, illegalMoveMode: 'stay', cooperativePacmen: true, numPacmen: 1,
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
  world: WorldState = { width: 0, height: 0, pellets: [], powerPellets: [], heatmap: [], isWall: () => true };

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
    this.world = {
      width: w,
      height: h,
      pellets,
      powerPellets: power,
      heatmap: Array.from({ length: h }, () => Array.from({ length: w }, () => 0)),
      isWall: (x, y) => y < 0 || x < 0 || y >= h || x >= w || grid[y][x] === 1,
    };
    this.pacmen = Array.from({ length: this.params.numPacmen }, (_, i) => ({ id: i, pos: { ...this.maze.pacStart }, score: 0 }));
    this.ghosts = Array.from({ length: this.params.numGhosts }, (_, i) => ({ id: i, pos: { ...this.maze.ghostStarts[i % this.maze.ghostStarts.length] }, aiType: 'classic', edibleTimer: 0 }));
    this.pelletsLeft = pellets.flat().filter(Boolean).length + power.flat().filter(Boolean).length;
    this.stepCount = 0;
    this.ghostsEatenCombo = 0;
    return this.observe();
  }


  getPacmen(): ReadonlyArray<{ id: number; pos: { x: number; y: number }; score: number }> {
    return this.pacmen;
  }
  getLegalActions(): Direction[] {
    const p = this.pacmen[0];
    return DIRECTIONS.filter((d) => !this.world.isWall(p.pos.x + DIR_VEC[d].x, p.pos.y + DIR_VEC[d].y));
  }

  private moveEntity(pos: { x: number; y: number }, d: Direction): void {
    const next = { x: pos.x + DIR_VEC[d].x, y: pos.y + DIR_VEC[d].y };
    if (!this.world.isWall(next.x, next.y)) {
      pos.x = next.x;
      pos.y = next.y;
    }
  }

  observe(): Observation {
    return encodeObservation(this.world, this.pacmen[0].pos, this.ghosts.map((g) => g.pos));
  }


  private movementIterations(speed: number): number {
    const whole = Math.floor(speed);
    const frac = speed - whole;
    return whole + (this.rng.next() < frac ? 1 : 0);
  }

  step(action: number): StepResult {
    this.stepCount += 1;
    let reward = this.params.reward.stepPenalty + this.params.reward.survivalReward;
    const pac = this.pacmen[0];
    const desired = actionToDirection(action);

    this.world.heatmap = this.world.heatmap.map((row) => row.map((v) => v * this.params.heatmapDecayRate));
    this.world.heatmap[pac.pos.y][pac.pos.x] += this.params.heatmapLearningRate;

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
      const legal = DIRECTIONS.filter((d) => !this.world.isWall(this.pacmen[i].pos.x + DIR_VEC[d].x, this.pacmen[i].pos.y + DIR_VEC[d].y));
      if (legal.length) this.moveEntity(this.pacmen[i].pos, legal[this.rng.int(legal.length)]);
    }

    if (this.world.pellets[pac.pos.y][pac.pos.x]) {
      this.world.pellets[pac.pos.y][pac.pos.x] = false;
      this.pelletsLeft -= 1;
      reward += this.params.reward.pelletReward;
      pac.score += this.params.reward.pelletReward;
    }
    if (this.world.powerPellets[pac.pos.y][pac.pos.x]) {
      this.world.powerPellets[pac.pos.y][pac.pos.x] = false;
      this.pelletsLeft -= 1;
      reward += this.params.reward.powerPelletReward;
      pac.score += this.params.reward.powerPelletReward;
      this.ghosts.forEach((g) => { g.edibleTimer = this.params.powerPelletDuration; });
      this.ghostsEatenCombo = 0; // Reset combo for new power pellet
    }

    for (const ghost of this.ghosts) {
      if (ghost.edibleTimer > 0) ghost.edibleTimer -= 1;
      const iters = this.movementIterations(this.params.ghostSpeed);
      for (let m = 0; m < iters; m += 1) {
        const move = chooseGhostMove(this.world, ghost, pac.pos);
        this.moveEntity(ghost.pos, move);
      }
    }

    let done = false;
    for (const ghost of this.ghosts) {
      const touch = this.params.captureRules === 'touch'
        ? Math.abs(ghost.pos.x - pac.pos.x) + Math.abs(ghost.pos.y - pac.pos.y) <= 1
        : ghost.pos.x === pac.pos.x && ghost.pos.y === pac.pos.y;
      if (touch) {
        if (ghost.edibleTimer > 0) {
          // Combo multiplier: each successive ghost eaten doubles the points (like classic Pac-Man)
          this.ghostsEatenCombo += 1;
          const comboReward = this.params.reward.ghostEatReward * this.ghostsEatenCombo;
          reward += comboReward;
          pac.score += comboReward;
          ghost.pos = { ...this.maze.ghostStarts[ghost.id % this.maze.ghostStarts.length] };
          ghost.edibleTimer = 0;
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
      done = true;
    }
    if (this.stepCount >= this.params.maxEpisodeSteps) done = true;
    return { obs: this.observe(), reward, done, info: { score: pac.score, pelletsLeft: this.pelletsLeft, step: this.stepCount } };
  }
}

export const createDefaultEnv = (): PacmanEnvironment => {
  const env = new PacmanEnvironment();
  env.reset(42);
  return env;
};
