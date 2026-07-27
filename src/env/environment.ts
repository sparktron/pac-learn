import { chooseGhostMove, GhostAIType } from '../ghosts/ghostAi';
import { DIR_VEC, DIRECTIONS, Direction, Action, actionToDirection, directionToAction, reverseAction, Vec2, wrapPosition } from '../engine/types';
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
    /**
     * Small additive penalty applied whenever the action chosen reverses the
     * previous action (e.g. up after down). Combats two-step oscillation that
     * the `lastAction` observation feature alone doesn't punish — without a
     * negative reward gradient, the agent has no reason to prefer forward
     * progress over a directionless wobble.
     *
     * Defaults to -2: small enough that a genuine escape reversal (with a
     * looming -100 deathPenalty) still wins out, but large enough to break
     * ties between two roughly-equal Q-values for opposite actions.
     */
    reversePenalty: number;
  };
  heatmapDecayRate: number;
  heatmapLearningRate: number;
  illegalMoveMode: 'noop' | 'stay';
  numPacmen: number;
  ghostReleaseInterval: number;
  /** Steps the ghosts spend chasing before flipping to scatter (D4.8). */
  chaseDuration: number;
  /** Steps the ghosts spend scattering before flipping back to chase (D4.8). */
  scatterDuration: number;
  /**
   * Cruise Elroy (D3.11): when true, Blinky (role 0) accelerates in two stages
   * as the maze clears — the classic late-game menace. Default off so training
   * baselines are unaffected.
   */
  elroyEnabled: boolean;
  /**
   * Per-ghost targeting personality override (A2). `ghostPersonalities[i]` sets
   * ghost i's role (0=Blinky,1=Pinky,2=Inky,3=Clyde); an undefined/out-of-range
   * entry falls back to the default `id % 4`. Empty array → all default →
   * baseline-identical.
   */
  ghostPersonalities: (number | undefined)[];
}

export interface GhostState { id: number; pos: { x: number; y: number }; aiType: GhostAIType; edibleTimer: number; releaseDelay: number; inBox: boolean; lastDir: Direction | null; pendingReverse: boolean; /** Targeting role 0=Blinky,1=Pinky,2=Inky,3=Clyde (A2). Undefined → id%4. */ personality?: number; }
interface PacState { id: number; pos: { x: number; y: number }; score: number; lifetimeScore: number; ghostsEatenCombo: number; }

export interface WorldState {
  width: number;
  height: number;
  pellets: boolean[][];
  powerPellets: boolean[][];
  heatmap: number[][];
  isWall(x: number, y: number): boolean;
  isGhostHouse(x: number, y: number): boolean;
  ghostHouseExit?: { x: number; y: number };
  /** A3: when true, movement/observation/ghost-AI wrap top↔bottom too.
   *  Undefined is treated as false (passes through wrapPosition's wrapY default). */
  verticalTunnel?: boolean;
}

export interface StepResult { obs: Observation; reward: number; done: boolean; info: { score: number; lifetimeScore: number; pelletsLeft: number; step: number }; }

const defaultParams: EnvParams = {
  mazeId: 'pacman-classic', pelletDensity: 1, numGhosts: 2, ghostSpeed: 0.95, pacmanSpeed: 1,
  enablePowerPellets: true, powerPelletDuration: 20, captureRules: 'tile', maxEpisodeSteps: 1000,
  // Default reward shaping is win-seeking:
  //   • winBonus 1000 dominates everything else, so the agent has a clear "go for the win" signal
  //   • survivalReward 0 (was 0.02) — survival reward incentivized loitering, not winning
  //   • pelletReward grows as pellets are cleared (handled in step()): late pellets are worth 6×
  //     the base reward, motivating the agent to chase the last few pellets near ghost-clustered zones
  reward: { pelletReward: 5, powerPelletReward: 20, deathPenalty: -100, stepPenalty: -0.1, survivalReward: 0, ghostEatReward: 30, winBonus: 1000, reversePenalty: -2 },
  heatmapDecayRate: 0.997, heatmapLearningRate: 0.03, illegalMoveMode: 'stay', numPacmen: 1,
  ghostReleaseInterval: 60,
  // Classic Pac-Man alternates 7s chase / 5s scatter at ~60 steps/sec.
  chaseDuration: 420, scatterDuration: 300,
  elroyEnabled: false,
  ghostPersonalities: [],
};

/**
 * Cruise Elroy speed for a ghost (D3.11). Returns `baseSpeed` unchanged unless
 * Elroy is enabled AND this is Blinky (role 0). Blinky then accelerates in two
 * stages as pellets are cleared: +0.10 once half the maze is gone, +0.25 once
 * 80% is gone. Pure + exported so the staging is unit-tested.
 */
export const cruiseElroySpeed = (
  baseSpeed: number,
  pelletsLeft: number,
  totalPellets: number,
  enabled: boolean,
  isBlinky: boolean,
): number => {
  if (!enabled || !isBlinky || totalPellets <= 0) return baseSpeed;
  const fractionEaten = 1 - pelletsLeft / totalPellets;
  const boost = fractionEaten >= 0.8 ? 0.25 : fractionEaten >= 0.5 ? 0.10 : 0;
  return baseSpeed + boost;
};

export class PacmanEnvironment {
  params: EnvParams = structuredClone(defaultParams);
  private rng = new SeededRng(42);
  private maze = MAZES[0];
  private pacmen: PacState[] = [];
  ghosts: GhostState[] = [];
  pelletsLeft = 0;
  /** Snapshot of pelletsLeft at episode start, used for pellet-escalation reward shaping. */
  totalPellets = 0;
  /** Active power pellets remaining (separate counter avoids re-scanning the grid). */
  powerPelletsLeft = 0;
  stepCount = 0;
  /** When true, step() decays and updates the heatmap even if no ghost uses
   *  it. Set by the UI when the heatmap overlay is being shown. Defaults to
   *  false so headless bench training doesn't pay the heatmap cost. */
  heatmapEnabled = false;
  private scatterChaseCycle = 0; // 0 = chase, 1 = scatter
  private phaseDuration = 0;
  private phaseTimer = 0;
  private pacLastDir: Direction = 'left';
  /** Last *requested* direction, set every step regardless of whether the
   *  move was legal. Pinky/Inky target tiles ahead of Pac-Man's intent —
   *  using pacLastDir (only updated on successful moves) made them aim
   *  4 tiles in a stale direction whenever Pac-Man was wall-bumped. */
  private pacDesiredDir: Direction = 'left';
  private lastAction: number = -1;
  // Anti-oscillation hard filter was removed: lastAction (in the observation
  // key) plus reversePenalty (a soft reward shaping) plus the new ghost-
  // heading observation give the agent enough signal to avoid the X→~X→X→~X
  // loop without us hard-removing a legal direction. The hard filter was also
  // implemented backwards (removed lastAction's direction instead of its
  // reverse) and was actively hurting policies in narrow corridors.
  // private secondLastAction: number = -1;
  // private thirdLastAction: number = -1;
  world: WorldState = { width: 0, height: 0, pellets: [], powerPellets: [], heatmap: [], isWall: () => true, isGhostHouse: () => false, verticalTunnel: false };

  setParams(params: Partial<EnvParams>): void {
    this.params = { ...this.params, ...params, reward: { ...this.params.reward, ...(params.reward ?? {}) } };
  }

  setGhostType(index: number, type: GhostAIType): void {
    if (this.ghosts[index]) this.ghosts[index].aiType = type;
  }

  /**
   * Curriculum helper: clear a fraction of remaining pellets, leaving the env
   * in a "late-game" state from a fresh episode. Used by the bench's
   * endgame-curriculum mode (Priority 3a) so the agent can learn how to
   * survive the final cluster of pellets without first having to navigate the
   * full maze.
   *
   * Strategy: clear pellets at random, but preferentially keep pellets that
   * are far from pacStart (those tend to be in the harder-to-reach corners).
   * This roughly mimics how a real endgame looks: pellets near the spawn are
   * already eaten, those in ghost-adjacent zones remain.
   *
   * @param targetRemainingFraction  desired pelletsLeft / totalPellets (e.g. 0.15
   *                                  for "leave 15% of pellets — bucket=0/1 endgame")
   * @param rngFn  Optional RNG for reproducible curriculum sampling. Falls back
   *               to the env's seeded RNG.
   */
  clearPelletsTo(targetRemainingFraction: number, rngFn?: () => number): void {
    if (this.totalPellets <= 0) return;
    const target = Math.max(1, Math.floor(this.totalPellets * targetRemainingFraction));
    if (this.pelletsLeft <= target) return;

    const rand = rngFn ?? ((): number => this.rng.next());
    const pacStart = this.maze.pacStart;

    // Collect all (kind, x, y) for active pellets along with distance from pacStart.
    type P = { kind: 'pellet' | 'power'; x: number; y: number; dist: number };
    const all: P[] = [];
    for (let y = 0; y < this.world.height; y += 1) {
      for (let x = 0; x < this.world.width; x += 1) {
        const dist = Math.abs(x - pacStart.x) + Math.abs(y - pacStart.y);
        if (this.world.pellets[y]?.[x]) all.push({ kind: 'pellet', x, y, dist });
        if (this.world.powerPellets[y]?.[x]) all.push({ kind: 'power', x, y, dist });
      }
    }
    // Sort by distance ASC + tiny noise, so close-to-pac pellets clear first
    // but ties are broken randomly (avoids systematic asymmetry). Use a
    // decorate-sort-undecorate pattern: rolling the noise inside the
    // comparator made `sort` see different orderings for the same (a,b)
    // pair on different comparisons, which violates the comparator contract
    // and is non-deterministic.
    const sortKeys = new Map<P, number>(
      all.map((p) => [p, p.dist + (rand() - rand()) * 0.5]),
    );
    // Sort key is built from `all`, so get() is guaranteed defined — no
    // `?? 0` fallback, which would hide a future contract violation as
    // a silent sort-on-zero.
    all.sort((a, b) => sortKeys.get(a)! - sortKeys.get(b)!);

    const toClear = this.pelletsLeft - target;
    for (let i = 0; i < toClear && i < all.length; i += 1) {
      const p = all[i];
      if (p.kind === 'pellet') {
        this.world.pellets[p.y][p.x] = false;
      } else {
        this.world.powerPellets[p.y][p.x] = false;
        this.powerPelletsLeft -= 1;
      }
    }
    this.pelletsLeft = target;
  }

  reset(seed = 42): Observation {
    this.rng = new SeededRng(seed);
    this.maze = MAZES.find((m) => m.id === this.params.mazeId) ?? MAZES[0];
    const { grid } = this.maze;
    const h = grid.length;
    const w = grid[0].length;
    const pacStart = this.maze.pacStart;
    // Use maze-defined power pellet positions (avoids placing them on walls).
    const powerPositions = this.params.enablePowerPellets
      ? (this.maze.powerPelletPositions ?? []).filter(
          (p) => grid[p.y]?.[p.x] === 0 && !(p.x === pacStart.x && p.y === pacStart.y),
        )
      : [];
    const power = Array.from({ length: h }, () => Array.from({ length: w }, () => false));
    powerPositions.forEach((p) => { power[p.y][p.x] = true; });
    const pellets = Array.from({ length: h }, (_, y) =>
      Array.from({ length: w }, (_, x) => {
        if (power[y][x]) return false;
        // Consume the RNG roll even for pacStart so subsequent pellet positions
        // stay identical to runs before this tile was excluded.
        const roll = grid[y][x] === 0 ? this.rng.next() : 1;
        if (x === pacStart.x && y === pacStart.y) return false;
        return roll < this.params.pelletDensity;
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
      verticalTunnel: this.maze.verticalTunnel ?? false,
    };
    this.pacmen = Array.from({ length: this.params.numPacmen }, (_, i) => ({ id: i, pos: { ...this.maze.pacStart }, score: 0, lifetimeScore: 0, ghostsEatenCombo: 0 }));
    this.ghosts = Array.from({ length: this.params.numGhosts }, (_, i) => ({
      id: i,
      pos: { ...this.maze.ghostStarts[i % this.maze.ghostStarts.length] },
      aiType: 'classic',
      edibleTimer: 0,
      inBox: hasGhostHouse,
      releaseDelay: i * this.params.ghostReleaseInterval,
      lastDir: null,
      pendingReverse: false,
      // A2: undefined when no override → ghostAi falls back to id % 4.
      personality: this.params.ghostPersonalities[i],
    }));
    this.pelletsLeft = pellets.flat().filter(Boolean).length + power.flat().filter(Boolean).length;
    this.totalPellets = this.pelletsLeft;
    this.powerPelletsLeft = power.flat().filter(Boolean).length;
    this.stepCount = 0;
    // Initialize scatter/chase phases: start with 7 second chase, alternate with 5 second scatter
    this.scatterChaseCycle = 0;
    this.phaseDuration = this.params.chaseDuration; // start in chase (D4.8)
    this.phaseTimer = 0;
    this.pacLastDir = 'left';
    this.pacDesiredDir = 'left';
    this.lastAction = -1;
    // this.secondLastAction = -1;
    // this.thirdLastAction = -1;
    return this.observe();
  }

  /** Deterministic random in [0,1) drawn from the env's seeded RNG. Use this
   *  in ghost AI / any cross-module code that wants reproducibility instead
   *  of Math.random(). */
  nextRand(): number { return this.rng.next(); }


  getPacmen(): ReadonlyArray<PacState> {
    return this.pacmen;
  }

  isScatterPhase(): boolean {
    return this.scatterChaseCycle === 1;
  }

  getPacLastDir(): Direction { return this.pacLastDir; }
  /** Pac-Man's intended direction even if the move was blocked by a wall.
   *  Use this for ghost targeting (Pinky/Inky) so they don't lock onto a
   *  stale heading while Pac-Man is held against a wall. */
  getPacDesiredDir(): Direction { return this.pacDesiredDir; }
  getBlinkyPos(): Vec2 { return this.ghosts[0]?.pos ?? { x: 0, y: 0 }; }

  // Mode change forces every ghost to reverse on its next move (classic Pac-Man
  // behavior). Stored per-ghost on GhostState.pendingReverse so each ghost
  // consumes its own flag — a single env-wide flag was eaten by ghost 0 only,
  // and ghosts 1..N never reversed on chase↔scatter transitions.

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
    // Tunnel wraparound shared with the ghost AI via wrapPosition (D3.1). y wraps
    // too when the maze opts into a vertical tunnel (A3); off otherwise.
    return wrapPosition(this.world.width, this.world.height, pos.x + DIR_VEC[d].x, pos.y + DIR_VEC[d].y, this.world.verticalTunnel);
  }

  private canMove(pos: { x: number; y: number }, d: Direction, avoidGhostHouse = false): boolean {
    const next = this.nextPosition(pos, d);
    return !this.world.isWall(next.x, next.y) && !(avoidGhostHouse && this.world.isGhostHouse(next.x, next.y));
  }

  /** Legal moves as branded action indices — the form the RL agents consume.
   *  Centralizes the `getLegalActions().map(directionToAction)` that every
   *  caller (trainer, eval, bench, AI-watch) used to inline. */
  getLegalActionIndices(): Action[] {
    return this.getLegalActions().map(directionToAction);
  }

  getLegalActions(): Direction[] {
    const p = this.pacmen[0];
    return DIRECTIONS.filter((d) => this.canMove(p.pos, d, true));
    // Anti-oscillation hard filter removed: between lastAction in the
    // observation key, the reversePenalty reward, and the ghost-heading
    // observation feature, the agent now has enough signal to learn to
    // avoid two-step loops without a hard legal-action mask. The hard
    // filter was also implemented backwards (removed lastAction's
    // direction instead of its reverse), so leaving it in actively hurt
    // policies in narrow corridors. Kept in git history for reference.
    // if (this.lastAction >= 0 && this.secondLastAction >= 0 && this.thirdLastAction >= 0) {
    //   const lastReversed   = reverseAction(this.lastAction)   === this.secondLastAction;
    //   const secondReversed = reverseAction(this.secondLastAction) === this.thirdLastAction;
    //   if (lastReversed && secondReversed) {
    //     const forbidden = DIRECTIONS[reverseAction(this.lastAction)];
    //     const noTripleReversal = all.filter((d) => d !== forbidden);
    //     if (noTripleReversal.length > 0) return noTripleReversal;
    //   }
    // }
  }

  private moveEntity(pos: { x: number; y: number }, d: Direction): void {
    const next = this.nextPosition(pos, d);
    if (!this.world.isWall(next.x, next.y)) {
      pos.x = next.x;
      pos.y = next.y;
    }
  }

  observe(): Observation {
    return this.observeAt(this.pacmen[0].pos);
  }

  /**
   * Encode the observation as if Pac-Man were at `pacPos`, with all other state
   * (ghosts, pellets, lastAction) unchanged. `observe()` is the special case
   * `pacPos = pacmen[0].pos`. Used by the Q-value overlay to show what the agent
   * would value being at each tile in the current game state.
   */
  observeAt(pacPos: Vec2): Observation {
    // Only ghosts that can actually catch Pac-Man enter the observation.
    // In-box ghosts are skipped in the collision loop, so exposing their positions
    // would only bloat the Q-table state space without affecting gameplay.
    // D4.6: a ghost still serving releaseDelay is also skipped in the collision
    // loop (see :541), so it can't catch Pac-Man either. Exclude it here for
    // parity — otherwise on houseless mazes (inBox=false but releaseDelay>0) the
    // agent would see a phantom threat it can walk straight through. No effect on
    // classic-maze training: those ghosts start inBox and aren't yet released.
    const activeGhosts = this.ghosts.filter((g) => !g.inBox && g.releaseDelay <= 0);
    return encodeObservation(
      this.world,
      pacPos,
      activeGhosts.map((g) => g.pos),
      activeGhosts.map((g) => g.edibleTimer > 0),
      this.lastAction,
      this.pelletsLeft,
      this.totalPellets,
      this.powerPelletsLeft,
      activeGhosts.map((g) => g.lastDir),
    );
  }


  private movementIterations(speed: number): number {
    const whole = Math.floor(speed);
    const frac = speed - whole;
    return whole + (this.rng.next() < frac ? 1 : 0);
  }

  /**
   * Per-pellet reward multiplier that grows as pellets are cleared. Late
   * pellets are worth more so the agent is motivated to chase the last few
   * (which usually cluster near ghosts) rather than die mid-maze with high
   * average reward-per-step. Multiplier: 1× at start, ramps to 6× for last pellet.
   *
   * Call BEFORE decrementing pelletsLeft for the current pellet.
   */
  private pelletEscalation(): number {
    if (this.totalPellets <= 0) return 1;
    const fractionEaten = 1 - this.pelletsLeft / this.totalPellets;
    return 1 + 5 * fractionEaten;
  }

  /** Collect the pellet under one Pac-Man after an atomic tile movement. */
  private collectPelletAt(pacman: PacState, contributesToTrainingReward: boolean): number {
    let reward = 0;
    const { x, y } = pacman.pos;
    if (this.world.pellets[y]?.[x]) {
      const pelletReward = this.params.reward.pelletReward * this.pelletEscalation();
      this.world.pellets[y][x] = false;
      this.pelletsLeft -= 1;
      if (contributesToTrainingReward) reward += pelletReward;
      pacman.score += pelletReward;
      pacman.lifetimeScore += pelletReward;
    }
    if (this.world.powerPellets[y]?.[x]) {
      const pelletReward = this.params.reward.powerPelletReward * this.pelletEscalation();
      this.world.powerPellets[y][x] = false;
      this.pelletsLeft -= 1;
      this.powerPelletsLeft -= 1;
      if (contributesToTrainingReward) reward += pelletReward;
      pacman.score += pelletReward;
      pacman.lifetimeScore += pelletReward;
      this.ghosts.forEach((ghost) => { ghost.edibleTimer = this.params.powerPelletDuration; });
      // The frightened phase is global, but each Pac-Man accumulates its own combo.
      this.pacmen.forEach((p) => { p.ghostsEatenCombo = 0; });
    }
    return reward;
  }

  private positionsCollide(
    pacman: PacState,
    ghost: GhostState,
    pacPrev?: Vec2,
    ghostPrev?: Vec2,
  ): boolean {
    const dx = Math.abs(ghost.pos.x - pacman.pos.x);
    const dy = Math.abs(ghost.pos.y - pacman.pos.y);
    const sameTile = dx === 0 && dy === 0;
    const adjacentTile = (dx <= 1 && dy === 0) || (dx === 0 && dy <= 1);
    const crossOver = pacPrev !== undefined && ghostPrev !== undefined
      && ghost.pos.x === pacPrev.x && ghost.pos.y === pacPrev.y
      && pacman.pos.x === ghostPrev.x && pacman.pos.y === ghostPrev.y;
    return this.params.captureRules === 'touch' ? adjacentTile : (sameTile || crossOver);
  }

  step(action: Action): StepResult {
    // Clamp to the [-1, 3] range that observationKey reserves for lastAction
    // (LAST_ACTION_BASE=5 after the +1 shift). An out-of-range action would
    // silently overflow its slot and collide with the next field
    // (pelletsRemainingBucket), corrupting Q-table keys for unrelated states.
    const clampedAction = Math.max(-1, Math.min(3, action));
    const prevAction = this.lastAction;
    // Anti-oscillation history shifts removed alongside the filter; only
    // lastAction is still needed (observation key + reversePenalty check).
    // this.thirdLastAction = this.secondLastAction;
    // this.secondLastAction = this.lastAction;
    this.lastAction = clampedAction;
    this.stepCount += 1;
    // Update scatter/chase phase timer
    this.phaseTimer += 1;
    if (this.phaseTimer >= this.phaseDuration) {
      this.phaseTimer = 0;
      this.scatterChaseCycle = 1 - this.scatterChaseCycle;
      // Phase durations are configurable (D4.8); defaults keep the classic
      // 7s chase / 5s scatter cadence.
      this.phaseDuration = this.scatterChaseCycle === 0 ? this.params.chaseDuration : this.params.scatterDuration;
      // Classic Pac-Man: ghosts reverse direction on mode change.
      for (const g of this.ghosts) g.pendingReverse = true;
    }

    let reward = this.params.reward.stepPenalty + this.params.reward.survivalReward;
    // prevAction is lastAction (number, -1 sentinel); inside the >=0 guard it's
    // a valid Action, so the cast is sound. clampedAction is likewise 0..3 here.
    if (prevAction >= 0 && clampedAction === reverseAction(prevAction as Action)) {
      reward += this.params.reward.reversePenalty;
    }
    const pac = this.pacmen[0];
    // Use clampedAction (not raw action) so a caller passing out-of-range
    // values can't make pacDesiredDir disagree with what observation/lastAction
    // see — they all derive from the same [-1, 3] view of the input.
    const desired = actionToDirection(clampedAction as Action);
    this.pacDesiredDir = desired;

    // N2: in-place decay (the prior .map().map() pattern allocated h+1 arrays
    // per step — measurable GC pressure in max-speed training). Also skip
    // entirely when no consumer (all ghosts are 'classic' AND the UI isn't
    // showing the heatmap overlay). Headless bench runs leave heatmapEnabled
    // false and stay on the fast path.
    const heatmapNeeded = this.heatmapEnabled || this.ghosts.some((g) => g.aiType !== 'classic');
    if (heatmapNeeded) {
      const decay = this.params.heatmapDecayRate;
      const heatmap = this.world.heatmap;
      for (let y = 0; y < heatmap.length; y += 1) {
        const row = heatmap[y];
        for (let x = 0; x < row.length; x += 1) row[x] *= decay;
      }
      heatmap[pac.pos.y][pac.pos.x] += this.params.heatmapLearningRate;
    }

    const finish = (done: boolean): StepResult => ({
      obs: this.observe(),
      reward,
      done,
      info: { score: pac.score, lifetimeScore: pac.lifetimeScore, pelletsLeft: this.pelletsLeft, step: this.stepCount },
    });
    const finishWin = (): StepResult => {
      reward += this.params.reward.winBonus;
      pac.score += this.params.reward.winBonus;
      pac.lifetimeScore += this.params.reward.winBonus;
      return finish(true);
    };

    // A non-edible secondary Pac-Man remains on the board after being caught.
    // Remember it for this tick so later microsteps cannot apply the death
    // penalty repeatedly.
    const defeatedPacmen = new Set<number>();
    const resolveCollision = (
      pacman: PacState,
      ghost: GhostState,
      pacPrev?: Vec2,
      ghostPrev?: Vec2,
    ): { primaryDied: boolean; ghostEaten: boolean } => {
      if (defeatedPacmen.has(pacman.id) || ghost.inBox || ghost.releaseDelay > 0
        || !this.positionsCollide(pacman, ghost, pacPrev, ghostPrev)) {
        return { primaryDied: false, ghostEaten: false };
      }
      if (ghost.edibleTimer > 0) {
        pacman.ghostsEatenCombo += 1;
        const comboReward = this.params.reward.ghostEatReward * pacman.ghostsEatenCombo;
        if (pacman.id === 0) reward += comboReward;
        pacman.score += comboReward;
        pacman.lifetimeScore += comboReward;
        ghost.pos = { ...this.maze.ghostStarts[ghost.id % this.maze.ghostStarts.length] };
        ghost.edibleTimer = 0;
        ghost.inBox = this.maze.ghostHouseExit !== undefined;
        ghost.releaseDelay = 0;
        ghost.lastDir = null;
        return { primaryDied: false, ghostEaten: true };
      }

      defeatedPacmen.add(pacman.id);
      if (pacman.id === 0) {
        reward += this.params.reward.deathPenalty;
        return { primaryDied: true, ghostEaten: false };
      }
      pacman.score += this.params.reward.deathPenalty;
      pacman.lifetimeScore += this.params.reward.deathPenalty;
      return { primaryDied: false, ghostEaten: false };
    };
    const resolvePacmanCollisions = (pacman: PacState): boolean => {
      for (const ghost of this.ghosts) {
        const collision = resolveCollision(pacman, ghost);
        if (collision.primaryDied) return true;
        // Once eaten, the ghost has reset; do not compare its new spawn tile
        // against other Pac-Men during the same atomic movement.
        if (collision.ghostEaten) break;
      }
      return false;
    };

    // Resolve gameplay after every atomic tile movement. Checking only the
    // final endpoints allowed speeds above 1 to skip pellets and pass through
    // ghosts between those endpoints.
    reward += this.collectPelletAt(pac, true);
    if (this.pelletsLeft <= 0) return finishWin();

    // movementIterations handles fractional speed; don't clamp to 1 or slow speeds have no effect.
    const pacPrevPositions = new Map<number, Vec2>();
    const pacIterations = this.movementIterations(this.params.pacmanSpeed);
    if (pacIterations === 0 && resolvePacmanCollisions(pac)) return finish(true);
    for (let m = 0; m < pacIterations; m += 1) {
      pacPrevPositions.set(pac.id, { ...pac.pos });
      if (this.getLegalActions().includes(desired)) {
        this.moveEntity(pac.pos, desired);
        this.pacLastDir = desired;
      } else if (this.params.illegalMoveMode === 'noop') {
        const legal = this.getLegalActions();
        if (legal.length) {
          const d = legal[this.rng.int(legal.length)];
          this.moveEntity(pac.pos, d);
          this.pacLastDir = d;
        }
      }
      reward += this.collectPelletAt(pac, true);
      // Preserve N3: clearing the final pellet wins before a same-microstep
      // collision can replace the terminal reward with a death penalty.
      if (this.pelletsLeft <= 0) return finishWin();
      // On intermediate high-speed tiles, resolve immediately before Pac-Man
      // can skip through an occupied tile. Defer the final tile until ghost
      // movement so the established one-tile swap/cross-over semantics remain.
      if (m < pacIterations - 1 && resolvePacmanCollisions(pac)) return finish(true);
    }

    // Extra Pac-Men move one tile per step and collect independently. Their
    // score changes do not contribute to the primary training reward.
    for (let i = 1; i < this.pacmen.length; i += 1) {
      const extraPac = this.pacmen[i];
      pacPrevPositions.set(extraPac.id, { ...extraPac.pos });
      const legal = DIRECTIONS.filter((d) => this.canMove(extraPac.pos, d, true));
      if (legal.length) this.moveEntity(extraPac.pos, legal[this.rng.int(legal.length)]);
      this.collectPelletAt(extraPac, false);
      if (this.pelletsLeft <= 0) return finishWin();
    }

    for (const ghost of this.ghosts) {
      // Tick edibleTimer and releaseDelay UNCONDITIONALLY each step. Before,
      // a ghost with releaseDelay > 0 also froze its edibleTimer — so a
      // power pellet eaten before all ghosts had released bestowed the full
      // edibility duration that began ticking only AFTER release. The ghost
      // then emerged already pre-loaded with maximum frightened time.
      if (ghost.edibleTimer > 0) ghost.edibleTimer -= 1;
      if (ghost.releaseDelay > 0) {
        ghost.releaseDelay -= 1;
        continue; // skip movement, but the timers above still tick
      }
      // Cruise Elroy (D3.11): Blinky (role 0) speeds up late-game when enabled.
      const ghostSpeed = cruiseElroySpeed(
        this.params.ghostSpeed, this.pelletsLeft, this.totalPellets,
        this.params.elroyEnabled, (ghost.personality ?? ghost.id % 4) === 0,
      );
      const iters = this.movementIterations(ghostSpeed);
      for (let m = 0; m < iters; m += 1) {
        const ghostPrev = { ...ghost.pos };
        const move = chooseGhostMove(this.world, ghost, pac.pos, this);
        if (move !== null) {
          this.moveEntity(ghost.pos, move);
          ghost.lastDir = move;
        }
        // Transition out of the box immediately after each tile, then check
        // collisions before another high-speed movement can skip over Pac-Man.
        if (ghost.inBox && !this.world.isGhostHouse(ghost.pos.x, ghost.pos.y)) {
          ghost.inBox = false;
        }
        let ghostEaten = false;
        for (const pacman of this.pacmen) {
          const collision = resolveCollision(pacman, ghost, pacPrevPositions.get(pacman.id), ghostPrev);
          if (collision.primaryDied) return finish(true);
          if (collision.ghostEaten) {
            ghostEaten = true;
            break;
          }
        }
        if (ghostEaten) break;
      }
      // Also handle zero-speed ghosts (and tests/debug tools that reposition a
      // ghost directly outside the pen without an intervening movement).
      if (ghost.inBox && !this.world.isGhostHouse(ghost.pos.x, ghost.pos.y)) {
        ghost.inBox = false;
      }
    }

    // Covers stationary entities and ghosts whose release delay reached zero
    // this tick without movement. Already-defeated secondary Pac-Men are
    // ignored, so this cannot double-apply a penalty.
    for (const pacman of this.pacmen) {
      if (resolvePacmanCollisions(pacman)) return finish(true);
    }

    // N3: the win path is now handled earlier (before ghost movement) so a
    // same-tick collision can't steal the win. We never reach here with
    // pelletsLeft <= 0 from a normal flow — the old gated win-bonus block
    // was removed as dead code.
    return finish(this.stepCount >= this.params.maxEpisodeSteps);
  }
}

export const createDefaultEnv = (): PacmanEnvironment => {
  const env = new PacmanEnvironment();
  env.reset(42);
  return env;
};
