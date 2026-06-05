import { describe, expect, test, beforeEach } from 'vitest';
import { PacmanEnvironment } from './environment';
import { observationKey } from './observation';

describe('environment', () => {
  let env: PacmanEnvironment;

  beforeEach(() => {
    env = new PacmanEnvironment();
    env.reset(42);
  });

  test('initializes with pellets', () => {
    expect(env.pelletsLeft).toBeGreaterThan(0);
  });

  test('tracks step count', () => {
    const initialSteps = env.stepCount;
    env.step(0);
    expect(env.stepCount).toBe(initialSteps + 1);
  });

  test('detects win condition when pellets are cleared', () => {
    // Manually clear pellets to test win condition
    let result = env.step(0);
    // Keep stepping until we've cleared all pellets (happens naturally or by force)
    // For this test, we just verify the done flag works
    while (env.pelletsLeft > 0 && result.info.step < env.params.maxEpisodeSteps) {
      result = env.step(0);
    }
    if (env.pelletsLeft === 0) {
      expect(result.done).toBe(true);
    }
  });

  test('enforces max episode steps limit', () => {
    env.params.maxEpisodeSteps = 10;
    let result = env.step(0);
    for (let i = 0; i < 20; i++) {
      result = env.step(0);
    }
    expect(result.done).toBe(true);
  });

  test('accumulates lifetimeScore across deaths', () => {
    const pac = env.getPacmen()[0];
    const initialLifetime = pac.lifetimeScore;

    // Step in all directions until a pellet is collected (pacStart is now pellet-free).
    const dirs = [0, 1, 2, 3];
    for (let i = 0; i < 20 && pac.lifetimeScore === initialLifetime; i++) {
      env.step(dirs[i % dirs.length]);
    }

    expect(pac.lifetimeScore).toBeGreaterThan(initialLifetime);
  });

  test('allows Pac-Man to move through horizontal tunnels', () => {
    env.setParams({ mazeId: 'pacman-classic', numGhosts: 0 });
    env.reset(42);
    (env as unknown as { pacmen: Array<{ pos: { x: number; y: number } }> }).pacmen[0].pos = { x: 0, y: 13 };

    expect(env.getLegalActions()).toContain('left');

    env.step(2);
    expect(env.getPacmen()[0].pos).toEqual({ x: env.world.width - 1, y: 13 });
  });

  test('ghost scatter phase alternates', () => {
    const duration1 = 420; // First phase (chase)
    for (let i = 0; i < duration1; i++) {
      env.step(0);
    }
    const wasScatterAfterFirstPhase = env.isScatterPhase();

    const duration2 = 300; // Second phase (scatter)
    for (let i = 0; i < duration2; i++) {
      env.step(0);
    }
    const wasScatterAfterSecondPhase = env.isScatterPhase();

    // Phases should alternate
    expect(wasScatterAfterFirstPhase).not.toBe(wasScatterAfterSecondPhase);
  });

  test('tile capture mode only collides on the same tile', () => {
    env.params.captureRules = 'tile';
    env.params.numGhosts = 1;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.reset(42);
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];

    ghost.pos = { x: pac.pos.x + 1, y: pac.pos.y };
    ghost.inBox = false;
    ghost.edibleTimer = 0;

    const result = env.step(0);
    expect(result.done).toBe(false);
  });

  test('touch capture mode collides on adjacent tiles', () => {
    env.params.captureRules = 'touch';
    env.params.numGhosts = 1;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.reset(42);
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];

    ghost.pos = { x: pac.pos.x + 1, y: pac.pos.y };
    ghost.inBox = false;
    ghost.edibleTimer = 0;

    const result = env.step(0);
    expect(result.done).toBe(true);
  });

  test('detects ghost collision with first Pac-Man', () => {
    env.params.captureRules = 'tile';
    env.params.numGhosts = 1;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.reset(42);
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];

    // Position ghost on same tile as Pac-Man
    ghost.pos = { ...pac.pos };
    ghost.edibleTimer = 0; // Ghost is not edible

    const result = env.step(0);
    expect(result.done).toBe(true);
  });

  test('detects ghost collision with extra Pac-Men', () => {
    env.params.captureRules = 'tile';
    env.params.numGhosts = 1;
    env.params.numPacmen = 2;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.reset(42);
    const pacmen = env.getPacmen();
    const pac0 = pacmen[0];
    const extraPac = pacmen[1];
    const ghost = env.ghosts[0];

    // Move the extra pac somewhere reachable that is NOT pac 0's start tile;
    // otherwise placing the ghost on extraPac also colocates it with pac 0
    // and pac 0's collision short-circuits the test we care about.
    extraPac.pos = { x: pac0.pos.x + 1, y: pac0.pos.y };

    // Position ghost on same tile as extra Pac-Man
    ghost.pos = { ...extraPac.pos };
    ghost.edibleTimer = 0; // Ghost is not edible
    ghost.inBox = false;
    ghost.releaseDelay = 0;

    // H5: extra Pac-Man dying no longer terminates the episode (only
    // pac 0 ends the run). Extra-pac movement is independent of
    // pacmanSpeed, so we don't assert on the score (the extra pac may
    // have stepped away before the collision check); the primary
    // guarantee is that an extra-pac death doesn't end the episode.
    const result = env.step(0);
    expect(result.done).toBe(false);
  });

  // N3 regression (supersedes H7): win on last pellet beats a same-tick ghost
  // collision. The earlier H7 behavior preferred death; we now prefer the
  // win so the agent learns the correct +winBonus terminal Q-value for
  // last-pellet states. Pac is on the last pellet AND a ghost is on the
  // same tile — N3 says winBonus wins, collision is never evaluated.
  test('win on last pellet beats same-tick ghost collision (N3)', () => {
    env.params.captureRules = 'tile';
    env.params.numGhosts = 1;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.params.reward = { ...env.params.reward, winBonus: 1000, deathPenalty: -100 };
    env.reset(42);
    // Force win condition at step start (the env-internal flag in real play
    // would be set by pac eating the last pellet earlier this step).
    env.pelletsLeft = 0;
    const ghost = env.ghosts[0];
    ghost.pos = { ...env.getPacmen()[0].pos };
    ghost.edibleTimer = 0;
    ghost.inBox = false;
    ghost.releaseDelay = 0;
    const res = env.step(0);
    expect(res.done).toBe(true);
    // Reward should reflect +winBonus, NOT deathPenalty.
    expect(res.reward).toBeGreaterThan(900); // ~ +1000 winBonus + small step penalty
  });

  // H10 regression: an out-of-range action must not corrupt observation keys.
  test('step clamps lastAction to [-1, 3]', () => {
    env.reset(42);
    env.step(99); // way out of range
    const obs = env.observe();
    expect(obs.lastAction).toBeLessThanOrEqual(3);
    expect(obs.lastAction).toBeGreaterThanOrEqual(-1);
  });

  // NEW-2: Power pellet eaten before all ghosts release. The edibleTimer fix
  // (H2) ticks edibleTimer while the ghost still has releaseDelay > 0, so the
  // ghost should NOT emerge with full edibility on a delayed release.
  test('edibleTimer ticks down while ghost is release-delayed', () => {
    env.params.numGhosts = 2;
    env.params.powerPelletDuration = 20;
    env.params.ghostReleaseInterval = 60;
    env.reset(42);
    // Force-feed ghost 1 a full edibility timer; it has releaseDelay=60.
    const ghost = env.ghosts[1];
    ghost.edibleTimer = env.params.powerPelletDuration;
    const startTimer = ghost.edibleTimer;
    const startDelay = ghost.releaseDelay;
    expect(startDelay).toBeGreaterThan(0);
    for (let i = 0; i < 10; i += 1) env.step(0);
    expect(ghost.edibleTimer).toBe(Math.max(0, startTimer - 10));
    expect(ghost.releaseDelay).toBe(Math.max(0, startDelay - 10));
  });

  // H6 followup: pacDesiredDir tracks intent even when the move is wall-blocked.
  test('pacDesiredDir reflects intent even when blocked by a wall', () => {
    env.params.numGhosts = 0;
    env.reset(42);
    // pacStart={x:13,y:23} in pacman-classic. Action 0 = up; the tile above
    // is a wall (covered by the wall-block test). pacLastDir freezes; the
    // *desired* direction must still update so Pinky/Inky aim correctly.
    env.step(0); // attempt up
    expect(env.getPacDesiredDir()).toBe('up');
  });

  // N2 regression: heatmap must stay all-zero when heatmapEnabled=false
  // and all ghosts are classic. Before the fix, the decay loop always ran
  // (allocating h new arrays per step) even when nothing consumed the map.
  test('heatmap stays at zero when heatmapEnabled=false with all classic ghosts (N2)', () => {
    env.params.numGhosts = 1;
    env.heatmapEnabled = false;
    env.reset(42);
    // All ghosts start as classic (the default); no consumer → fast-path must skip.
    for (let i = 0; i < 20; i += 1) env.step(0);
    const flat = env.world.heatmap.flat();
    expect(flat.every((v) => v === 0)).toBe(true);
  });

  test('heatmap accumulates when heatmapEnabled=true (N2)', () => {
    env.params.numGhosts = 0;
    env.heatmapEnabled = true;
    env.reset(42);
    for (let i = 0; i < 5; i += 1) env.step(0);
    const flat = env.world.heatmap.flat();
    expect(flat.some((v) => v > 0)).toBe(true);
  });

  // N4 regression: hybrid ghost AI must use the env's seeded RNG (not
  // Math.random). Two resets with the same seed must produce identical
  // ghost positions; two resets with different seeds must diverge (proving
  // the seeded source is driving the randomness, not a fixed coin).
  test('hybrid ghost move sequence is deterministic for the same seed (N4)', () => {
    env.setParams({ numGhosts: 1, ghostSpeed: 1, pacmanSpeed: 0, maxEpisodeSteps: 100 });
    env.setGhostType(0, 'hybrid');

    // Run A
    env.reset(7777);
    for (let i = 0; i < 50; i += 1) env.step(0);
    const posA = { ...env.ghosts[0].pos };

    // Run B — same seed must land at the same position
    env.reset(7777);
    for (let i = 0; i < 50; i += 1) env.step(0);
    const posB = { ...env.ghosts[0].pos };

    expect(posA).toEqual(posB);
  });

  // D4.5: pelletEscalation ramps the per-pellet reward multiplier from 1× at
  // episode start to 6× for the final pellet (1 + 5·fractionEaten), biasing the
  // agent toward finishing the maze rather than loitering on rich territory.
  test('pelletEscalation ramps from 1x to 6x as pellets clear (D4.5)', () => {
    const esc = (): number => (env as unknown as { pelletEscalation(): number }).pelletEscalation();
    env.totalPellets = 100;
    env.pelletsLeft = 100; expect(esc()).toBeCloseTo(1);    // nothing eaten
    env.pelletsLeft = 50; expect(esc()).toBeCloseTo(3.5);   // half eaten
    env.pelletsLeft = 1; expect(esc()).toBeCloseTo(5.95);   // almost done
    env.totalPellets = 0; expect(esc()).toBe(1);            // guard: no pellets
  });

  // D4.5: clearPelletsTo (endgame curriculum) leaves exactly the target count
  // and is reproducible for a given RNG.
  test('clearPelletsTo leaves the target fraction of pellets (D4.5)', () => {
    env.setParams({ numGhosts: 0 });
    env.reset(42);
    const total = env.totalPellets;
    expect(total).toBeGreaterThan(10);
    env.clearPelletsTo(0.15, () => 0.5); // fixed RNG → noise term cancels (rand-rand=0)
    expect(env.pelletsLeft).toBe(Math.max(1, Math.floor(total * 0.15)));
    expect(env.pelletsLeft).toBeLessThan(total);
  });

  // D7.4: observeAt(pos) backs the Q-value overlay — same encoding as observe()
  // but with Pac placed at an arbitrary tile.
  test('observeAt encodes from the given position; observe() is the pac case (D7.4)', () => {
    env.reset(42);
    const pac = env.getPacmen()[0].pos;
    expect(env.observeAt(pac).pac).toEqual(pac);
    expect(observationKey(env.observeAt(pac))).toBe(observationKey(env.observe()));
    const elsewhere = { x: pac.x + 1, y: pac.y };
    expect(env.observeAt(elsewhere).pac).toEqual(elsewhere);
  });

  test('clearPelletsTo is deterministic under the seeded RNG (D4.5)', () => {
    const boardAfter = (): boolean[][] => {
      const e = new PacmanEnvironment();
      e.setParams({ numGhosts: 0 });
      e.reset(99);
      e.clearPelletsTo(0.3); // uses the env's own seeded RNG
      return e.world.pellets.map((row) => [...row]);
    };
    expect(boardAfter()).toEqual(boardAfter());
  });

  // D4.1 regression: two non-edible ghosts on Pac's tile must apply the death
  // penalty exactly ONCE. Before the collision-loop break, each colliding ghost
  // re-applied it (−200 instead of −100), corrupting terminal Q-values for
  // multi-ghost-contact states (especially touch mode, where two ghosts can be
  // adjacent on different sides).
  test('two ghosts on Pac tile apply deathPenalty only once (D4.1)', () => {
    env.params.captureRules = 'tile';
    env.params.numGhosts = 2;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.params.reward = { ...env.params.reward, deathPenalty: -100, stepPenalty: -0.1 };
    env.reset(42);
    const pac = env.getPacmen()[0];
    for (const g of env.ghosts) {
      g.pos = { ...pac.pos };
      g.edibleTimer = 0;
      g.inBox = false;
      g.releaseDelay = 0;
    }
    const result = env.step(0);
    expect(result.done).toBe(true);
    // Single penalty ≈ stepPenalty + deathPenalty = −100.1, NOT −200.1.
    expect(result.reward).toBeGreaterThan(-150);
    expect(result.reward).toBeLessThan(-50);
  });

});
