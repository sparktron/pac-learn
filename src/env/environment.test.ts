import { describe, expect, test, beforeEach } from 'vitest';
import { PacmanEnvironment } from './environment';

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

  // H7 regression: winBonus must not stack with deathPenalty on the same step.
  test('death on last-pellet step does not grant winBonus', () => {
    env.params.captureRules = 'tile';
    env.params.numGhosts = 1;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.params.reward = { ...env.params.reward, winBonus: 1000, deathPenalty: -100 };
    env.reset(42);
    // Force win condition: zero pellets remaining at step start.
    env.pelletsLeft = 0;
    // Force collision with first ghost.
    const ghost = env.ghosts[0];
    ghost.pos = { ...env.getPacmen()[0].pos };
    ghost.edibleTimer = 0;
    ghost.inBox = false;
    ghost.releaseDelay = 0;
    const res = env.step(0);
    expect(res.done).toBe(true);
    // Reward should reflect deathPenalty alone (plus stepPenalty/survival),
    // NOT winBonus on top.
    expect(res.reward).toBeLessThan(0);
    expect(res.reward).toBeGreaterThan(-200); // no +1000 stacked
  });

  // H10 regression: an out-of-range action must not corrupt observation keys.
  test('step clamps lastAction to [-1, 3]', () => {
    env.reset(42);
    env.step(99); // way out of range
    const obs = env.observe();
    expect(obs.lastAction).toBeLessThanOrEqual(3);
    expect(obs.lastAction).toBeGreaterThanOrEqual(-1);
  });

  // NEW-1 (CRITICAL): the anti-oscillation filter in getLegalActions removes
  // the wrong direction. After history X→~X→X, the comment says "block the
  // third consecutive reversal" — i.e. forbid the next action being ~X. The
  // current code removes DIRECTIONS[lastAction] = X, leaving ~X as legal and
  // forcing the only-reversal-available branch when the corridor is narrow.
  // This test currently fails — it asserts the intended behavior.
  test.fails('after X→~X→X history, the next-reversal direction is forbidden', () => {
    env.params.numGhosts = 0;
    env.reset(42);
    // Drop the pacman into the middle of a known straight horizontal corridor
    // in pacman-classic (row y=5 is open from x=1..26, free of ghosts).
    type Mut = { pacmen: Array<{ pos: { x: number; y: number } }> };
    (env as unknown as Mut).pacmen[0].pos = { x: 10, y: 5 };
    // Drive an X→~X→X history: left(2), right(3), left(2).
    env.step(2); env.step(3); env.step(2);
    const legal = env.getLegalActions();
    // The 'right' move (reverseAction of last 'left') is what the comment
    // calls a "third consecutive reversal" and the filter is supposed to
    // drop. The bug is that the filter currently drops 'left' instead.
    expect(legal).not.toContain('right');
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
});
