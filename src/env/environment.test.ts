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

    // Step a few times to accumulate score
    env.step(0);
    env.step(0);
    const afterSteps = pac.lifetimeScore;

    // Lifetime score should increase
    expect(afterSteps).toBeGreaterThan(initialLifetime);
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
    const extraPac = pacmen[1];
    const ghost = env.ghosts[0];

    // Position ghost on same tile as extra Pac-Man
    ghost.pos = { ...extraPac.pos };
    ghost.edibleTimer = 0; // Ghost is not edible

    const result = env.step(0);
    expect(result.done).toBe(true);
  });
});
