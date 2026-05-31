import { describe, expect, test } from 'vitest';
import { PacmanEnvironment, createDefaultEnv } from '../env/environment';

describe('maze collisions', () => {
  test('wall tiles block movement', () => {
    const env = createDefaultEnv();
    env.reset(1);
    const before = { ...env.getPacmen()[0].pos };
    env.step(0); // up into wall — row above pacStart {x:13,y:23} is a wall tile
    expect(env.getPacmen()[0].pos).toEqual(before);
  });

  // Helper: a frozen env (speeds 0) with one ghost we can place by hand, so the
  // collision branch under test is the only moving part.
  const frozenEnv = (): PacmanEnvironment => {
    const env = new PacmanEnvironment();
    env.params.captureRules = 'tile';
    env.params.numGhosts = 1;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.reset(42);
    return env;
  };

  // D4.2: eating an edible ghost is a combo-rewarded, non-terminal event that
  // resets the ghost to its start tile. The old review noted zero coverage here.
  test('eating an edible ghost rewards the combo and resets the ghost (D4.2)', () => {
    const env = frozenEnv();
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];
    const startTile = { ...ghost.pos }; // ghost's spawn, captured before we move it
    ghost.pos = { ...pac.pos };
    ghost.edibleTimer = 10;
    ghost.inBox = false;
    ghost.releaseDelay = 0;
    const scoreBefore = pac.score;

    const res = env.step(0);

    expect(res.done).toBe(false); // eating ≠ dying
    expect(pac.score - scoreBefore).toBe(env.params.reward.ghostEatReward); // combo ×1
    expect(ghost.edibleTimer).toBe(0); // no longer edible
    expect(ghost.pos).toEqual(startTile); // sent back to the pen entrance
  });

  // D4.2: a power pellet makes every ghost edible and resets the eat combo.
  test('eating a power pellet makes all ghosts edible and resets combo (D4.2)', () => {
    const env = new PacmanEnvironment();
    env.params.numGhosts = 2;
    env.params.pacmanSpeed = 0;
    env.params.ghostSpeed = 0;
    env.params.powerPelletDuration = 25;
    env.reset(42);
    const pac = env.getPacmen()[0];
    // Plant a power pellet under Pac and give him a stale combo to clear.
    env.world.powerPellets[pac.pos.y][pac.pos.x] = true;
    env.powerPelletsLeft += 1;
    pac.ghostsEatenCombo = 3;
    env.ghosts.forEach((g) => { g.edibleTimer = 0; });

    env.step(0);

    // edibleTimer is set to powerPelletDuration then ticked once this step → >0.
    expect(env.ghosts.every((g) => g.edibleTimer > 0)).toBe(true);
    expect(pac.ghostsEatenCombo).toBe(0);
  });

  // D4.2: a ghost still serving its releaseDelay isn't "live" and cannot catch
  // Pac-Man even when colocated (guards the houseless-maze death trap, H2).
  test('a ghost still serving releaseDelay cannot catch Pac-Man (D4.2)', () => {
    const env = frozenEnv();
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];
    ghost.pos = { ...pac.pos };
    ghost.edibleTimer = 0;
    ghost.inBox = false; // houseless-style: out of the pen but not yet released
    ghost.releaseDelay = 5;

    const res = env.step(0);

    expect(res.done).toBe(false);
  });

  // D4.2: positive control — a released, non-edible ghost on Pac's tile is a death.
  test('a released ghost on Pac-Man\'s tile is a death (D4.2)', () => {
    const env = frozenEnv();
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];
    ghost.pos = { ...pac.pos };
    ghost.edibleTimer = 0;
    ghost.inBox = false;
    ghost.releaseDelay = 0;

    const res = env.step(0);

    expect(res.done).toBe(true);
    expect(res.reward).toBeLessThan(0); // deathPenalty dominates
  });
});
