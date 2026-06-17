import { describe, expect, test } from 'vitest';
import { PacmanEnvironment, createDefaultEnv } from '../env/environment';
import { toAction } from '../engine/types';

describe('maze collisions', () => {
  test('wall tiles block movement', () => {
    const env = createDefaultEnv();
    env.reset(1);
    const before = { ...env.getPacmen()[0].pos };
    env.step(toAction(0)); // up into wall — row above pacStart {x:13,y:23} is a wall tile
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

    const res = env.step(toAction(0));

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

    env.step(toAction(0));

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

    const res = env.step(toAction(0));

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

    const res = env.step(toAction(0));

    expect(res.done).toBe(true);
    expect(res.reward).toBeLessThan(0); // deathPenalty dominates
  });

  // D4.2 / issue #22: the cross-over swap branch. Under 'tile' rules a pac and
  // ghost can trade tiles in a single tick (pass through each other) and end on
  // *different* tiles — so `sameTile` can't see it and only the `crossOver`
  // branch catches the capture. The audit flagged this branch as *possibly*
  // dead, conjecturing a chaser never steps onto pac's just-vacated tile. It's
  // reachable: in a 1-wide corridor the chaser, finding pac now on its own
  // tile, tie-breaks (up>left>down>right) onto pac's old tile — a true swap.
  //
  // Corridor: only row y0 is open, so both entities can move left/right only.
  // Speeds are exactly 1 → movementIterations returns 1 with no RNG draw, so
  // the whole scenario is deterministic.
  const corridorEnv = (y0: number): PacmanEnvironment => {
    const env = new PacmanEnvironment();
    env.params.captureRules = 'tile';
    env.params.numGhosts = 1;
    env.params.pacmanSpeed = 1;
    env.params.ghostSpeed = 1;
    env.reset(42);
    env.world.isWall = (_x, y) => y !== y0; // 1-wide horizontal hall on row y0
    return env;
  };

  // y0 = the classic maze's pac-start row — far from the ghost house, so the
  // corridor tiles are never treated as ghost-house tiles by the (un-overridden)
  // isGhostHouse, and the ghost's flee/chase candidates are just {left,right}.
  const CORRIDOR_ROW = 23;

  test('tile-mode cross-over swap is a death — sameTile misses it, crossOver catches it (D4.2, #22)', () => {
    const env = corridorEnv(CORRIDOR_ROW);
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];
    pac.pos = { x: 12, y: CORRIDOR_ROW };
    ghost.pos = { x: 13, y: CORRIDOR_ROW };
    ghost.edibleTimer = 0;
    ghost.inBox = false;
    ghost.releaseDelay = 0;
    ghost.lastDir = null;

    const res = env.step(toAction(3)); // move right, onto the ghost's tile

    // True swap: each ended on the other's previous tile. They are on DIFFERENT
    // tiles (dx=1), so sameTile is false — the capture can only have come from
    // the crossOver branch.
    expect(pac.pos).toEqual({ x: 13, y: CORRIDOR_ROW });
    expect(ghost.pos).toEqual({ x: 12, y: CORRIDOR_ROW });
    expect(res.done).toBe(true);
    expect(res.reward).toBeLessThan(0); // deathPenalty
  });

  test('tile-mode cross-over with an edible ghost is an eat, not a death (D4.2, #22)', () => {
    const env = corridorEnv(CORRIDOR_ROW);
    const pac = env.getPacmen()[0];
    const ghost = env.ghosts[0];
    const startTile = { ...ghost.pos }; // real spawn — eaten ghosts reset here
    pac.pos = { x: 12, y: CORRIDOR_ROW };
    ghost.pos = { x: 13, y: CORRIDOR_ROW };
    ghost.edibleTimer = 10; // frightened → fleeing, swaps left into pac's old tile
    ghost.inBox = false;
    ghost.releaseDelay = 0;
    ghost.lastDir = null;
    const scoreBefore = pac.score;

    const res = env.step(toAction(3)); // swap into the ghost

    expect(res.done).toBe(false); // eating ≠ dying
    expect(pac.score).toBeGreaterThan(scoreBefore); // combo reward credited
    expect(ghost.pos).toEqual(startTile); // sent back to spawn after the eat
  });
});
