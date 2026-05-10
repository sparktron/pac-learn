import { describe, expect, test, beforeEach } from 'vitest';
import { chooseGhostMove } from './ghostAi';
import type { GhostState, WorldState } from '../env/environment';
import type { Vec2 } from '../engine/types';

describe('ghost AI', () => {
  let world: WorldState;
  let ghost: GhostState;

  beforeEach(() => {
    // Simple 5x5 grid with walls on edges, open in middle
    const isWall = (x: number, y: number) => x <= 0 || x >= 4 || y <= 0 || y >= 4;
    world = {
      width: 5,
      height: 5,
      pellets: [],
      powerPellets: [],
      heatmap: Array.from({ length: 5 }, () => Array(5).fill(0)),
      isWall,
    };
    ghost = { id: 0, pos: { x: 2, y: 2 }, aiType: 'classic', edibleTimer: 0, releaseDelay: 0 };
  });

  test('classic AI moves toward Pac-Man', () => {
    const pacPos: Vec2 = { x: 3, y: 2 };
    const move = chooseGhostMove(world, ghost, pacPos);
    // Should move right toward Pac-Man
    expect(move).toBe('right');
  });

  test('classic AI avoids walls', () => {
    const pacPos: Vec2 = { x: 0, y: 0 }; // Off the board (wall)
    const move = chooseGhostMove(world, ghost, pacPos);
    // Should return a valid move (not null)
    expect(['up', 'down', 'left', 'right']).toContain(move);
  });

  test('returns null when completely surrounded', () => {
    // Create a ghost in an isolated 1x1 space (surrounded by walls)
    ghost.pos = { x: 2, y: 2 };
    const isolatedWorld: WorldState = {
      ...world,
      isWall: (x, y) => (x === 2 && y === 2) ? false : true, // Only (2,2) is open
    };
    const move = chooseGhostMove(isolatedWorld, ghost, { x: 0, y: 0 });
    expect(move).toBeNull();
  });
});
