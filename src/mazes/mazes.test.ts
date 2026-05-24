import { describe, expect, test } from 'vitest';
import { generateMaze, MAZES } from './mazes';

describe('maze generation', () => {
  test('generates valid maze with correct dimensions', () => {
    const maze = generateMaze(100, 21, 15);
    expect(maze.grid).toHaveLength(15);
    expect(maze.grid[0]).toHaveLength(21);
  });

  test('ensures start position is open', () => {
    const maze = generateMaze(100);
    const { pacStart, grid } = maze;
    expect(grid[pacStart.y]?.[pacStart.x]).toBe(0);
  });

  test('ensures ghost starts are passable (open or ghost-house floor)', () => {
    const maze = generateMaze(100);
    maze.ghostStarts.forEach((pos) => {
      expect([0, 2]).toContain(maze.grid[pos.y]?.[pos.x]);
    });
  });

  test('ensures power pellet positions are open', () => {
    const maze = generateMaze(100);
    maze.powerPelletPositions.forEach((pos) => {
      expect(maze.grid[pos.y]?.[pos.x]).toBe(0);
    });
  });

  test('generates same maze for same seed', () => {
    const maze1 = generateMaze(100);
    const maze2 = generateMaze(100);
    expect(maze1.grid).toEqual(maze2.grid);
  });

  test('generates different mazes for different seeds', () => {
    const maze1 = generateMaze(100);
    const maze2 = generateMaze(101);
    // Mazes should be structurally different
    const diff = maze1.grid.flat().filter((v, i) => v !== maze2.grid.flat()[i]).length;
    expect(diff).toBeGreaterThan(0);
  });

  // M14 regression: corner-rounding via findOpenNear can collapse two corners
  // onto the same tile in tight procedural mazes; the dedup must drop the dup.
  test('powerPelletPositions are unique within a maze', () => {
    for (const m of MAZES) {
      const keys = new Set(m.powerPelletPositions.map((p) => `${p.x},${p.y}`));
      expect(keys.size, `maze ${m.id} has duplicate power pellet positions`).toBe(m.powerPelletPositions.length);
    }
  });

  // M16 regression: MAZES.length must equal static+procedural, not a hard-coded
  // constant. Adding a new static maze must not silently overwrite a procedural slot.
  test('MAZES.length matches static + procedural slot count', () => {
    const ids = MAZES.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain('pacman-classic');
  });
});
