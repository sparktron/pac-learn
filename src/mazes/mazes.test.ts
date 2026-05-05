import { describe, expect, test } from 'vitest';
import { generateMaze } from './mazes';

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

  test('ensures ghost starts are open', () => {
    const maze = generateMaze(100);
    maze.ghostStarts.forEach((pos) => {
      expect(maze.grid[pos.y]?.[pos.x]).toBe(0);
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
});
