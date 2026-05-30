import { describe, expect, test } from 'vitest';
import { generateMaze, MAZES, STATIC_MAZES, validateMaze } from './mazes';

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

  // N9 regression: any maze whose grid contains ghost-house floor tiles ('2')
  // must have a non-null ghostHouseExit that points to a passable (value 0) tile.
  // Before N9, parse() never set ghostHouseExit for custom mazes, leaving the
  // ghost BFS pathfinder without an exit and freezing ghosts in the pen.
  test('every maze with ghost-house floor tiles has a valid ghostHouseExit (N9)', () => {
    const allMazes = [...STATIC_MAZES, ...MAZES].filter(
      (m, i, arr) => arr.findIndex((n) => n.id === m.id) === i, // dedupe
    );
    for (const maze of allMazes) {
      const has2 = maze.grid.some((row) => row.includes(2));
      if (!has2) continue;
      expect(maze.ghostHouseExit, `maze '${maze.id}' has '2' tiles but ghostHouseExit is undefined`).toBeDefined();
      const ex = maze.ghostHouseExit!;
      expect(
        maze.grid[ex.y]?.[ex.x],
        `maze '${maze.id}' ghostHouseExit (${ex.x},${ex.y}) is not an open tile`,
      ).toBe(0);
    }
  });

  // N9: verify the classic maze specifically — it is the only static maze
  // built with createClassicMaze() (not parse()) and must also pass.
  test('pacman-classic ghostHouseExit points to an open tile (N9)', () => {
    const classic = STATIC_MAZES.find((m) => m.id === 'pacman-classic')!;
    expect(classic.ghostHouseExit).toBeDefined();
    const ex = classic.ghostHouseExit!;
    expect(classic.grid[ex.y]?.[ex.x]).toBe(0);
  });

  // D2.5 + D2.3: validateMaze bundles the structural invariants (open starts,
  // unique/open power pellets, ghost-house exit, full reachability). Every
  // shipped maze must pass with zero violations.
  test('every maze passes validateMaze (D2.5)', () => {
    for (const m of MAZES) {
      expect(validateMaze(m), `maze ${m.id}: ${validateMaze(m).join('; ')}`).toEqual([]);
    }
  });

  // D2.3: explicit reachability — no open (pellet-bearing) tile may be
  // isolated from pacStart, or the level is unwinnable.
  test('all open tiles are reachable from pacStart (D2.3)', () => {
    for (const m of MAZES) {
      const violations = validateMaze(m).filter((e) => e.includes('unreachable'));
      expect(violations, `maze ${m.id}`).toEqual([]);
    }
  });

  // D2.4: spawn tiles must be passable (guards the findOpenNear wall-fallback).
  test('pacStart and ghostStarts are passable tiles (D2.4)', () => {
    for (const m of MAZES) {
      expect(m.grid[m.pacStart.y]?.[m.pacStart.x], `maze ${m.id} pacStart`).toBe(0);
      m.ghostStarts.forEach((g, i) => {
        expect([0, 2], `maze ${m.id} ghostStart[${i}]`).toContain(m.grid[g.y]?.[g.x]);
      });
    }
  });

  // D2.4: even input dimensions are bumped to odd so the backtracker works.
  test('generateMaze coerces even dimensions to odd (D2.4)', () => {
    const maze = generateMaze(7, 20, 14);
    expect(maze.grid).toHaveLength(15); // 14 -> 15
    expect(maze.grid[0]).toHaveLength(21); // 20 -> 21
  });

  // D2.1: pacStart must never coincide with an advertised power pellet (the
  // env drops it, so listing it would over-report powerPelletPositions).
  test('pacStart is never an advertised power pellet (D2.1)', () => {
    for (const m of MAZES) {
      const onStart = m.powerPelletPositions.some(
        (p) => p.x === m.pacStart.x && p.y === m.pacStart.y,
      );
      expect(onStart, `maze ${m.id}`).toBe(false);
    }
  });

  // D2.5: validateMaze flags a deliberately corrupted maze (negative test).
  test('validateMaze reports violations on a broken maze (D2.5)', () => {
    const good = generateMaze(100);
    const broken = { ...good, pacStart: { x: 0, y: 0 } }; // (0,0) is a wall
    const errs = validateMaze(broken);
    expect(errs.length).toBeGreaterThan(0);
    expect(errs.some((e) => e.includes('pacStart'))).toBe(true);
  });
});
