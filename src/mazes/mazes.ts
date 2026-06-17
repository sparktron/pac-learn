import { SeededRng } from '../engine/prng';

export interface MazeDefinition {
  id: string;
  name: string;
  grid: number[][]; // 1 wall, 0 open, 2 ghost-house floor (passable, no pellets)
  pacStart: { x: number; y: number };
  ghostStarts: Array<{ x: number; y: number }>;
  powerPelletPositions: Array<{ x: number; y: number }>;
  wallColor?: string;
  ghostHouseExit?: { x: number; y: number }; // first open tile outside the ghost house
  /**
   * A3: when true, movement wraps top↔bottom as well as left↔right, so an open
   * tile on the top edge connects to the aligned open tile on the bottom edge
   * (a vertical tunnel). Default/undefined = false → only the horizontal side
   * tunnels wrap, exactly as before.
   */
  verticalTunnel?: boolean;
}

// ── Static maze layouts ──────────────────────────────────────────────

const m2 = [
  '111111111111111111111',
  '100000000100000000001',
  '101111010101011110101',
  '100001000000010000001',
  '101101011111010110101',
  '100000010001000000001',
  '111011010101011011101',
  '100010000100000100001',
  '101010111010111010101',
  '100010001000100010001',
  '101110101010101011101',
  '100000100010001000001',
  '101011101110111010101',
  '100000000000000000001',
  '101111010111010111101',
  '100000010001010000001',
  '111111111111111111111',
];

const m3 = [
  '11111111111111111',
  '10000001000000001',
  '10111010101110101',
  '10001000000010001',
  '11101011101011101',
  '10000010001000001',
  '10110110101101101',
  '10000000100000001',
  '10111010101011101',
  '10100010001000101',
  '10101110111011101',
  '10000000000000001',
  '11111111111111111',
];

// A3 demo maze: a clear central column (col 6) open from the top edge to the
// bottom edge, with the rest of the border walled. With verticalTunnel=true,
// stepping up off the top mouth (6,0) wraps to the bottom mouth (6,12). All
// open tiles are also connected internally, so validateMaze reachability holds
// even without the wrap (the tunnel is a loop, not the only path).
const mVerticalLoop = [
  '1111110111111',
  '1000000000001',
  '1010100010101',
  '1000000000001',
  '1010100010101',
  '1000000000001',
  '1010100010101',
  '1000000000001',
  '1010100010101',
  '1000000000001',
  '1010100010101',
  '1000000000001',
  '1111110111111',
];

const parse = (id: string, name: string, rows: string[], wallColor?: string, verticalTunnel = false): MazeDefinition => {
  const grid = rows.map((r) => r.split('').map((c) => Number(c)));
  const h = grid.length;
  const w = grid[0].length;
  const pacStart = findOpenNear(grid, 1, 1);
  const ghostStarts = [
    findOpenNear(grid, w - 2, 1),
    findOpenNear(grid, w - 2, h - 2),
    findOpenNear(grid, Math.floor(w / 2), Math.floor(h / 2)),
    findOpenNear(grid, 1, h - 2),
  ];
  // Place power pellets in the four quadrant corners (away from walls where
  // possible). Skip any that resolve onto pacStart: the env drops a power
  // pellet on the start tile (environment.ts), so advertising it here makes
  // powerPelletPositions over-report vs. what is actually placed (D2.1).
  const pp = findPowerPelletPositions(grid, w, h, pacStart);
  // N9: if the maze includes any ghost-house floor tiles (value 2), the env
  // needs a ghostHouseExit so in-box ghosts can BFS their way out. Without
  // one, chooseGhostMove returns null and the ghosts sit frozen in the pen.
  // Auto-detect: the topmost open tile (value 0) directly above the topmost
  // '2' tile. If no such tile exists, throw — silently shipping a broken
  // maze is worse than a loud error at module load.
  let ghostHouseExit: { x: number; y: number } | undefined;
  const has2 = grid.some((row) => row.includes(2));
  if (has2) {
    outer:
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        if (grid[y][x] !== 2) continue;
        for (let yy = y - 1; yy >= 0; yy -= 1) {
          if (grid[yy][x] === 0) { ghostHouseExit = { x, y: yy }; break outer; }
          if (grid[yy][x] === 1) break;
        }
      }
    }
    if (!ghostHouseExit) {
      throw new Error(`maze '${id}' has '2' tiles but no open exit tile could be auto-detected above any of them`);
    }
  }
  return {
    id,
    name,
    grid,
    pacStart,
    ghostStarts,
    powerPelletPositions: pp,
    ghostHouseExit,
    wallColor,
    verticalTunnel,
  };
};

function findOpenNear(grid: number[][], tx: number, ty: number): { x: number; y: number } {
  if (grid[ty]?.[tx] === 0) return { x: tx, y: ty };
  for (let r = 1; r < 5; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const y = ty + dy, x = tx + dx;
        if (y > 0 && x > 0 && y < grid.length - 1 && x < grid[0].length - 1 && grid[y][x] === 0) return { x, y };
      }
    }
  }
  // D2.2: rather than fall back to the (likely wall) target tile — which would
  // spawn Pac/ghosts inside a wall — scan the whole interior for any open tile.
  // Only a maze with zero open tiles (degenerate) returns the original target.
  for (let y = 1; y < grid.length - 1; y += 1) {
    for (let x = 1; x < grid[0].length - 1; x += 1) {
      if (grid[y][x] === 0) return { x, y };
    }
  }
  return { x: tx, y: ty };
}

function findPowerPelletPositions(
  grid: number[][],
  w: number,
  h: number,
  pacStart?: { x: number; y: number },
): Array<{ x: number; y: number }> {
  const candidates = [
    { x: 1, y: 1 },
    { x: w - 2, y: 1 },
    { x: 1, y: h - 2 },
    { x: w - 2, y: h - 2 },
  ];
  const seen = new Set<number>();
  const out: Array<{ x: number; y: number }> = [];
  for (const c of candidates) {
    const p = findOpenNear(grid, c.x, c.y);
    if (grid[p.y][p.x] !== 0) continue;
    // D2.1: the env never places a power pellet on pacStart, so omit it here.
    if (pacStart && p.x === pacStart.x && p.y === pacStart.y) continue;
    // Dedupe: on small/tight procedural mazes two corners can resolve to
    // the same open tile via findOpenNear. The boolean power-pellet grid
    // is idempotent so the count stays correct, but downstream consumers
    // that iterate this array (length-based) would double-count.
    const k = p.y * w + p.x;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(p);
  }
  return out;
}

// Returns the set of tiles reachable from (sx,sy) over passable tiles
// (0 = open, 2 = ghost-house floor), encoded as y*w+x. Walls (1) block.
function reachableFrom(grid: number[][], sx: number, sy: number): Set<number> {
  const h = grid.length;
  const w = grid[0].length;
  const seen = new Set<number>();
  const passable = (x: number, y: number) =>
    y >= 0 && y < h && x >= 0 && x < w && (grid[y][x] === 0 || grid[y][x] === 2);
  if (!passable(sx, sy)) return seen;
  const stack = [[sx, sy] as const];
  seen.add(sy * w + sx);
  while (stack.length) {
    const [x, y] = stack.pop()!;
    for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]] as const) {
      const nx = x + dx;
      const ny = y + dy;
      const k = ny * w + nx;
      if (passable(nx, ny) && !seen.has(k)) {
        seen.add(k);
        stack.push([nx, ny]);
      }
    }
  }
  return seen;
}

/**
 * Validate a maze's structural invariants. Returns a list of human-readable
 * violation messages (empty = valid). Centralizes the checks that several
 * historical bugs slipped past (start-on-wall, phantom/dup power pellets,
 * missing ghost-house exit, unreachable pellets that make a maze unwinnable).
 */
export function validateMaze(m: MazeDefinition): string[] {
  const errs: string[] = [];
  const grid = m.grid;
  const h = grid.length;
  const w = h ? grid[0].length : 0;
  const at = (p: { x: number; y: number }) => grid[p.y]?.[p.x];

  if (!h || !w) {
    errs.push('grid is empty');
    return errs;
  }

  if (at(m.pacStart) !== 0) {
    errs.push(`pacStart (${m.pacStart.x},${m.pacStart.y}) is not an open tile`);
  }
  m.ghostStarts.forEach((g, i) => {
    if (at(g) !== 0 && at(g) !== 2) {
      errs.push(`ghostStart[${i}] (${g.x},${g.y}) is not passable`);
    }
  });

  const ppSeen = new Set<number>();
  m.powerPelletPositions.forEach((p) => {
    if (at(p) !== 0) errs.push(`power pellet (${p.x},${p.y}) is not an open tile`);
    const k = p.y * w + p.x;
    if (ppSeen.has(k)) errs.push(`duplicate power pellet at (${p.x},${p.y})`);
    ppSeen.add(k);
  });

  const has2 = grid.some((row) => row.includes(2));
  if (has2) {
    if (!m.ghostHouseExit) {
      errs.push("maze has ghost-house ('2') tiles but no ghostHouseExit");
    } else if (at(m.ghostHouseExit) !== 0) {
      errs.push(`ghostHouseExit (${m.ghostHouseExit.x},${m.ghostHouseExit.y}) is not an open tile`);
    }
  }

  // Reachability: every open (pellet-bearing) tile must be reachable from
  // pacStart, otherwise the level can never be won.
  const reach = reachableFrom(grid, m.pacStart.x, m.pacStart.y);
  let unreachable = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (grid[y][x] === 0 && !reach.has(y * w + x)) unreachable += 1;
    }
  }
  if (unreachable > 0) {
    errs.push(`${unreachable} open tile(s) unreachable from pacStart (unwinnable)`);
  }

  return errs;
}

// Classic Pac-Man arcade maze — authentic 28×31 layout.
// Tile values: 1=wall, 0=open path (pellets placed here), 2=ghost-house floor (passable, no pellets)
const createClassicMaze = (): MazeDefinition => {
  const mazeStr = [
    '1111111111111111111111111111',
    '1000000000000110000000000001',
    '1011110111110110111110111101',
    '1011110111110110111110111101',
    '1011110111110110111110111101',
    '1000000000000000000000000001',
    '1011110110111111110110111101',
    '1011110110111111110110111101',
    '1000000110000110000110000001',
    '1111110111110110111110111111',
    '1111110111110110111110111111',
    '1111110110000000000110111111',
    '1111110110111001110110111111',
    '2222220110122222210112222222',
    '1111110110122222210110111111',
    '1111110110122222210110111111',
    '1111110110111111110110111111',
    '1111110110000000000110111111',
    '1111110110111111110110111111',
    '1111110110111111110110111111',
    '1000000000000110000000000001',
    '1011110111110110111110111101',
    '1011110111110110111110111101',
    '1000110000000000000000110001',
    '1101110110111111110110111011',
    '1101110110111111110110111011',
    '1000000110000000000110000001',
    '1011110110111111110110111101',
    '1011110110111111110110111101',
    '1000000000000000000000000001',
    '1111111111111111111111111111',
  ];
  const grid = mazeStr.map((r) => r.split('').map((c) => Number(c)));

  return {
    id: 'pacman-classic',
    name: 'Pac-Man Classic',
    grid,
    pacStart: { x: 13, y: 23 },
    ghostStarts: [
      { x: 14, y: 14 },
      { x: 13, y: 14 },
      { x: 15, y: 14 },
      { x: 14, y: 15 },
    ],
    powerPelletPositions: [
      { x: 1, y: 3 },
      { x: 26, y: 3 },
      { x: 1, y: 23 },
      { x: 26, y: 23 },
    ],
    ghostHouseExit: { x: 13, y: 11 },
    wallColor: '#1e3a8a',
  };
};

export const STATIC_MAZES: MazeDefinition[] = [
  createClassicMaze(),
  parse('arena', 'Arena', m2, '#6b21a8'),
  parse('corridors', 'Corridors', m3, '#065f46'),
  parse('vertical-loop', 'Vertical Loop', mVerticalLoop, '#0e7490', true),
];

// ── Procedural maze generation ──────────────────────────────────────

export function generateMaze(seed: number, width = 21, height = 15, wallColor?: string): MazeDefinition {
  // Ensure odd dimensions for proper maze gen
  const w = width % 2 === 0 ? width + 1 : width;
  const h = height % 2 === 0 ? height + 1 : height;
  const rng = new SeededRng(seed);

  // Start with all walls
  const grid: number[][] = Array.from({ length: h }, () => Array.from({ length: w }, () => 1));

  // Recursive backtracker to carve passages
  const visited = Array.from({ length: h }, () => Array.from({ length: w }, () => false));

  function carve(cx: number, cy: number): void {
    visited[cy][cx] = true;
    grid[cy][cx] = 0;

    // Shuffle directions
    const dirs = [
      { dx: 0, dy: -2 },
      { dx: 0, dy: 2 },
      { dx: -2, dy: 0 },
      { dx: 2, dy: 0 },
    ];
    // Fisher-Yates shuffle
    for (let i = dirs.length - 1; i > 0; i--) {
      const j = rng.int(i + 1);
      [dirs[i], dirs[j]] = [dirs[j], dirs[i]];
    }

    for (const { dx, dy } of dirs) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx > 0 && nx < w - 1 && ny > 0 && ny < h - 1 && !visited[ny][nx]) {
        // Carve wall between current and next
        grid[cy + dy / 2][cx + dx / 2] = 0;
        carve(nx, ny);
      }
    }
  }

  // Start carving from (1, 1)
  carve(1, 1);

  // Open extra passages to create loops (mazes with no loops are bad for pac-man)
  const extraPassages = Math.floor((w * h) * 0.08);
  for (let i = 0; i < extraPassages; i++) {
    const x = 1 + rng.int(Math.floor((w - 2) / 2)) * 2;
    const y = 1 + rng.int(Math.floor((h - 2) / 2)) * 2;
    // Try to open a random adjacent wall
    const dirs = [
      { dx: 1, dy: 0 },
      { dx: -1, dy: 0 },
      { dx: 0, dy: 1 },
      { dx: 0, dy: -1 },
    ];
    const d = dirs[rng.int(4)];
    const wx = x + d.dx, wy = y + d.dy;
    if (wx > 0 && wx < w - 1 && wy > 0 && wy < h - 1 && grid[wy][wx] === 1) {
      // Check that both sides are open
      const bx = x + d.dx * 2, by = y + d.dy * 2;
      if (bx > 0 && bx < w - 1 && by > 0 && by < h - 1 && grid[by][bx] === 0) {
        grid[wy][wx] = 0;
      }
    }
  }

  // Create a ghost house in the center (tile value 2 = ghost-house floor)
  const cx = Math.floor(w / 2);
  const cy = Math.floor(h / 2);
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const gx = cx + dx, gy = cy + dy;
      if (gx > 0 && gx < w - 1 && gy > 0 && gy < h - 1) {
        grid[gy][gx] = 2;
      }
    }
  }
  // Open a corridor from the ghost house to the maze above it
  if (cy - 2 > 0) grid[cy - 2][cx] = 0;

  const pacStart = findOpenNear(grid, 1, h - 2);
  const pp = findPowerPelletPositions(grid, w, h, pacStart);

  return {
    id: `proc-${seed}`,
    name: `Procedural #${seed}`,
    grid,
    pacStart,
    ghostStarts: [
      { x: cx, y: cy },
      findOpenNear(grid, cx - 1, cy),
      findOpenNear(grid, cx + 1, cy),
      findOpenNear(grid, cx, cy + 1),
    ],
    powerPelletPositions: pp,
    ghostHouseExit: cy - 2 > 0 ? { x: cx, y: cy - 2 } : undefined,
    wallColor: wallColor ?? randomWallColor(rng),
  };
}

function randomWallColor(rng: SeededRng): string {
  const colors = ['#1e3a8a', '#7c2d12', '#6b21a8', '#065f46', '#991b1b', '#0e7490', '#4338ca', '#be185d'];
  return colors[rng.int(colors.length)];
}

// Lazy-loaded procedural maze cache (generated on-demand, not at startup)
const procMazeCache = new Map<number, MazeDefinition>();

function getProcMaze(seed: number): MazeDefinition {
  if (!procMazeCache.has(seed)) {
    procMazeCache.set(seed, generateMaze(seed));
  }
  return procMazeCache.get(seed)!;
}

// Combine static and procedurally-generated mazes. Procedural seeds are
// listed centrally so adding a static maze doesn't silently overwrite an
// existing procedural slot (the previous hard-coded indices 3..7 would).
const PROC_SEEDS = [100, 101, 102, 103, 104] as const;

// Placeholder array: static mazes first, then one slot per procedural seed
// that we replace with a lazy getter below.
export const MAZES: MazeDefinition[] = [
  ...STATIC_MAZES,
  ...PROC_SEEDS.map((s) => ({
    id: `proc-${s}`, name: `Procedural #${s}`, grid: [], pacStart: { x: 0, y: 0 }, ghostStarts: [], powerPelletPositions: [],
  })),
];

// Replace the procedural slots with lazy getters that materialize on first read.
PROC_SEEDS.forEach((seed, i) => {
  Object.defineProperty(MAZES, STATIC_MAZES.length + i, {
    get: () => getProcMaze(seed),
    enumerable: true,
  });
});
MAZES.length = STATIC_MAZES.length + PROC_SEEDS.length;
