import { SeededRng } from '../engine/prng';

export interface MazeDefinition {
  id: string;
  name: string;
  grid: number[][]; // 1 wall, 0 open
  pacStart: { x: number; y: number };
  ghostStarts: Array<{ x: number; y: number }>;
  powerPelletPositions: Array<{ x: number; y: number }>;
  wallColor?: string;
}

// ── Static maze layouts ──────────────────────────────────────────────

const m1 = [
  '1111111111111111111',
  '1000000010000000001',
  '1011110101011110101',
  '1010000000000001001',
  '1010111011101110101',
  '1000100000001000001',
  '1110101111101011101',
  '1000100010001000001',
  '1011101010101011101',
  '1000001000100000001',
  '1010111010111010101',
  '1000100000001000001',
  '1011101110101110101',
  '1000000010000000001',
  '1111111111111111111',
];

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

const parse = (id: string, name: string, rows: string[], wallColor?: string): MazeDefinition => {
  const grid = rows.map((r) => r.split('').map((c) => Number(c)));
  const h = grid.length;
  const w = grid[0].length;
  // Place power pellets in the four quadrant corners (away from walls where possible)
  const pp = findPowerPelletPositions(grid, w, h);
  return {
    id,
    name,
    grid,
    pacStart: findOpenNear(grid, 1, 1),
    ghostStarts: [
      findOpenNear(grid, w - 2, 1),
      findOpenNear(grid, w - 2, h - 2),
      findOpenNear(grid, Math.floor(w / 2), Math.floor(h / 2)),
      findOpenNear(grid, 1, h - 2),
    ],
    powerPelletPositions: pp,
    wallColor,
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
  return { x: tx, y: ty };
}

function findPowerPelletPositions(grid: number[][], w: number, h: number): Array<{ x: number; y: number }> {
  const candidates = [
    { x: 1, y: 1 },
    { x: w - 2, y: 1 },
    { x: 1, y: h - 2 },
    { x: w - 2, y: h - 2 },
  ];
  return candidates.map((c) => findOpenNear(grid, c.x, c.y)).filter((p) => grid[p.y][p.x] === 0);
}

export const STATIC_MAZES: MazeDefinition[] = [
  parse('classic', 'Classic', m1, '#1e3a8a'),
  parse('arena', 'Arena', m2, '#6b21a8'),
  parse('corridors', 'Corridors', m3, '#065f46'),
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

  // Create a ghost house in the center
  const cx = Math.floor(w / 2);
  const cy = Math.floor(h / 2);
  // Clear a 3x3 area for the ghost house
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const gx = cx + dx, gy = cy + dy;
      if (gx > 0 && gx < w - 1 && gy > 0 && gy < h - 1) {
        grid[gy][gx] = 0;
      }
    }
  }
  // Ensure ghost house has entry from top
  if (cy - 2 > 0) grid[cy - 2][cx] = 0;

  const pp = findPowerPelletPositions(grid, w, h);

  return {
    id: `proc-${seed}`,
    name: `Procedural #${seed}`,
    grid,
    pacStart: findOpenNear(grid, 1, h - 2),
    ghostStarts: [
      { x: cx, y: cy },
      findOpenNear(grid, cx - 1, cy),
      findOpenNear(grid, cx + 1, cy),
      findOpenNear(grid, cx, cy + 1),
    ],
    powerPelletPositions: pp,
    wallColor: wallColor ?? randomWallColor(rng),
  };
}

function randomWallColor(rng: SeededRng): string {
  const colors = ['#1e3a8a', '#7c2d12', '#6b21a8', '#065f46', '#991b1b', '#0e7490', '#4338ca', '#be185d'];
  return colors[rng.int(colors.length)];
}

// Convenience: generate several procedural mazes
export function generateProcMazes(count = 5, baseSeed = 100): MazeDefinition[] {
  return Array.from({ length: count }, (_, i) => generateMaze(baseSeed + i));
}

export const MAZES: MazeDefinition[] = [
  ...STATIC_MAZES,
  ...generateProcMazes(5, 100),
];
