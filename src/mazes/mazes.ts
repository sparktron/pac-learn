export interface MazeDefinition {
  id: string;
  name: string;
  grid: number[][]; // 1 wall, 0 open
  pacStart: { x: number; y: number };
  ghostStarts: Array<{ x: number; y: number }>;
}

const m1 = [
  '1111111111111',
  '1000001000001',
  '1011101011101',
  '1000000000001',
  '1010111110101',
  '1000100010001',
  '1110101010111',
  '1000001000001',
  '1011101011101',
  '1000000000001',
  '1111111111111',
];

const m2 = [
  '111111111111111',
  '100000010000001',
  '101111010111101',
  '101000000000101',
  '101011111110101',
  '100010000010001',
  '111010111010111',
  '100000101000001',
  '101110101011101',
  '100000000000001',
  '111111111111111',
];

const m3 = [
  '1111111111111',
  '1000100001001',
  '1010101101011',
  '1000001000001',
  '1110101010111',
  '1000100010001',
  '1011101011101',
  '1000000000001',
  '1010111110101',
  '1000001000001',
  '1111111111111',
];

const parse = (id: string, name: string, rows: string[]): MazeDefinition => ({
  id,
  name,
  grid: rows.map((r) => r.split('').map((c) => Number(c))),
  pacStart: { x: 1, y: 1 },
  ghostStarts: [{ x: rows[0].length - 2, y: 1 }, { x: rows[0].length - 2, y: rows.length - 2 }],
});

export const MAZES: MazeDefinition[] = [parse('classic', 'Classic', m1), parse('spiral', 'Spiral', m2), parse('lanes', 'Lanes', m3)];
