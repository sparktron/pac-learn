export type Direction = 'up' | 'down' | 'left' | 'right';

export const DIRECTIONS: Direction[] = ['up', 'down', 'left', 'right'];

export interface Vec2 {
  x: number;
  y: number;
}

export const DIR_VEC: Record<Direction, Vec2> = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

export const actionToDirection = (action: number): Direction => DIRECTIONS[Math.max(0, Math.min(3, action))];
