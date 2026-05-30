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

// Tunnel wraparound on the left/right edges. Shared by the env and the ghost
// AI so their movement math can never drift apart (D3.1). x-axis only — no
// maze currently uses a vertical tunnel; `height` is accepted for a future
// y-wrap without a signature change.
export const wrapPosition = (width: number, _height: number, x: number, y: number): Vec2 => {
  let wx = x;
  if (wx < 0) wx = width - 1;
  else if (wx >= width) wx = 0;
  return { x: wx, y };
};

// DIRECTIONS = ['up', 'down', 'left', 'right'] groups each axis as adjacent pairs,
// so reversing an action is a single bit flip: up(0)↔down(1), left(2)↔right(3).
// (The previous `(a + 2) % 4` formula assumed a rotational ['up','right','down','left']
// ordering and was silently wrong against the actual DIRECTIONS array.)
export const reverseAction = (a: number): number => a ^ 1;
