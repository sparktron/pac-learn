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

// A nominal (branded) action: an index into DIRECTIONS, 0..3. It is a `number`
// at runtime (so it still works as an array index / in arithmetic), but the
// brand makes it a *compile error* to pass a bare number — or a number computed
// in some other ordering — where an Action is expected. This is the guard
// against the direction-order bug class that shipped twice (C3, M2/D4.1): the
// only sanctioned ways to mint an Action are the converters below, all anchored
// to DIRECTIONS as the single source of truth.
declare const ActionBrand: unique symbol;
export type Action = number & { readonly [ActionBrand]: true };

// The four valid actions, branded, in DIRECTIONS order. Safe literal source.
export const ACTIONS: readonly Action[] = [0, 1, 2, 3] as Action[];

// Direction → action index. Use this instead of `DIRECTIONS.indexOf(d)` so a
// wrongly-ordered local table can never mint a bogus action.
export const directionToAction = (d: Direction): Action => DIRECTIONS.indexOf(d) as Action;

// The one sanctioned number → Action escape hatch (keyboard codes, loaded
// policies, CLI args). Clamps/truncates into range. Grep `toAction` to audit
// every spot untyped input crosses into the action space.
export const toAction = (n: number): Action => Math.max(0, Math.min(3, Math.trunc(n))) as Action;

export const actionToDirection = (action: Action): Direction => DIRECTIONS[Math.max(0, Math.min(3, action))];

// Tunnel wraparound. Shared by the env and the ghost AI so their movement math
// can never drift apart (D3.1). x always wraps (mazes have horizontal side
// tunnels). y wraps only when `wrapY` is set — i.e. the active maze opts into a
// vertical tunnel (A3); off by default so existing mazes are byte-identical (an
// out-of-bounds y stays out of bounds and is rejected as a wall by callers).
export const wrapPosition = (width: number, height: number, x: number, y: number, wrapY = false): Vec2 => {
  let wx = x;
  if (wx < 0) wx = width - 1;
  else if (wx >= width) wx = 0;
  let wy = y;
  if (wrapY) {
    if (wy < 0) wy = height - 1;
    else if (wy >= height) wy = 0;
  }
  return { x: wx, y: wy };
};

// DIRECTIONS = ['up', 'down', 'left', 'right'] groups each axis as adjacent pairs,
// so reversing an action is a single bit flip: up(0)↔down(1), left(2)↔right(3).
// (The previous `(a + 2) % 4` formula assumed a rotational ['up','right','down','left']
// ordering and was silently wrong against the actual DIRECTIONS array.)
export const reverseAction = (a: Action): Action => (a ^ 1) as Action;
