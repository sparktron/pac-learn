import { describe, expect, test } from 'vitest';
import { DIRECTIONS, actionToDirection, reverseAction, directionToAction, toAction, type Action } from './types';

// These helpers are deliberately exercised with bare numbers cast to Action so
// the converters' own range/clamp behavior is what's under test (not toAction's).
const a = (n: number): Action => n as Action;

describe('direction helpers', () => {
  // These two helpers are the root cause of two historical shipped bugs:
  //   C3  — anti-reversal used (a+2)%4, wrong for the paired DIRECTIONS order
  //   H10 — lastAction was stored unclamped, overflowing the observation key
  // Pin both so a regression can't quietly reintroduce either.

  test('DIRECTIONS order is the canonical paired-axis order', () => {
    expect(DIRECTIONS).toEqual(['up', 'down', 'left', 'right']);
  });

  test('reverseAction flips each action to its true geometric opposite', () => {
    expect(reverseAction(a(0))).toBe(1); // up   -> down
    expect(reverseAction(a(1))).toBe(0); // down -> up
    expect(reverseAction(a(2))).toBe(3); // left -> right
    expect(reverseAction(a(3))).toBe(2); // right-> left
  });

  test('reverseAction is an involution over the action space', () => {
    for (const n of [0, 1, 2, 3]) {
      expect(reverseAction(reverseAction(a(n)))).toBe(n);
    }
  });

  test('actionToDirection maps in-range actions', () => {
    expect(actionToDirection(a(0))).toBe('up');
    expect(actionToDirection(a(1))).toBe('down');
    expect(actionToDirection(a(2))).toBe('left');
    expect(actionToDirection(a(3))).toBe('right');
  });

  test('actionToDirection clamps out-of-range actions (H10 guard)', () => {
    expect(actionToDirection(a(-2))).toBe('up');
    expect(actionToDirection(a(-1))).toBe('up');
    expect(actionToDirection(a(4))).toBe('right');
    expect(actionToDirection(a(99))).toBe('right');
  });

  // D1.5: the whole point of the brand — a bare number can't masquerade as an
  // Action. These @ts-expect-error lines are enforced by `npm run typecheck`:
  // if the brand is ever weakened, the suppressed error vanishes and typecheck
  // fails on the now-unused directive. Runtime assertions just keep the names live.
  test('Action is nominal: bare numbers are rejected, converters mint valid actions', () => {
    // @ts-expect-error a raw number is not assignable to Action
    const bad: Action = 0;
    void bad;
    const fromDir: Action = directionToAction('left');
    const fromRaw: Action = toAction(2);
    expect(fromDir).toBe(2);   // 'left' is index 2 in DIRECTIONS
    expect(fromRaw).toBe(2);
  });
});
