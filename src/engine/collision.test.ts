import { describe, expect, test } from 'vitest';
import { createDefaultEnv } from '../env/environment';

describe('maze collisions', () => {
  test('wall tiles block movement', () => {
    const env = createDefaultEnv();
    env.reset(1);
    const before = { ...env.getPacmen()[0].pos };
    env.step(0); // up into wall — row above pacStart {x:13,y:23} is a wall tile
    expect(env.getPacmen()[0].pos).toEqual(before);
  });
});
