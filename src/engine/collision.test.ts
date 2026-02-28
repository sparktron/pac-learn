import { describe, expect, test } from 'vitest';
import { createDefaultEnv } from '../env/environment';

describe('maze collisions', () => {
  test('wall tiles block movement', () => {
    const env = createDefaultEnv();
    env.reset(1);
    const before = { ...env.getPacmen()[0].pos };
    env.step(2); // left into wall on default start
    expect(env.getPacmen()[0].pos).toEqual(before);
  });
});
