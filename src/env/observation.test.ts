import { describe, expect, test } from 'vitest';
import { createDefaultEnv } from './environment';
import { observationKey } from './observation';

describe('observation encoding', () => {
  test('is deterministic with same seed', () => {
    const a = createDefaultEnv();
    const b = createDefaultEnv();
    const oa = a.reset(123);
    const ob = b.reset(123);
    expect(observationKey(oa)).toBe(observationKey(ob));
  });
});
