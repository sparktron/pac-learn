import { describe, expect, test } from 'vitest';
import { SeededRng } from './prng';

describe('SeededRng', () => {
  test('same seed produces identical sequence (reproducibility)', () => {
    const a = new SeededRng(12345);
    const b = new SeededRng(12345);
    const seqA = Array.from({ length: 20 }, () => a.next());
    const seqB = Array.from({ length: 20 }, () => b.next());
    expect(seqA).toEqual(seqB);
  });

  test('different seeds diverge', () => {
    const a = new SeededRng(1);
    const b = new SeededRng(2);
    expect(a.next()).not.toBe(b.next());
  });

  test('next() is always in [0, 1)', () => {
    const rng = new SeededRng(0xC0FFEE);
    for (let i = 0; i < 10000; i += 1) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  test('seed 0 is well-defined and non-degenerate', () => {
    const rng = new SeededRng(0);
    const v1 = rng.next();
    const v2 = rng.next();
    expect(v1).toBeGreaterThanOrEqual(0);
    expect(v1).toBeLessThan(1);
    expect(v1).not.toBe(v2);
  });

  test('int(n) is always in [0, n)', () => {
    const rng = new SeededRng(42);
    for (let i = 0; i < 10000; i += 1) {
      const v = rng.int(7);
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(7);
    }
  });

  test('int() guards non-positive and NaN bounds (D1.1)', () => {
    const rng = new SeededRng(1);
    expect(rng.int(0)).toBe(0);
    expect(rng.int(-5)).toBe(0);
    expect(rng.int(NaN)).toBe(0);
  });

  test('int(n) covers the full range over many draws', () => {
    const rng = new SeededRng(99);
    const seen = new Set<number>();
    for (let i = 0; i < 2000; i += 1) seen.add(rng.int(4));
    expect(seen).toEqual(new Set([0, 1, 2, 3]));
  });
});
