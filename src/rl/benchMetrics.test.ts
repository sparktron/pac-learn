import { describe, expect, test } from 'vitest';
import { inferTermReason, percentile } from './benchMetrics';

describe('inferTermReason (D8.4)', () => {
  test('pelletsLeft 0 → won', () => {
    expect(inferTermReason(0, 137, 1000)).toBe('won');
  });

  test('step cap reached without clearing → timeout', () => {
    expect(inferTermReason(42, 1000, 1000)).toBe('timeout');
    expect(inferTermReason(42, 1001, 1000)).toBe('timeout');
  });

  test('caught before either → died', () => {
    expect(inferTermReason(42, 200, 1000)).toBe('died');
  });

  test('won takes priority over timeout (last pellet on the final step)', () => {
    expect(inferTermReason(0, 1000, 1000)).toBe('won');
  });
});

describe('percentile (D8.4)', () => {
  test('empty → NaN, single → the element', () => {
    expect(percentile([], 0.5)).toBeNaN();
    expect(percentile([7], 0.05)).toBe(7);
  });

  test('p0 = min, p1 = max', () => {
    const a = [1, 2, 3, 4, 5];
    expect(percentile(a, 0)).toBe(1);
    expect(percentile(a, 1)).toBe(5);
  });

  test('p50 is the median for odd and even lengths', () => {
    expect(percentile([1, 2, 3], 0.5)).toBe(2);
    expect(percentile([10, 20, 30, 40], 0.5)).toBeCloseTo(25, 5); // interpolated
  });

  test('linear interpolation between samples', () => {
    // idx = (2-1)*0.25 = 0.25 → 0*0.75 + 10*0.25 = 2.5
    expect(percentile([0, 10], 0.25)).toBeCloseTo(2.5, 5);
    // idx = (4)*0.05 = 0.2 → 0*0.8 + 5*0.2 = 1.0 over [0,5,10,15,20]
    expect(percentile([0, 5, 10, 15, 20], 0.05)).toBeCloseTo(1.0, 5);
  });
});
