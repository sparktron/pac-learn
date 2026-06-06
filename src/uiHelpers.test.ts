import { describe, expect, test } from 'vitest';
import { movingAverage, buildSparkPath, computeDelta, fmtNum, safeNum, safeLocalGet, safeLocalSet } from './uiHelpers';

describe('movingAverage', () => {
  test('partial windows divide by count, full windows by w', () => {
    // w=2: [1, (1+2)/2, (2+3)/2, (3+4)/2]
    expect(movingAverage([1, 2, 3, 4], 2)).toEqual([1, 1.5, 2.5, 3.5]);
  });

  test('window larger than input averages all seen so far', () => {
    expect(movingAverage([2, 4], 10)).toEqual([2, 3]);
  });

  test('empty input → empty output', () => {
    expect(movingAverage([], 5)).toEqual([]);
  });
});

describe('buildSparkPath', () => {
  test('returns empty paths for fewer than 2 points', () => {
    expect(buildSparkPath([])).toEqual({ line: '', fill: '' });
    expect(buildSparkPath([5])).toEqual({ line: '', fill: '' });
  });

  test('produces an SVG path that starts with M for a normal series', () => {
    const { line, fill } = buildSparkPath([1, 5, 2, 8]);
    expect(line.startsWith('M ')).toBe(true);
    expect(fill.startsWith('M 0,90')).toBe(true);
    expect(fill.endsWith('Z')).toBe(true);
  });

  // C6 regression: Math.min(...values) spread threw RangeError past ~125k points.
  test('does not throw on a very long series (C6)', () => {
    const long = new Array(200_000).fill(0).map((_, i) => i % 100);
    expect(() => buildSparkPath(long)).not.toThrow();
    expect(buildSparkPath(long).line.startsWith('M ')).toBe(true);
  });

  test('skips non-finite samples without poisoning the range', () => {
    const { line } = buildSparkPath([1, NaN, 3, Infinity, 2]);
    expect(line.startsWith('M ')).toBe(true);
    expect(line.includes('NaN')).toBe(false);
  });

  test('all-NaN series yields empty paths', () => {
    expect(buildSparkPath([NaN, NaN, NaN])).toEqual({ line: '', fill: '' });
  });
});

describe('computeDelta', () => {
  test('flat for short series', () => {
    expect(computeDelta([1, 2, 3])).toEqual({ pct: 0, dir: 'flat' });
  });

  test('detects an upward trend', () => {
    const d = computeDelta([10, 10, 10, 20]);
    expect(d.dir).toBe('up');
    expect(d.pct).toBeGreaterThan(0);
  });

  test('detects a downward trend', () => {
    expect(computeDelta([100, 100, 100, 50]).dir).toBe('down');
  });

  // M21 regression: tiny |prev| must not produce astronomical percentages.
  test('treats a near-zero baseline as flat (M21)', () => {
    expect(computeDelta([0.001, 0.001, 0.001, 5]).dir).toBe('flat');
    expect(computeDelta([-0.001, -0.001, -0.001, 5])).toEqual({ pct: 0, dir: 'flat' });
  });
});

describe('fmtNum', () => {
  test('formats with the given decimals', () => {
    expect(fmtNum(3.14159, 2)).toBe('3.14');
    expect(fmtNum(42, 0)).toBe('42');
  });

  test('renders an em-dash for non-finite input', () => {
    expect(fmtNum(NaN)).toBe('—');
    expect(fmtNum(Infinity)).toBe('—');
  });
});

describe('safeNum', () => {
  // M9 regression: empty / half-typed input must keep the previous value, not
  // silently become 0 or NaN and propagate into reward calcs.
  test('keeps prev for empty / half-typed input (M9)', () => {
    expect(safeNum('', 7)).toBe(7);
    expect(safeNum('-', 7)).toBe(7);
    expect(safeNum('.', 7)).toBe(7);
    expect(safeNum('1.2.3', 7)).toBe(7);
    expect(safeNum('Infinity', 7)).toBe(7);
  });

  test('parses valid numbers', () => {
    expect(safeNum('42', 0)).toBe(42);
    expect(safeNum('-3.5', 0)).toBe(-3.5);
    expect(safeNum('0', 9)).toBe(0);
  });
});

describe('safe localStorage (D7.10)', () => {
  test('safeLocalGet returns null and safeLocalSet does not throw when storage throws', () => {
    const orig = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    // Simulate private-mode / disabled storage: accessing localStorage throws.
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      get() { throw new Error('SecurityError: storage disabled'); },
    });
    try {
      expect(safeLocalGet('any-key')).toBeNull();
      expect(() => safeLocalSet('any-key', 'v')).not.toThrow();
    } finally {
      if (orig) Object.defineProperty(globalThis, 'localStorage', orig);
      else delete (globalThis as Record<string, unknown>).localStorage;
    }
  });

  test('round-trips a value when storage is available', () => {
    // Provide a minimal in-memory localStorage for the round-trip path.
    const orig = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    const store = new Map<string, string>();
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
        setItem: (k: string, v: string) => { store.set(k, v); },
      },
    });
    try {
      safeLocalSet('pac-learn-algorithm', 'linear');
      expect(safeLocalGet('pac-learn-algorithm')).toBe('linear');
      expect(safeLocalGet('missing')).toBeNull();
    } finally {
      if (orig) Object.defineProperty(globalThis, 'localStorage', orig);
      else delete (globalThis as Record<string, unknown>).localStorage;
    }
  });
});
