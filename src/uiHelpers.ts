// Pure UI helpers extracted from App.tsx (D7.7) so they can be unit-tested.
// These have a history of bugs (C6 spark crash on long arrays, M21 tiny-prev
// delta, M9 empty/half-typed input) — exactly the kind of logic that warrants
// regression coverage independent of the React tree.

// Rolling-window average in O(n) (was O(n·w) via Array.slice+reduce per index).
// On a 100k-episode run with w=20 the old form ran 2M operations per render.
export const movingAverage = (values: number[], w: number): number[] => {
  const out = new Array<number>(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
    if (i >= w) sum -= values[i - w];
    out[i] = sum / Math.min(i + 1, w);
  }
  return out;
};

export const buildSparkPath = (values: number[]): { line: string; fill: string } => {
  if (values.length < 2) return { line: '', fill: '' };
  // Avoid Math.min(...values) — spread on a long array (>~125k) throws
  // RangeError in V8. Long training runs would crash the entire spark
  // render. NaNs are skipped so a single bad sample doesn't poison the
  // whole y-range.
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) return { line: '', fill: '' };
  const span = Math.max(0.0001, mx - mn);
  const H = 90;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * 400;
    const safeV = Number.isFinite(v) ? v : mn;
    const y = H - 6 - ((safeV - mn) / span) * (H - 16);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return {
    line: `M ${pts.join(' L ')}`,
    fill: `M 0,${H} L ${pts.join(' L ')} L 400,${H} Z`,
  };
};

export const computeDelta = (values: number[]): { pct: number; dir: 'up' | 'down' | 'flat' } => {
  if (values.length < 4) return { pct: 0, dir: 'flat' };
  const recent = values[values.length - 1];
  const prev = values[Math.max(0, values.length - Math.max(2, Math.floor(values.length * 0.25)))];
  // Guard against tiny |prev| producing astronomical percentages displayed
  // as "▲ 999999.9%" — the strict prev === 0 check let prev = -0.001 through.
  if (!Number.isFinite(prev) || !Number.isFinite(recent) || Math.abs(prev) < 0.01) {
    return { pct: 0, dir: 'flat' };
  }
  const pct = ((recent - prev) / Math.abs(prev)) * 100;
  if (Math.abs(pct) < 0.05) return { pct: 0, dir: 'flat' };
  return { pct: Math.abs(pct), dir: pct >= 0 ? 'up' : 'down' };
};

export const fmtNum = (v: number, decimals = 1): string => {
  if (!isFinite(v)) return '—';
  return v.toFixed(decimals);
};

// Parse a numeric input value, keeping the previous value when the input is
// empty/half-typed. Without this, clearing a field gives Number('') === 0
// (silently resets numGhosts/etc to zero) and typing '-' alone gives NaN,
// which then propagates into every reward calculation and Q-update.
export const safeNum = (raw: string, prev: number): number => {
  if (raw === '' || raw === '-' || raw === '.') return prev;
  const n = Number(raw);
  return Number.isFinite(n) ? n : prev;
};

// D7.10: localStorage throws in private-mode / disabled-storage / SSR contexts.
// Reading it directly in a useState initializer would crash the whole React
// tree on mount. These wrappers degrade gracefully to "no persistence".
export const safeLocalGet = (key: string): string | null => {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};

export const safeLocalSet = (key: string, value: string): void => {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable — skip persistence */
  }
};
