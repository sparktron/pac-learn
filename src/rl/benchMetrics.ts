// Pure metric helpers extracted from scripts/overnight-bench.ts (D8.4) so they
// live in src/ scope where they are typechecked and unit-tested. The bench
// itself is outside the tsconfig/vitest include (D8.6), so logic kept inline
// there ships unverified — exactly how D8.1 (a .q crash) slipped through.

/**
 * Classify how an episode ended from its final state.
 *   - `won`     — all pellets cleared (pelletsLeft === 0)
 *   - `timeout` — hit the step cap without winning
 *   - `died`    — caught by a ghost before either
 * `won` takes priority over `timeout`: clearing the last pellet on the final
 * allowed step is a win, not a timeout.
 */
export const inferTermReason = (
  pelletsLeft: number,
  stepCount: number,
  maxSteps: number,
): 'won' | 'timeout' | 'died' => {
  if (pelletsLeft === 0) return 'won';
  if (stepCount >= maxSteps) return 'timeout';
  return 'died';
};

/**
 * Linear-interpolated percentile of a sorted-ascending array.
 * `p` is a fraction in [0,1] (e.g. 0.05 for p5). Empty → NaN; single → element.
 */
export const percentile = (sortedAsc: number[], p: number): number => {
  if (sortedAsc.length === 0) return NaN;
  if (sortedAsc.length === 1) return sortedAsc[0];
  const idx = (sortedAsc.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedAsc[lo];
  const frac = idx - lo;
  return sortedAsc[lo] * (1 - frac) + sortedAsc[hi] * frac;
};
