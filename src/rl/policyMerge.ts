import type { SerializedPolicy } from './qlearning';

/**
 * Visit-weighted federated merge of tabular Q-policies (extracted from
 * scripts/merge-policies.ts so it can be unit-tested and typechecked).
 *
 * Merge semantics (per state, per slot):
 *   • Merged Q is the visit-weighted average over workers that actually visited
 *     the slot (weight = √visits, a taper so one hot worker can't drown out its
 *     peers). Slots no worker visited stay at optimisticInit — this is the C1
 *     fix: never average a learned −90 with an untouched init 50.
 *   • The output visitTable is the sum of raw input visits per slot, so a merged
 *     policy resumed in another federated run tapers identically to a first pass.
 *   • ε is reset to the MAX across inputs so a resumed worker explores rather
 *     than running near-greedy from a decayed end-of-training ε.
 *
 * Throws on: empty input, mixed observationKeyVersion, or any policy missing
 * visitTable (legacy format — N14 refuses these loudly rather than guessing).
 */
export interface MergeStats {
  mergedPolicies: number;
  totalStates: number;
  sharedStates: number;
  sharedFraction: number;
  avgWorkersPerState: number;
  slotsWithVisitsUsed: number;
  maxEpsilon: number;
}

export interface MergeResult {
  merged: SerializedPolicy;
  stats: MergeStats;
}

const visitWeight = (v: number): number => (v > 0 ? Math.sqrt(v) : 0);

export const mergePolicies = (policies: SerializedPolicy[]): MergeResult => {
  if (policies.length === 0) throw new Error('mergePolicies: no policies to merge');

  const version = policies[0].observationKeyVersion;
  for (let i = 1; i < policies.length; i += 1) {
    if (policies[i].observationKeyVersion !== version) {
      throw new Error(
        `observationKeyVersion mismatch: policy ${i} has ${policies[i].observationKeyVersion}, expected ${version}`,
      );
    }
  }
  for (let i = 0; i < policies.length; i += 1) {
    if (!policies[i].visitTable) {
      throw new Error(`policy ${i} has no visitTable (legacy format) — re-train with current code`);
    }
  }

  const init = policies[0].hyper.optimisticInit ?? 50;

  const qSums = new Map<string, [number, number, number, number]>();
  const visitSums = new Map<string, [number, number, number, number]>();
  const rawVisitSums = new Map<string, [number, number, number, number]>();
  const stateCount = new Map<string, number>();
  let slotsWithVisitsUsed = 0;

  for (const policy of policies) {
    for (const [key, values] of Object.entries(policy.qTable)) {
      let qSum = qSums.get(key);
      let vSum = visitSums.get(key);
      let rvSum = rawVisitSums.get(key);
      if (!qSum) { qSum = [0, 0, 0, 0]; qSums.set(key, qSum); }
      if (!vSum) { vSum = [0, 0, 0, 0]; visitSums.set(key, vSum); }
      if (!rvSum) { rvSum = [0, 0, 0, 0]; rawVisitSums.set(key, rvSum); }
      stateCount.set(key, (stateCount.get(key) ?? 0) + 1);

      const slotVisits = policy.visitTable![key];
      for (let i = 0; i < 4; i += 1) {
        const q = values[i];
        if (q === undefined) continue;
        const rawVisits = slotVisits?.[i] ?? 0;
        const w = visitWeight(rawVisits);
        if (w > 0) {
          slotsWithVisitsUsed += 1;
          qSum[i] += q * w;
          vSum[i] += w;
          rvSum[i] += rawVisits;
        }
      }
    }
  }

  const mergedQ: Record<string, number[]> = {};
  const mergedVisits: Record<string, number[]> = {};
  let sharedStates = 0;
  for (const [key, qSum] of qSums.entries()) {
    if ((stateCount.get(key) ?? 0) > 1) sharedStates += 1;
    const vSum = visitSums.get(key)!;
    const rvSum = rawVisitSums.get(key)!;
    mergedQ[key] = qSum.map((s, i) => (vSum[i] > 0 ? s / vSum[i] : init));
    mergedVisits[key] = [rvSum[0], rvSum[1], rvSum[2], rvSum[3]];
  }

  const totalCoverage = Array.from(stateCount.values()).reduce((a, b) => a + b, 0);
  const maxEpsilon = policies.reduce((m, p) => Math.max(m, p.hyper.epsilon), 0);

  const merged: SerializedPolicy = {
    algorithm: policies[0].algorithm,
    mazeId: policies[0].mazeId,
    timestamp: new Date().toISOString(),
    numGhostsEncoded: policies[0].numGhostsEncoded,
    observationKeyVersion: policies[0].observationKeyVersion,
    hyper: { ...policies[0].hyper, epsilon: maxEpsilon },
    qTable: mergedQ,
    visitTable: mergedVisits,
  };

  return {
    merged,
    stats: {
      mergedPolicies: policies.length,
      totalStates: stateCount.size,
      sharedStates,
      sharedFraction: stateCount.size > 0 ? sharedStates / stateCount.size : 0,
      avgWorkersPerState: stateCount.size > 0 ? totalCoverage / stateCount.size : 0,
      slotsWithVisitsUsed,
      maxEpsilon,
    },
  };
};
