import type { EnvParams } from '../env/environment';

// Single source of truth for reward presets (D5.11). Previously hand-mirrored in
// three places — App.tsx, scripts/overnight-bench.ts, and presetBench.test.ts —
// where drift caused real bugs (M5 stale γ baseline, N20 'default' diverging from
// the env default). Import from here instead of re-declaring.
//
// Empirical notes:
//   • T2 (2026-07-29) confirmed the default's deathPenalty=-50 and 10× pellet
//     escalation alongside the linear γ=0.997 default: 37.17% vs 33.25% mean
//     greedy wins across five seeds/four panels at 2,000 episodes.
//   • The other presets are kept for experimentation, not because they converge.

export type RewardConfig = EnvParams['reward'];

export const REWARD_PRESETS: Record<string, RewardConfig> = {
  default:             { pelletReward: 5,  powerPelletReward: 20, deathPenalty: -50,  stepPenalty: -0.1,  survivalReward: 0,    ghostEatReward: 30,  winBonus: 1000, pelletEscalationMax: 10, potentialShapingScale: 0, potentialShapingGamma: 0.997, reversePenalty: -2 },
  'ghost-hunting':     { pelletReward: 2,  powerPelletReward: 30, deathPenalty: -50,  stepPenalty: -0.05, survivalReward: 0.01, ghostEatReward: 80,  winBonus: 100,  pelletEscalationMax: 6, potentialShapingScale: 0, potentialShapingGamma: 0.997, reversePenalty: -2 },
  'pellet-collection': { pelletReward: 15, powerPelletReward: 40, deathPenalty: -120, stepPenalty: -0.1,  survivalReward: 0.02, ghostEatReward: 20,  winBonus: 300, pelletEscalationMax: 6, potentialShapingScale: 0, potentialShapingGamma: 0.997, reversePenalty: -2 },
  'survival':          { pelletReward: 3,  powerPelletReward: 20, deathPenalty: -250, stepPenalty: -0.05, survivalReward: 0.2,  ghostEatReward: 50,  winBonus: 100, pelletEscalationMax: 6, potentialShapingScale: 0, potentialShapingGamma: 0.997, reversePenalty: -2 },
};
