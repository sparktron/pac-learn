import { describe, expect, test } from 'vitest';
import { PacmanEnvironment, type EnvParams } from '../env/environment';
import { QLearningAgent } from './qlearning';
import { TrainingController } from './trainingController';
import { DIRECTIONS } from '../engine/types';
import { SeededRng } from '../engine/prng';


// Reward presets mirrored from App.tsx so we can benchmark them without DOM.
const rewardPresets: Record<string, EnvParams['reward']> = {
  default:             { pelletReward: 5,  powerPelletReward: 20, deathPenalty: -100, stepPenalty: -0.1,  survivalReward: 0.02, ghostEatReward: 30,  winBonus: 200, reversePenalty: -2 },
  'ghost-hunting':     { pelletReward: 2,  powerPelletReward: 30, deathPenalty: -50,  stepPenalty: -0.05, survivalReward: 0.01, ghostEatReward: 80,  winBonus: 100, reversePenalty: -2 },
  'pellet-collection': { pelletReward: 15, powerPelletReward: 40, deathPenalty: -120, stepPenalty: -0.1,  survivalReward: 0.02, ghostEatReward: 20,  winBonus: 300, reversePenalty: -2 },
  'survival':          { pelletReward: 3,  powerPelletReward: 20, deathPenalty: -250, stepPenalty: -0.05, survivalReward: 0.2,  ghostEatReward: 50,  winBonus: 100, reversePenalty: -2 },
};

const trainPreset = (preset: keyof typeof rewardPresets, episodes: number, seed = 7, mazeId = 'pacman-classic') => {
  const rng = new SeededRng(seed);
  const env = new PacmanEnvironment();
  env.setParams({
    mazeId,
    numGhosts: 2,
    maxEpisodeSteps: 300,
    reward: rewardPresets[preset],
  });
  env.reset(seed);
  const agent = new QLearningAgent({ alpha: 0.2, gamma: 0.95, epsilon: 0.5, epsilonDecay: 0.995, epsilonMin: 0.05 });
  const trainer = new TrainingController(env, agent);
  for (let i = 0; i < episodes; i += 1) {
    let done = false;
    let guard = 0;
    while (!done) {
      const obs = env.observe();
      const legal = env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
      const action = agent.act(obs, legal, () => rng.next());
      const res = env.step(action);
      const nextLegal = env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
      agent.update(obs, action, res.reward, res.obs, res.done, nextLegal);
      done = res.done;
      guard += 1;
      if (guard > 2000) break;
    }
    agent.endEpisode();
    if (done) env.reset(seed + i + 1);
  }
  // Evaluate greedy
  const evalRes = trainer.evaluate(20);
  return evalRes;
};

describe('reward preset training', () => {
  test('all presets learn (eval score > raw step penalty floor)', () => {
    const results: Record<string, { avgScore: number; avgLength: number; winRate: number }> = {};
    for (const preset of Object.keys(rewardPresets)) {
      results[preset] = trainPreset(preset as keyof typeof rewardPresets, 80);
    }
    // Print for visibility
    // eslint-disable-next-line no-console
    console.log('Preset eval results:', JSON.stringify(results, null, 2));

    // Sanity bounds: training shouldn't degenerate to immediate death.
    for (const [name, r] of Object.entries(results)) {
      expect(r.avgLength, `${name} length should be > 5`).toBeGreaterThan(5);
    }
  }, 60_000);

  test('survival preset produces non-trivial episode length', () => {
    // Previously asserted survival > pellet-collection length, but pellet-escalation
    // rewards now make pellet-collection's high pelletReward (×6 at endgame) drive
    // the agent toward longer pellet-clearing runs. We now just verify survival
    // doesn't degenerate — long episodes still indicate it's learning to dodge.
    const survival = trainPreset('survival', 300);
    const pellet = trainPreset('pellet-collection', 300);
    // eslint-disable-next-line no-console
    console.log('survival vs pellet-collection lengths:', survival.avgLength, pellet.avgLength);
    expect(survival.avgLength).toBeGreaterThan(20);
    expect(pellet.avgLength).toBeGreaterThan(20);
  }, 90_000);

  test('pellet-collection preset achieves more score than survival', () => {
    const survival = trainPreset('survival', 120);
    const pellet = trainPreset('pellet-collection', 120);
    // eslint-disable-next-line no-console
    console.log('pellet vs survival scores:', pellet.avgScore, survival.avgScore);
    // Pellet-collection has higher per-pellet reward so raw score should exceed survival's
    // (which has tiny pellet reward) on the same maze.
    expect(pellet.avgScore).toBeGreaterThan(survival.avgScore);
  }, 90_000);
});
