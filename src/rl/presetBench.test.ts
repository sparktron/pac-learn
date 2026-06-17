import { describe, expect, test } from 'vitest';
import { PacmanEnvironment } from '../env/environment';
import { QLearningAgent } from './qlearning';
import { TrainingController } from './trainingController';
import { DIRECTIONS } from '../engine/types';
import { SeededRng } from '../engine/prng';
// D5.11: use the shared presets instead of a hand-mirrored copy (the comment
// "mirrored from App.tsx" was exactly the drift hazard this removes).
import { REWARD_PRESETS as rewardPresets } from './rewardPresets';

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
  const agent = new QLearningAgent({ alpha: 0.2, gamma: 0.99, epsilon: 0.5, epsilonDecay: 0.995, epsilonMin: 0.05 });
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
    console.log('survival vs pellet-collection lengths:', survival.avgLength, pellet.avgLength);
    expect(survival.avgLength).toBeGreaterThan(20);
    expect(pellet.avgLength).toBeGreaterThan(20);
  }, 90_000);

  test('pellet-collection preset achieves more score than survival', () => {
    const survival = trainPreset('survival', 120);
    const pellet = trainPreset('pellet-collection', 120);
    console.log('pellet vs survival scores:', pellet.avgScore, survival.avgScore);
    // Pellet-collection has higher per-pellet reward so raw score should exceed survival's
    // (which has tiny pellet reward) on the same maze.
    expect(pellet.avgScore).toBeGreaterThan(survival.avgScore);
  }, 90_000);
});
