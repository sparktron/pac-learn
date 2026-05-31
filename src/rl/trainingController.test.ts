import { describe, expect, test, vi } from 'vitest';
import { PacmanEnvironment } from '../env/environment';
import { observationKey } from '../env/observation';
import { QLearningAgent } from './qlearning';
import { TrainingController } from './trainingController';

const build = (params: Partial<Parameters<PacmanEnvironment['setParams']>[0]> = {}) => {
  const env = new PacmanEnvironment();
  env.setParams({ numGhosts: 0, maxEpisodeSteps: 5, ...params });
  env.reset(42);
  const agent = new QLearningAgent({ alpha: 0.1, gamma: 0.9, epsilon: 0.3, epsilonDecay: 1, epsilonMin: 0 });
  const trainer = new TrainingController(env, agent);
  trainer.setSeed(1);
  return { env, agent, trainer };
};

describe('TrainingController', () => {
  // C4: evaluate() uses a dedicated, freshly-seeded RNG each call, so repeated
  // calls with no training in between are identical (no drift from consuming a
  // shared stream).
  test('evaluate() is deterministic and self-contained across repeated calls (C4)', () => {
    const { trainer } = build();
    const r1 = trainer.evaluate(4);
    const r2 = trainer.evaluate(4);
    expect(r2).toEqual(r1);
  });

  // N18: evaluate() restores the training env to the recorded current seed, so a
  // mid-training eval doesn't leave the env stranded on an eval episode.
  test('evaluate() restores the env to the current episode seed (N18)', () => {
    const { env, trainer } = build();
    env.reset(777);
    trainer.setCurrentSeed(777);
    const before = observationKey(env.observe());

    trainer.evaluate(3); // internally resets the env ~3× to eval seeds

    const after = observationKey(env.observe());
    expect(after).toBe(before); // env back at the seed-777 start state
  });

  // singleStep() must pass an empty nextLegalActions on a terminal step so the
  // agent bootstraps from 0 (not from a phantom next state).
  test('singleStep() passes [] as nextLegalActions on a terminal step', () => {
    const { env, agent, trainer } = build({ maxEpisodeSteps: 1 });
    env.reset(42);
    trainer.setCurrentSeed(42);
    const spy = vi.spyOn(agent, 'update');

    trainer.singleStep(); // step 1 hits maxEpisodeSteps=1 → done

    const lastCall = spy.mock.calls[spy.mock.calls.length - 1];
    expect(lastCall?.[4]).toBe(true);     // done
    expect(lastCall?.[5]).toEqual([]);    // nextLegalActions empty on terminal
  });

  // H11: replayRecording() returns cloned positions, so a consumer mutating the
  // replay output can't corrupt the stored recording for a later replay.
  test('replayRecording() returns clones that do not alias the recording (H11)', () => {
    const { trainer } = build({ maxEpisodeSteps: 4 });
    trainer.setRecording(true);
    trainer.runSteps(12); // a few short episodes → recordings saved
    expect(trainer.getRecordingCount()).toBeGreaterThan(0);

    const rec = trainer.getLatestRecording()!;
    const replay1 = trainer.replayRecording(rec);
    expect(replay1.positions.length).toBeGreaterThan(0);
    replay1.positions[0].pac.x = 9999; // mutate the replay output

    const replay2 = trainer.replayRecording(rec);
    expect(replay2.positions[0].pac.x).not.toBe(9999); // recording untouched
  });

  // resetStats() clears the accumulated episode arrays in place.
  test('resetStats() clears accumulated episode stats', () => {
    const { trainer } = build({ maxEpisodeSteps: 3 });
    trainer.runSteps(9);
    expect(trainer.stats.episodeScores.length).toBeGreaterThan(0);
    trainer.resetStats();
    expect(trainer.stats.episodeScores).toHaveLength(0);
    expect(trainer.stats.episodeLengths).toHaveLength(0);
    expect(trainer.stats.epsilons).toHaveLength(0);
  });
});
