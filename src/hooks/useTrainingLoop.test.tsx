// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';
import { createDefaultEnv } from '../env/environment';
import { QLearningAgent } from '../rl/qlearning';
import { TrainingController } from '../rl/trainingController';
import { useTrainingLoop, trainingSpeedPresets, type UseTrainingLoopArgs } from './useTrainingLoop';

const baseHyper = { alpha: 0.1, gamma: 0.99, epsilon: 0.5, epsilonDecay: 0.999997, epsilonMin: 0.2 };

const setup = (over: Partial<UseTrainingLoopArgs> = {}) => {
  const env = createDefaultEnv();
  const agent = new QLearningAgent(baseHyper);
  const trainer = new TrainingController(env, agent);
  const requestRender = vi.fn();
  const args: UseTrainingLoopArgs = {
    env, agent, trainer, seed: 7, numGhosts: 2, mazeId: 'pacman-classic', requestRender, ...over,
  };
  return { env, agent, trainer, requestRender, args };
};

beforeEach(() => {
  // Stub rAF so trainer.start() doesn't spin a real loop in the test.
  vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('useTrainingLoop', () => {
  test('start/stop toggles isTraining', () => {
    const { args } = setup();
    const { result } = renderHook(() => useTrainingLoop(args));
    expect(result.current.isTraining).toBe(false);
    act(() => result.current.startTraining());
    expect(result.current.isTraining).toBe(true);
    act(() => result.current.stopTraining());
    expect(result.current.isTraining).toBe(false);
  });

  test('updateTrainingSpeed applies the preset values', () => {
    const { args } = setup();
    const { result } = renderHook(() => useTrainingLoop(args));
    act(() => result.current.updateTrainingSpeed('turbo'));
    expect(result.current.trainingSpeed).toBe('turbo');
    expect(result.current.stepsPerFrame).toBe(trainingSpeedPresets.turbo.stepsPerFrame);
    expect(result.current.renderEveryNSteps).toBe(trainingSpeedPresets.turbo.renderEveryNSteps);
  });

  test('startTraining pins trainedNumGhosts and starts the trainer', () => {
    const { agent, trainer, args } = setup({ numGhosts: 3 });
    const startSpy = vi.spyOn(trainer, 'start');
    const { result } = renderHook(() => useTrainingLoop(args));
    act(() => result.current.startTraining());
    expect(agent.trainedNumGhosts).toBe(3);
    expect(startSpy).toHaveBeenCalledOnce();
  });

  test('haltAndResetStats stops the loop and clears trainer stats', () => {
    const { trainer, args } = setup();
    const { result } = renderHook(() => useTrainingLoop(args));
    act(() => result.current.startTraining());
    trainer.stats.episodeScores.push(1, 2, 3);
    act(() => result.current.haltAndResetStats());
    expect(result.current.isTraining).toBe(false);
    expect(trainer.stats.episodeScores).toHaveLength(0);
  });

  test('a seed change resets the env and requests a render (structural reset)', () => {
    const { env, requestRender, args } = setup();
    const resetSpy = vi.spyOn(env, 'reset');
    const { rerender } = renderHook((p: UseTrainingLoopArgs) => useTrainingLoop(p), { initialProps: args });
    // Mount runs the effect once but early-returns (seed/structural unchanged).
    resetSpy.mockClear();
    requestRender.mockClear();
    rerender({ ...args, seed: 99 });
    expect(resetSpy).toHaveBeenCalledWith(99);
    expect(requestRender).toHaveBeenCalled();
  });
});
