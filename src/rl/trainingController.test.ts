import { describe, expect, test, vi } from 'vitest';
import { TrainingController } from './trainingController';

const envStub = {
  observe: () => ({ pac: { x: 1, y: 1 }, ghosts: [], wallMask: 0, nearestPelletDir: 0, ghostRel: [] }),
  getLegalActions: () => ['up'],
  step: () => ({ obs: { pac: { x: 1, y: 1 }, ghosts: [], wallMask: 0, nearestPelletDir: 0, ghostRel: [] }, reward: 0, done: true, info: { score: 0, pelletsLeft: 0, step: 1 } }),
  reset: () => ({ pac: { x: 1, y: 1 }, ghosts: [], wallMask: 0, nearestPelletDir: 0, ghostRel: [] }),
};

const agentStub = {
  hyper: { epsilon: 0.5, epsilonDecay: 1, epsilonMin: 0, alpha: 0.1, gamma: 0.9 },
  act: () => 0,
  update: () => undefined,
  endEpisode: () => undefined,
};

describe('TrainingController', () => {
  test('evaluate validates episode count', () => {
    const trainer = new TrainingController(envStub as never, agentStub as never);
    expect(() => trainer.evaluate(0)).toThrow('episodes must be a positive integer');
    expect(() => trainer.evaluate(-2)).toThrow('episodes must be a positive integer');
  });

  test('start does not schedule duplicate loops when already running', () => {
    const trainer = new TrainingController(envStub as never, agentStub as never);
    const originalRaf = (globalThis as { requestAnimationFrame?: (cb: FrameRequestCallback) => number }).requestAnimationFrame;
    const rafMock = vi.fn(() => 1);
    (globalThis as { requestAnimationFrame?: (cb: FrameRequestCallback) => number }).requestAnimationFrame = rafMock;

    trainer.start(() => 1, () => 1, () => undefined);
    trainer.start(() => 1, () => 1, () => undefined);

    expect(rafMock).toHaveBeenCalledTimes(1);
    trainer.stop();

    if (originalRaf) {
      (globalThis as { requestAnimationFrame?: (cb: FrameRequestCallback) => number }).requestAnimationFrame = originalRaf;
    } else {
      delete (globalThis as { requestAnimationFrame?: (cb: FrameRequestCallback) => number }).requestAnimationFrame;
    }
  });
});
