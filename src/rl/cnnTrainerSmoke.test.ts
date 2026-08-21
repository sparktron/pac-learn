import { describe, expect, it, vi } from 'vitest';
import { runCnnTrainerSmoke, warmCnnInference } from './cnnTrainerSmoke';

describe('runCnnTrainerSmoke', () => {
  it('rejects a work budget that cannot reach a replay-warmed batch update', async () => {
    await expect(runCnnTrainerSmoke({ totalSteps: 127 })).rejects.toThrow(
      'totalSteps must allow at least one replay-warmed batch update',
    );
  });

  it('warms inference directly instead of taking an epsilon-random action', async () => {
    const profileAct = vi.fn().mockResolvedValue({ forwardMs: 1, readbackMs: 2 });
    const state = { data: new Float32Array(1) };

    await warmCnnInference({ profileAct }, state);

    expect(profileAct).toHaveBeenCalledOnce();
    expect(profileAct).toHaveBeenCalledWith(state);
  });
});
