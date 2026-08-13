import { describe, expect, it } from 'vitest';
import { runCnnTrainerSmoke } from './cnnTrainerSmoke';

describe('runCnnTrainerSmoke', () => {
  it('rejects a work budget that cannot reach a replay-warmed batch update', async () => {
    await expect(runCnnTrainerSmoke({ totalSteps: 127 })).rejects.toThrow(
      'totalSteps must allow at least one replay-warmed batch update',
    );
  });
});
