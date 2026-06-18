// @vitest-environment jsdom
import { afterEach, describe, expect, test } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';
import { useGameEnv } from './useGameEnv';

afterEach(cleanup);

describe('useGameEnv', () => {
  test('provides an env plus default params and reward preset', () => {
    const { result } = renderHook(() => useGameEnv());
    expect(result.current.env).toBeDefined();
    expect(result.current.rewardPreset).toBe('default');
    expect(result.current.params.reward.pelletReward).toBe(5); // default preset
  });

  test('live-applies param edits to the same env instance (no reset)', () => {
    const { result } = renderHook(() => useGameEnv());
    const envRef = result.current.env;
    act(() => { result.current.setParams((p) => ({ ...p, maxEpisodeSteps: 555 })); });
    expect(result.current.params.maxEpisodeSteps).toBe(555);
    expect(result.current.env).toBe(envRef);               // same instance — not rebuilt
    expect(result.current.env.params.maxEpisodeSteps).toBe(555); // applied via the effect
  });

  test('params.reward is a deep copy — edits do not corrupt the shared preset (N16)', () => {
    const { result } = renderHook(() => useGameEnv());
    act(() => { result.current.setParams((p) => ({ ...p, reward: { ...p.reward, pelletReward: 999 } })); });
    // A fresh mount still sees the pristine default preset, proving no shared ref.
    const { result: fresh } = renderHook(() => useGameEnv());
    expect(fresh.current.params.reward.pelletReward).toBe(5);
  });
});
