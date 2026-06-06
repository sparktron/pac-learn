import { describe, expect, test } from 'vitest';
import { REWARD_PRESETS } from './rewardPresets';
import { createDefaultEnv } from '../env/environment';

describe('REWARD_PRESETS (D5.11)', () => {
  // N20: the UI/bench 'default' preset must exactly equal the env's
  // out-of-the-box reward config. If they drift, selecting "default" silently
  // trains against a different reward shape than the env default — the exact
  // bug N20 fixed reactively (winBonus was 200 in the UI vs 1000 in the env).
  // Now that the preset is shared, this test is the single guard for it.
  test("'default' matches the env default reward (N20)", () => {
    const envDefault = createDefaultEnv().params.reward;
    expect(REWARD_PRESETS.default).toEqual(envDefault);
  });

  // Every preset must carry the complete reward key set (a missing key used to
  // be silently back-filled by env.setParams, hiding the gap — see D9.5).
  test('every preset has the full reward key set', () => {
    const keys = Object.keys(createDefaultEnv().params.reward).sort();
    for (const [name, cfg] of Object.entries(REWARD_PRESETS)) {
      expect(Object.keys(cfg).sort(), name).toEqual(keys);
    }
  });
});
