import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react';
import { createDefaultEnv, type EnvParams, type PacmanEnvironment } from '../env/environment';
import { REWARD_PRESETS } from '../rl/rewardPresets';

export interface GameEnv {
  env: PacmanEnvironment;
  params: EnvParams;
  setParams: Dispatch<SetStateAction<EnvParams>>;
  rewardPreset: string;
  setRewardPreset: Dispatch<SetStateAction<string>>;
}

/**
 * Owns the environment instance and the editable {@link EnvParams} (plus the
 * selected reward-preset name), and **live-applies** param edits to the env
 * without resetting it.
 *
 * N6: editing a reward field or a speed must NOT call `env.reset` — that would
 * kill the in-flight episode and wipe the trainer's progress while the user is
 * still typing. Structural resets (maze / ghost count / seed) and the
 * heatmap/ghost-AI syncs are deliberately left in `App` for now: they're
 * coupled to the training loop and the view mode, and move in later A5 slices.
 * Keeping this hook's effect first preserves the original effect ordering
 * (live-apply ran before the structural reset).
 */
export function useGameEnv(): GameEnv {
  const env = useMemo(() => createDefaultEnv(), []);
  // N16: structuredClone so params (and its nested reward object) shares no
  // reference with env.params or the REWARD_PRESETS entries — a direct reference
  // would be mutated by reward-field edits and silently corrupt the preset.
  const [params, setParams] = useState<EnvParams>(
    () => structuredClone({ ...env.params, reward: REWARD_PRESETS['default'] }),
  );
  const [rewardPreset, setRewardPreset] = useState<string>('default');

  // Live-apply on every params change; NO env.reset (see N6 above).
  useEffect(() => { env.setParams(params); }, [env, params]);

  return { env, params, setParams, rewardPreset, setRewardPreset };
}
