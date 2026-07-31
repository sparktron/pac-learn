import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import type { PacmanEnvironment } from '../env/environment';
import type { QLearningAgent } from '../rl/qlearning';
import type { LinearQLearningAgent } from '../rl/linearQlearning';
import type { TrainingController } from '../rl/trainingController';

// Training-speed presets: stepsPerFrame × renderEveryN × frame pacing. Live here
// (not App) since they're the loop's concern; `slow/normal` pace by wall-clock
// interval, `fast/turbo/max` run flat-out and cap render frequency / frame time.
export const trainingSpeedPresets = {
  slow:   { stepsPerFrame: 1,         renderEveryNSteps: 1,    frameIntervalMs: 240, maxFrameMs: 0 },
  normal: { stepsPerFrame: 1,         renderEveryNSteps: 1,    frameIntervalMs: 120, maxFrameMs: 0 },
  fast:   { stepsPerFrame: 20,        renderEveryNSteps: 5,    frameIntervalMs: 0,   maxFrameMs: 0 },
  turbo:  { stepsPerFrame: 1000,      renderEveryNSteps: 50,   frameIntervalMs: 0,   maxFrameMs: 0 },
  max:    { stepsPerFrame: 1_000_000, renderEveryNSteps: 1000, frameIntervalMs: 0,   maxFrameMs: 12 },
} as const;
export type TrainingSpeed = keyof typeof trainingSpeedPresets;
export const trainingSpeedOptions = Object.keys(trainingSpeedPresets) as TrainingSpeed[];

export interface TrainingLoop {
  isTraining: boolean;
  /** Start/resume the rAF training loop. reseed=false preserves the trainer's
   *  RNG stream (used when auto-resuming across a structural change). */
  startTraining: (reseed?: boolean) => void;
  stopTraining: () => void;
  /** Stop the loop, mark idle, and clear trainer stats + the stats cursor.
   *  Shared by the Reset / Reset-Q / algorithm-switch / ghost-count-reset flows. */
  haltAndResetStats: () => void;
  trainingSpeed: TrainingSpeed;
  updateTrainingSpeed: (speed: TrainingSpeed) => void;
  stepsPerFrame: number;
  setStepsPerFrame: Dispatch<SetStateAction<number>>;
  renderEveryNSteps: number;
  setRenderEveryNSteps: Dispatch<SetStateAction<number>>;
}

export interface UseTrainingLoopArgs {
  env: PacmanEnvironment;
  agent: QLearningAgent | LinearQLearningAgent;
  trainer: TrainingController;
  seed: number;
  numGhosts: number;
  mazeId: string;
  /** Bump the host's render tick. Must be stable (e.g. a useCallback) so the
   *  structural-reset effect's dependency list doesn't churn every render. */
  requestRender: () => void;
}

/**
 * Owns the training loop: isTraining state, the speed presets/refs, start/stop,
 * the Space-bar toggle, and the structural-reset effect (a maze/ghost-count/seed
 * change pauses training, resets the env, and resumes — A5 slice 3).
 *
 * The effects read live values through refs so the long-lived rAF loop and the
 * Space handler never capture stale closures (D7.1). Mount this AFTER the env's
 * renderer/heatmap effects so the structural reset still runs before the
 * ghost-AI re-apply (env.reset rebuilds ghosts as 'classic'; the ghost-AI effect
 * then restores the selected type).
 */
export function useTrainingLoop({
  env, agent, trainer, seed, numGhosts, mazeId, requestRender,
}: UseTrainingLoopArgs): TrainingLoop {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSpeed, setTrainingSpeed] = useState<TrainingSpeed>('normal');
  const [stepsPerFrame, setStepsPerFrame]                     = useState<number>(trainingSpeedPresets.normal.stepsPerFrame);
  const [renderEveryNSteps, setRenderEveryNSteps]             = useState<number>(trainingSpeedPresets.normal.renderEveryNSteps);
  const [trainingFrameIntervalMs, setTrainingFrameIntervalMs] = useState<number>(trainingSpeedPresets.normal.frameIntervalMs);
  const [trainingMaxFrameMs, setTrainingMaxFrameMs]           = useState<number>(trainingSpeedPresets.normal.maxFrameMs);

  const lastStatsLengthRef         = useRef(0);
  const stepsPerFrameRef           = useRef(stepsPerFrame);
  const renderEveryNRef            = useRef(renderEveryNSteps);
  const trainingFrameIntervalMsRef = useRef(trainingFrameIntervalMs);
  const trainingMaxFrameMsRef      = useRef(trainingMaxFrameMs);
  const isTrainingRef              = useRef(isTraining);
  const startTrainingRef           = useRef<(reseed?: boolean) => void>();
  const stopTrainingRef            = useRef<() => void>();
  useEffect(() => {
    stepsPerFrameRef.current = stepsPerFrame;
    renderEveryNRef.current = renderEveryNSteps;
    trainingFrameIntervalMsRef.current = trainingFrameIntervalMs;
    trainingMaxFrameMsRef.current = trainingMaxFrameMs;
    isTrainingRef.current = isTraining;
  }, [stepsPerFrame, renderEveryNSteps, trainingFrameIntervalMs, trainingMaxFrameMs, isTraining]);

  const updateTrainingSpeed = (speed: TrainingSpeed): void => {
    const p = trainingSpeedPresets[speed];
    setTrainingSpeed(speed);
    setStepsPerFrame(p.stepsPerFrame);
    setRenderEveryNSteps(p.renderEveryNSteps);
    setTrainingFrameIntervalMs(p.frameIntervalMs);
    setTrainingMaxFrameMs(p.maxFrameMs);
  };

  const startTraining = (reseed = true): void => {
    trainer.stop();
    if (reseed) trainer.setSeed(seed);
    // N18: always sync the trainer's episodeSeed to the current seed so that
    // evaluate() restores the env correctly even before the first episode ends.
    trainer.setCurrentSeed(seed);
    // N7: pin the numGhosts the Q-table is trained against (idempotent on resume).
    agent.setTrainedNumGhosts(numGhosts);
    setIsTraining(true);
    lastStatsLengthRef.current = trainer.stats.episodeScores.length;
    trainer.start(
      () => stepsPerFrameRef.current,
      () => renderEveryNRef.current,
      () => {
        if (trainer.stats.episodeScores.length > lastStatsLengthRef.current) {
          lastStatsLengthRef.current = trainer.stats.episodeScores.length;
        }
        requestRender();
      },
      {
        getFrameIntervalMs: () => trainingFrameIntervalMsRef.current,
        getMaxFrameMs:      () => trainingMaxFrameMsRef.current,
      },
    );
  };

  const stopTraining = (): void => { trainer.stop(); setIsTraining(false); };

  const haltAndResetStats = (): void => {
    trainer.stop();
    setIsTraining(false);
    trainer.resetStats();
    lastStatsLengthRef.current = 0;
  };

  useEffect(() => {
    startTrainingRef.current = startTraining;
    stopTrainingRef.current = stopTraining;
  });

  // Structural reset: a mazeId / numGhosts / seed change needs a fresh env.
  // Training is paused across the reset so a Q-update can't bridge the boundary
  // (its obs is pre-reset and its nextObs is post-reset → garbage Q-values),
  // then resumed (reseed only when the seed itself changed).
  const lastSeedRef = useRef(seed);
  const lastStructuralRef = useRef(`${mazeId}|${numGhosts}`);
  useEffect(() => {
    const structural = `${mazeId}|${numGhosts}`;
    const seedChanged = lastSeedRef.current !== seed;
    if (lastStructuralRef.current === structural && !seedChanged) return;
    lastStructuralRef.current = structural;
    lastSeedRef.current = seed;
    const wasTraining = isTrainingRef.current;
    if (wasTraining) trainer.stop();
    env.reset(seed);
    requestRender();
    if (wasTraining) startTrainingRef.current?.(seedChanged);
  }, [env, trainer, mazeId, numGhosts, seed, requestRender]);

  // Space = toggle training. D7.1: go through the refs, not the first-render
  // closures, so pressing Space after changing seed uses the current value.
  useEffect(() => {
    const onSpace = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'BUTTON' || target.tagName === 'SELECT') return;
      e.preventDefault();
      if (isTrainingRef.current) stopTrainingRef.current?.();
      else startTrainingRef.current?.();
    };
    window.addEventListener('keydown', onSpace);
    return () => window.removeEventListener('keydown', onSpace);
  }, []);

  return {
    isTraining, startTraining, stopTraining, haltAndResetStats,
    trainingSpeed, updateTrainingSpeed,
    stepsPerFrame, setStepsPerFrame,
    renderEveryNSteps, setRenderEveryNSteps,
  };
}
