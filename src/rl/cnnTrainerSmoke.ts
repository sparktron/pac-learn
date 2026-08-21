/**
 * T6 end-to-end trainer smoke — the narrowest run that answers "is the portable
 * path fast enough to attempt the 2k-episode gate?"
 *
 * The corrected 2026-07-30 portable benchmark measured *updates* (8.3/sec at
 * batch 64) and concluded WebGL was not categorically blocked. But with
 * `trainEvery=128` an update happens once per 128 environment steps, while
 * action selection happens on every one. A 2k-episode curve is roughly 700k
 * environment steps and only ~5.5k updates: at 8.3 updates/sec the update cost
 * is about eleven minutes, so it was never the deciding term. Inference is the
 * 128x-more-frequent operation and has never been measured.
 *
 * This runs the real loop — encode, select, step, observe, periodically learn —
 * and attributes wall time to encoding, inference, environment, replay, and
 * batch updates, so the projection to the full gate is arithmetic rather than
 * guesswork. The first real update is wrapped in `tf.profile()` to expose its
 * kernels and scalar-loss readback. It also splits action
 * selection into forward and readback, because `act()` ends in a GPU→CPU
 * transfer and the remedy differs completely depending on which term dominates.
 *
 * Development-only: reached via the query-gated panel, never bundled into a
 * production build.
 */
import { PacmanEnvironment } from '../env/environment';
import { SeededRng } from '../engine/prng';
import {
  CNN_DQN_DEFAULTS,
  CnnDqnAgent,
  encodeCnnState,
  type CnnDqnHyperParams,
  type CnnTrainProfile,
} from './cnnDqn';
import { initializeTensorRuntime, tf, type TensorBackend } from './tfRuntime';

/** Steps and updates a full 2k-episode gate would need. See module header. */
export interface GateProjection {
  /**
   * Environment steps in the modelled curve, from this run's own mean episode
   * length. An untrained policy dies early (~60 steps) where the promoted
   * linear agent averages ~355, so on a short smoke this UNDERSTATES a real
   * curve — by roughly 5x. The projection is therefore a floor, not a forecast.
   */
  steps: number;
  /** Updates implied by `trainEvery` over those steps. */
  updates: number;
  encodingHours: number;
  envHours: number;
  inferenceHours: number;
  replayHours: number;
  updateHours: number;
  totalHours: number;
  /**
   * Hours the curve would take if action selection never read back — i.e. the
   * measured forward+upload cost only. Derived from the directly-measured
   * forward time, NOT by subtracting readback from the in-loop act average:
   * those two are sampled differently (in-loop mean vs post-loop probes) and
   * the subtraction can go negative, which silently clamps to zero and reports
   * a fictitious "free" curve.
   *
   * Still a lower bound: on a deferred backend the forward kernels may not
   * have executed when predict() returns, so some of that work is being paid
   * for inside the readback. Treat this as "the best case on-device selection
   * could approach", not a promise.
   */
  totalHoursWithoutReadback: number;
}

export interface CnnTrainerSmokeResult {
  backend: string;
  episodes: number;
  steps: number;
  updates: number;
  batchSize: number;
  trainEvery: number;
  /** Mean per-operation wall time, milliseconds. */
  encodingMs: number;
  envStepMs: number;
  actMs: number;
  actForwardMs: number;
  actReadbackMs: number;
  replayMs: number;
  updateMs: number;
  /** Share of total measured wall time, 0-1. */
  encodingShare: number;
  envShare: number;
  actShare: number;
  replayShare: number;
  updateShare: number;
  /** The first in-loop batch update, including kernel and scalar-readback timing. */
  updateProfile: CnnTrainProfile | null;
  wallClockSec: number;
  stepsPerSec: number;
  tensorMemory: ReturnType<typeof tf.memory>;
  projection: GateProjection;
}

export interface CnnTrainerSmokeOptions {
  backend?: TensorBackend;
  /** Fixed work budget so a smoke cannot turn into several 1,000-step episodes. */
  totalSteps?: number;
  maxSteps?: number;
  trainEvery?: number;
  warmupTransitions?: number;
  seed?: number;
  hyper?: Partial<CnnDqnHyperParams>;
  /** Curve the projection models. Defaults to the roadmap's first gate. */
  gateEpisodes?: number;
  /** Action-selection passes to profile for the forward/readback split. */
  profileSamples?: number;
  onProgress?: (step: number, totalSteps: number) => void;
}

const SECONDS_PER_HOUR = 3_600;

/** Warm the greedy inference path regardless of the agent's exploration rate. */
export const warmCnnInference = async (
  agent: Pick<CnnDqnAgent, 'profileAct'>,
  state: ReturnType<typeof encodeCnnState>,
): Promise<void> => {
  await agent.profileAct(state);
};

export const runCnnTrainerSmoke = async (
  options: CnnTrainerSmokeOptions = {},
): Promise<CnnTrainerSmokeResult> => {
  const totalSteps = options.totalSteps ?? 256;
  const maxSteps = options.maxSteps ?? 1_000;
  const trainEvery = options.trainEvery ?? 128;
  // Deliberately below the bench's 256: a short smoke collects only a few
  // hundred transitions, so the production warmup would gate out every update
  // and the smoke would report an update cost of zero for a phase it never ran.
  const warmupTransitions = options.warmupTransitions ?? 64;
  const seed = options.seed ?? 7;
  const gateEpisodes = options.gateEpisodes ?? 2_000;
  const profileSamples = options.profileSamples ?? 20;
  if (!Number.isInteger(totalSteps) || totalSteps < 1) throw new Error('totalSteps must be a positive integer');
  if (!Number.isInteger(maxSteps) || maxSteps < 1) throw new Error('maxSteps must be a positive integer');
  if (!Number.isInteger(trainEvery) || trainEvery < 1) throw new Error('trainEvery must be a positive integer');
  if (!Number.isInteger(warmupTransitions) || warmupTransitions < 1) {
    throw new Error('warmupTransitions must be a positive integer');
  }
  if (!Number.isInteger(gateEpisodes) || gateEpisodes < 1) {
    throw new Error('gateEpisodes must be a positive integer');
  }
  if (!Number.isInteger(profileSamples) || profileSamples < 1) {
    throw new Error('profileSamples must be a positive integer');
  }
  const batchSize = options.hyper?.batchSize ?? CNN_DQN_DEFAULTS.batchSize;
  const lastScheduledUpdate = Math.floor(totalSteps / trainEvery) * trainEvery;
  if (lastScheduledUpdate < Math.max(warmupTransitions, batchSize)) {
    throw new Error('totalSteps must allow at least one replay-warmed batch update');
  }

  const backend = await initializeTensorRuntime(options.backend);
  const agent = new CnnDqnAgent({ seed, ...options.hyper });
  const env = new PacmanEnvironment();
  env.setParams({ mazeId: 'pacman-classic', numGhosts: 2, maxEpisodeSteps: maxSteps });
  const rng = new SeededRng(seed);

  let steps = 0;
  let completedEpisodes = 0;
  let completedEpisodeSteps = 0;
  let currentEpisodeSteps = 0;
  let updates = 0;
  let encodingMs = 0;
  let envMs = 0;
  let actMs = 0;
  let replayMs = 0;
  let updateMs = 0;
  let updateProfile: CnnTrainProfile | null = null;

  try {
    // Warm inference once so its first shader compilation and tensor upload do
    // not swamp the per-step measurement. The first training update remains
    // intentionally cold and profiled inside the timed loop.
    env.reset(seed);
    await warmCnnInference(agent, encodeCnnState(env));

    env.reset(rng.int(1_000_000));
    const startedAt = performance.now();
    while (steps < totalSteps) {
      const encodeStartedAt = performance.now();
      const state = encodeCnnState(env);
      encodingMs += performance.now() - encodeStartedAt;

      const actStartedAt = performance.now();
      const action = await agent.act(state, env.getLegalActionIndices(), () => rng.next());
      actMs += performance.now() - actStartedAt;

      const envStartedAt = performance.now();
      const result = env.step(action);
      const nextLegal = result.done ? [] : env.getLegalActionIndices();
      envMs += performance.now() - envStartedAt;

      const nextEncodeStartedAt = performance.now();
      const nextState = encodeCnnState(env);
      encodingMs += performance.now() - nextEncodeStartedAt;

      const replayStartedAt = performance.now();
      agent.observe({
        state,
        action,
        reward: result.reward,
        nextState,
        done: result.done,
        nextLegalActions: nextLegal,
      });
      replayMs += performance.now() - replayStartedAt;
      steps += 1;
      currentEpisodeSteps += 1;

      if (
        steps % trainEvery === 0
        && agent.replay.size >= Math.max(warmupTransitions, agent.hyper.batchSize)
      ) {
        const updateStartedAt = performance.now();
        let loss: number | null;
        if (updateProfile === null) {
          updateProfile = await agent.profileTrainBatch(
            agent.replay.sample(agent.hyper.batchSize, () => rng.next()),
          );
          loss = updateProfile.loss;
        } else {
          loss = await agent.learn(() => rng.next());
        }
        updateMs += performance.now() - updateStartedAt;
        if (loss !== null) updates += 1;
      }
      if (result.done) {
        completedEpisodes += 1;
        completedEpisodeSteps += currentEpisodeSteps;
        currentEpisodeSteps = 0;
        agent.endEpisode();
        if (steps < totalSteps) env.reset(rng.int(1_000_000));
      }
      if (steps % 32 === 0 || steps === totalSteps) options.onProgress?.(steps, totalSteps);
    }
    const wallClockSec = (performance.now() - startedAt) / 1_000;

    // Forward/readback split, sampled after the loop so the model is warm.
    let forwardTotal = 0;
    let readbackTotal = 0;
    const profileState = encodeCnnState(env);
    for (let sample = 0; sample < profileSamples; sample += 1) {
      const { forwardMs, readbackMs } = await agent.profileAct(profileState);
      forwardTotal += forwardMs;
      readbackTotal += readbackMs;
    }

    const perEncoding = encodingMs / Math.max(steps, 1);
    const perEnvStep = envMs / Math.max(steps, 1);
    const perAct = actMs / Math.max(steps, 1);
    const perReplay = replayMs / Math.max(steps, 1);
    const perUpdate = updateMs / Math.max(updates, 1);
    const perForward = forwardTotal / profileSamples;
    const perReadback = readbackTotal / profileSamples;
    const measuredMs = encodingMs + envMs + actMs + replayMs + updateMs;

    // Project the gate from this run's own average episode length rather than
    // a remembered constant, so a policy that dies early is not mistaken for a
    // cheap curve.
    const stepsPerEpisode = completedEpisodes > 0
      ? completedEpisodeSteps / completedEpisodes
      : steps;
    const gateSteps = Math.round(stepsPerEpisode * gateEpisodes);
    const gateUpdates = Math.floor(gateSteps / trainEvery);
    const hours = (ms: number): number => ms / 1_000 / SECONDS_PER_HOUR;
    const encodingHours = hours(perEncoding * gateSteps);
    const envHours = hours(perEnvStep * gateSteps);
    const inferenceHours = hours(perAct * gateSteps);
    const replayHours = hours(perReplay * gateSteps);
    const updateHours = hours(perUpdate * gateUpdates);

    return {
      backend,
      episodes: completedEpisodes,
      steps,
      updates,
      batchSize: agent.hyper.batchSize,
      trainEvery,
      encodingMs: perEncoding,
      envStepMs: perEnvStep,
      actMs: perAct,
      actForwardMs: perForward,
      actReadbackMs: perReadback,
      replayMs: perReplay,
      updateMs: perUpdate,
      encodingShare: encodingMs / Math.max(measuredMs, 1),
      envShare: envMs / Math.max(measuredMs, 1),
      actShare: actMs / Math.max(measuredMs, 1),
      replayShare: replayMs / Math.max(measuredMs, 1),
      updateShare: updateMs / Math.max(measuredMs, 1),
      updateProfile,
      wallClockSec,
      stepsPerSec: steps / Math.max(wallClockSec, 0.001),
      tensorMemory: tf.memory(),
      projection: {
        steps: gateSteps,
        updates: gateUpdates,
        encodingHours,
        envHours,
        inferenceHours,
        replayHours,
        updateHours,
        totalHours: encodingHours + envHours + inferenceHours + replayHours + updateHours,
        totalHoursWithoutReadback:
          encodingHours + envHours + hours(perForward * gateSteps) + replayHours + updateHours,
      },
    };
  } finally {
    agent.dispose();
  }
};
