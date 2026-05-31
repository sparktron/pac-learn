import { SeededRng } from '../engine/prng';
import { DIRECTIONS, type Vec2 } from '../engine/types';
import type { PacmanEnvironment } from '../env/environment';
import type { QLearningAgent } from './qlearning';
import type { LinearQLearningAgent } from './linearQlearning';

export interface TrainingStats {
  episodeScores: number[];
  episodeLengths: number[];
  epsilons: number[];
}

export interface EpisodeFrame {
  action: number;
  reward: number;
  pacPos: { x: number; y: number };
  ghostPositions: Array<{ x: number; y: number }>;
}

export interface EpisodeRecording {
  seed: number;
  mazeId: string;
  frames: EpisodeFrame[];
  finalScore: number;
  finalLength: number;
}

export class TrainingController {
  private running = false;
  private loopId = 0;
  private rng = new SeededRng(7);
  readonly stats: TrainingStats = { episodeScores: [], episodeLengths: [], epsilons: [] };
  private recordingEpisodes = false;
  private recordings: EpisodeRecording[] = [];
  private currentEpisodeFrames: EpisodeFrame[] = [];
  private episodeSeed = 0;

  constructor(private env: PacmanEnvironment, private agent: QLearningAgent | LinearQLearningAgent) {}

  setSeed(seed: number): void {
    this.rng = new SeededRng(seed);
  }

  /**
   * N18: record the seed the environment was just reset with so that
   * evaluate() can restore the in-flight episode state correctly even
   * before the first episode has completed (episodeSeed defaults to 0,
   * which would reset the env to the wrong seed after an early evaluate call).
   * Call this whenever env.reset(seed) is called outside the training loop.
   */
  setCurrentSeed(seed: number): void {
    this.episodeSeed = seed;
  }

  resetStats(): void {
    this.stats.episodeScores.length = 0;
    this.stats.episodeLengths.length = 0;
    this.stats.epsilons.length = 0;
  }

  singleStep(): void {
    const obs = this.env.observe();
    const legal = this.env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
    const action = this.agent.act(obs, legal, () => this.rng.next());
    const res = this.env.step(action);
    const nextLegal = res.done ? [] : this.env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
    this.agent.update(obs, action, res.reward, res.obs, res.done, nextLegal);

    // Record frame if recording
    if (this.recordingEpisodes) {
      this.currentEpisodeFrames.push({
        action,
        reward: res.reward,
        pacPos: { ...this.env.getPacmen()[0].pos },
        ghostPositions: this.env.ghosts.map((g) => ({ ...g.pos })),
      });
    }

    if (res.done) {
      // D5.5: dropped the per-episode console.log. It fired on every episode
      // completion — in headless max-speed training that's thousands of lines/sec
      // of I/O in the hot loop. Episode results are already in this.stats; the
      // bench does its own reporting.
      this.agent.endEpisode();
      this.stats.episodeScores.push(res.info.score);
      this.stats.episodeLengths.push(res.info.step);
      this.stats.epsilons.push(this.agent.hyper.epsilon);

      // Save recording if enabled
      if (this.recordingEpisodes && this.currentEpisodeFrames.length > 0) {
        this.recordings.push({
          seed: this.episodeSeed,
          mazeId: this.env.params.mazeId,
          frames: this.currentEpisodeFrames,
          finalScore: res.info.score,
          finalLength: res.info.step,
        });
      }
      this.currentEpisodeFrames = [];

      this.episodeSeed = this.rng.int(1_000_000);
      this.env.reset(this.episodeSeed);
    }
  }

  runSteps(steps: number): void {
    for (let i = 0; i < steps; i += 1) this.singleStep();
  }

  start(
    getStepsPerFrame: () => number,
    renderEveryNSteps: () => number,
    onFrame: () => void,
    options: { getFrameIntervalMs?: () => number; getMaxFrameMs?: () => number } = {},
  ): void {
    this.running = true;
    const myId = ++this.loopId;
    let lastRunAt = 0;
    const loop = (now: number) => {
      if (!this.running || this.loopId !== myId) return;
      const frameIntervalMs = Math.max(0, options.getFrameIntervalMs?.() ?? 0);
      if (now - lastRunAt >= frameIntervalMs) {
        lastRunAt = now;
        const steps = Math.max(0, Math.floor(getStepsPerFrame()));
        const maxFrameMs = Math.max(0, options.getMaxFrameMs?.() ?? 0);
        const startedAt = performance.now();
        let rendered = false;
        let completedSteps = 0;
        for (let i = 0; i < steps; i += 1) {
          // Break promptly when stop() is called mid-frame. Without this,
          // max-speed mode could execute up to stepsPerFrame (1M) more
          // updates after the user clicked Stop before the outer rAF
          // tick caught up.
          if (!this.running || this.loopId !== myId) break;
          this.singleStep();
          completedSteps = i + 1;
          if (completedSteps % Math.max(1, renderEveryNSteps()) === 0) {
            rendered = true;
            onFrame();
          }
          if (maxFrameMs > 0 && performance.now() - startedAt >= maxFrameMs) break;
        }
        if (completedSteps > 0 && !rendered) onFrame();
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
  }

  evaluate(episodes: number, evalSeed = 0xE0A1): { avgScore: number; avgLength: number; winRate: number } {
    // Use a dedicated RNG and a fresh instance each call so eval is fully
    // deterministic and does NOT consume from the training stream — otherwise
    // two training runs that differ only in how often the user clicked
    // "evaluate" diverge after the first call.
    const evalRng = new SeededRng(evalSeed);
    const old = this.agent.hyper.epsilon;
    const oldEndgameEps = this.agent.hyper.endgameEpsilon;
    this.agent.hyper.epsilon = 0;
    this.agent.hyper.endgameEpsilon = 0; // suppress endgame ε floor during eval
    let score = 0;
    let len = 0;
    let wins = 0;
    for (let i = 0; i < episodes; i += 1) {
      this.env.reset(i + 1000);
      let done = false;
      while (!done) {
        const obs = this.env.observe();
        const legal = this.env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
        const action = this.agent.act(obs, legal, () => evalRng.next());
        const res = this.env.step(action);
        done = res.done;
        if (done) {
          score += res.info.score;
          len += res.info.step;
          if (res.info.pelletsLeft === 0) wins += 1;
        }
      }
    }
    this.agent.hyper.epsilon = old;
    this.agent.hyper.endgameEpsilon = oldEndgameEps;
    // Restore the training env to the seed of the in-flight episode so the
    // next singleStep() doesn't bridge a hidden reset boundary, producing
    // garbage (obs, action, reward, nextObs) Q-updates.
    this.env.reset(this.episodeSeed);
    return { avgScore: score / episodes, avgLength: len / episodes, winRate: wins / episodes };
  }

  setRecording(enabled: boolean): void {
    this.recordingEpisodes = enabled;
    if (!enabled) this.currentEpisodeFrames = []; // Clear partial episode on disable
  }

  exportRecordings(): string {
    return JSON.stringify(this.recordings, null, 2);
  }

  clearRecordings(): void {
    this.recordings = [];
  }

  getRecordingCount(): number {
    return this.recordings.length;
  }

  getLatestRecording(): EpisodeRecording | null {
    return this.recordings.length > 0 ? this.recordings[this.recordings.length - 1] : null;
  }

  replayRecording(recording: EpisodeRecording): { positions: Array<{ pac: Vec2; ghosts: Vec2[] }> } {
    // Clone frame objects on push so consumers (e.g. the canvas renderer's
    // tunnel-wrap clamps) can't mutate the recording in place — that would
    // make a second replay of the same recording show different positions.
    const positions: Array<{ pac: Vec2; ghosts: Vec2[] }> = [];
    for (const frame of recording.frames) {
      positions.push({
        pac: { ...frame.pacPos },
        ghosts: frame.ghostPositions.map((g) => ({ ...g })),
      });
    }
    return { positions };
  }
}
