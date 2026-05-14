import { SeededRng } from '../engine/prng';
import { DIRECTIONS, type Vec2 } from '../engine/types';
import type { PacmanEnvironment } from '../env/environment';
import type { QLearningAgent } from './qlearning';

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

  constructor(private env: PacmanEnvironment, private agent: QLearningAgent) {}

  setSeed(seed: number): void {
    this.rng = new SeededRng(seed);
  }

  singleStep(): void {
    const obs = this.env.observe();
    const legal = this.env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
    const action = this.agent.act(obs, legal, () => this.rng.next());
    const res = this.env.step(action);
    const nextLegal = this.env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
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
      console.log(`Episode done: step=${res.info.step} score=${res.info.score} pelletsLeft=${res.info.pelletsLeft}`);
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

  evaluate(episodes: number): { avgScore: number; avgLength: number; winRate: number } {
    const old = this.agent.hyper.epsilon;
    this.agent.hyper.epsilon = 0;
    let score = 0;
    let len = 0;
    let wins = 0;
    for (let i = 0; i < episodes; i += 1) {
      this.env.reset(i + 1000);
      let done = false;
      while (!done) {
        const obs = this.env.observe();
        const legal = this.env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
        const action = this.agent.act(obs, legal, () => this.rng.next());
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
    // Reconstruct episode positions for playback
    const positions = [];
    let pacPos = this.env.getPacmen()[0].pos;
    let ghostPositions = this.env.ghosts.map((g) => g.pos);

    for (const frame of recording.frames) {
      pacPos = frame.pacPos;
      ghostPositions = frame.ghostPositions;
      positions.push({ pac: pacPos, ghosts: ghostPositions });
    }

    return { positions };
  }
}
