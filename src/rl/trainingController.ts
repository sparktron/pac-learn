import { SeededRng } from '../engine/prng';
import { DIRECTIONS } from '../engine/types';
import type { PacmanEnvironment } from '../env/environment';
import type { QLearningAgent } from './qlearning';

export interface TrainingStats {
  episodeScores: number[];
  episodeLengths: number[];
  epsilons: number[];
}

export class TrainingController {
  private running = false;
  private rng = new SeededRng(7);
  readonly stats: TrainingStats = { episodeScores: [], episodeLengths: [], epsilons: [] };

  constructor(private env: PacmanEnvironment, private agent: QLearningAgent) {}

  setSeed(seed: number): void {
    this.rng = new SeededRng(seed);
  }

  singleStep(): void {
    const obs = this.env.observe();
    const legal = this.env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
    const action = this.agent.act(obs, legal, () => this.rng.next());
    const res = this.env.step(action);
    this.agent.update(obs, action, res.reward, res.obs, res.done);
    if (res.done) {
      this.agent.endEpisode();
      this.stats.episodeScores.push(res.info.score);
      this.stats.episodeLengths.push(res.info.step);
      this.stats.epsilons.push(this.agent.hyper.epsilon);
      this.env.reset(this.rng.int(1_000_000));
    }
  }

  runSteps(steps: number): void {
    for (let i = 0; i < steps; i += 1) this.singleStep();
  }

  start(getStepsPerFrame: () => number, renderEveryNSteps: () => number, onFrame: () => void): void {
    this.running = true;
    const loop = () => {
      if (!this.running) return;
      const steps = getStepsPerFrame();
      this.runSteps(steps);
      if (this.stats.episodeScores.length % Math.max(1, renderEveryNSteps()) === 0) onFrame();
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
}
