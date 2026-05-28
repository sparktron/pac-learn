# Pac-Man Learning System Assessment (2026-05-27)

## Executive summary

The current system does **not** need a full rewrite. It needs a focused set of **algorithmic upgrades and measurement fixes**. The environment and training loop are already deterministic, test-covered, and structurally sound enough to iterate on.

The biggest limiter is not one catastrophic bug; it is the combination of:
1) a very compressed tabular state representation,
2) single-step TD backups with sparse delayed credit,
3) training/evaluation metrics that can mask policy quality, and
4) no mechanism for policy generalization across similar states.

## What looks healthy

- Deterministic seeded RNG and deterministic eval path are implemented.
- `load()` guards against incompatible key versions and ghost-count mismatch.
- Observation key versioning exists and is explicit.
- Core environment/observation/Q-learning tests are passing.

## Fundamental issues to confront

### 1) Tabular Q-learning state aliasing is likely dominating learning failure

The observation encoder intentionally compresses full board state into a coarse key. That keeps Q-table size manageable, but it aliases distinct tactical situations into the same bucket.

Consequence: updates from incompatible situations overwrite each other, flattening policy quality in high-risk endgame states.

### 2) Credit assignment is still weak for long-horizon planning

The agent uses one-step Q-learning with shaped rewards. Even with win bonus and pellet escalation, successful trajectories are long and stochastic.

Consequence: the gradient that should teach "survive now to win later" is noisy and diluted.

### 3) Exploration is global-state-light and may under-cover critical tails

Exploration is mostly epsilon-greedy with optional endgame epsilon floor. This is better than static epsilon, but still coarse.

Consequence: rare but decisive states (final pellets under pressure) are under-sampled relative to their importance.

### 4) Current benchmark signals can overstate "learning"

Several tests validate that scores improve over a baseline floor, but win rate remains low in many regimes.

Consequence: training can look healthy while core objective (reliable clears) remains unsolved.

## Rewrite decision

**Recommendation: no full rewrite now.**

Do a staged upgrade path:
- Preserve environment + renderer + controller architecture.
- Replace only the learner stack and offline evaluation protocol.

A full rewrite would increase risk and lose debugging leverage while not directly solving the representation/credit-assignment problem.

## Two viable paths

### Option A (recommended): Incremental upgrade to function approximation

- Keep current environment API.
- Add a feature-vector encoder and linear Q approximator (or tiny MLP) first.
- Add n-step returns / eligibility traces.
- Keep deterministic eval harness and compare against tabular baseline.

**Pros:** small diffs, fast iteration, low migration risk, easy A/B.
**Cons:** may still plateau before deep RL methods.

### Option B: Move directly to DQN-style agent

- Replay buffer, target network, mini-batch updates.
- Either vector features or local grid observations.
- Offline checkpoint eval at fixed seeds.

**Pros:** better generalization and long-horizon learning potential.
**Cons:** larger complexity jump; more tuning and infra needed.

## Comparable systems worth copying from

Use these for architecture patterns, not blind cloning:

1. **Gymnasium + Stable-Baselines3 Atari baselines**
   - Proven training loops (replay, target nets, evaluation callbacks).
   - Good reference for deterministic eval and checkpoint selection.

2. **CleanRL single-file DQN implementations**
   - Minimal, readable implementations with reproducibility focus.
   - Great for introducing DQN without framework sprawl.

3. **Dopamine (Google)**
   - Research-grade, compact RL baselines emphasizing reproducibility and ablations.

## Near-term action plan (no rewrite)

1. Add a **fixed-seed eval suite** with confidence intervals (e.g., 200 episodes × fixed seed sets).
2. Add **state-visitation diagnostics** for endgame buckets and death-near-win transitions.
3. Introduce **n-step Q-learning** (or TD(λ)) in current tabular setup to improve credit assignment.
4. If gains are limited, migrate to **linear/MLP Q approximator** while keeping current env API.
5. Only then decide whether a full DQN transition is justified.

## Bottom line

- The system is salvageable.
- The core issue is representational/algorithmic, not app architecture.
- Start with incremental learner upgrades and better eval rigor before considering a full rewrite.
