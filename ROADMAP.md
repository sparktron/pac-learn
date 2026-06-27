# Development Roadmap — Training Quality

The structural/refactor backlog (A1–A5, B1, B2) is **done**. The remaining
problem is the one that matters: **the agent doesn't learn to win.** The
2-ghost track peaks at ~2.5% greedy-eval win rate and `p5 ≈ 55` pellets left
(out of ~218), and has been **plateaued there since 2026-05-16** — the
curriculum knob is saturated and every later experiment (linear FA, α sweeps)
moved nothing. This roadmap is focused entirely on **making training better**.

Empirical history + baselines: **`test_history.md`** (read its *Findings* and
*Current State* first). Refactor-era history: `archive/DEEP_DIVE_2026-05-30.md`.

---

## Diagnosis (from the 2026-06-27 training-code review)

Two root causes, both visible in the code, explain the plateau. Every item
below targets one of them.

### Root cause A — the state representation aliases away the maze
`observationKey()` (`src/env/observation.ts:334`) packs only: local wall mask,
*direction to the nearest pellet within radius 12*, two nearest-ghost zone+heading
codes, last action, and two coarse pellet-count buckets. It encodes **neither
Pac-Man's position nor which pellets remain.** The agent is therefore a *reactive
controller* — it can chase the nearest pellet and dodge a local ghost, but it
cannot route around the maze or reason about where the remaining pellets are.
Thousands of genuinely different board states collapse to one key, so a single
Q-value must serve contradictory situations. **Tabular Q-learning cannot fix
aliasing** — this is the hard ceiling behind Open Question #4 ("may be a
fundamental limit of the state representation"). It is.

### Root cause B — sparse win signal + reward balance that makes the last pellets EV-negative
- **Credit assignment is 1-step.** `update()` (`qlearning.ts:134`) does vanilla
  one-step Q-learning — no n-step returns, no eligibility traces. The terminal
  `winBonus` propagates **one tile per re-visit**, so it has to be re-encountered
  ~290 times along a ~290-step winning path to reach the opening move. With wins
  at <1% this backup is astronomically slow.
- **The discount horizon is shorter than the maze.** γ=0.99 over a ~290-step
  optimal path gives γ²⁹⁰ ≈ 0.054, so `winBonus=1000` is worth only **~53** at
  the start — *less* than the cumulative pellet reward (~3800 undiscounted). The
  win barely influences opening decisions.
- **The last pellets are rationally refused.** A late pellet pays
  `pelletReward × escalation` ≤ 5×6 = **30**; grabbing it risks `deathPenalty =
  −100`. The escalation (max 6×, `environment.ts:433`) never overcomes the death
  risk, so a value-maximizing agent farms the easy ~75% and **stops** — exactly
  the `p5 ≈ 55` plateau we observe.
- **Optimism leaks into the greedy policy.** `bestNext` falls back to
  `optimisticInit` (50) for unseen next-states (`qlearning.ts:151`), inflating
  Q broadly toward 50; unvisited slots stay at 50; greedy `act()` then
  **tie-breaks randomly** among equal Q-values (`qlearning.ts:129`). In eval
  (ε=0) any state with unvisited/ties degrades to a random walk — which is why
  greedy eval underperforms ε-greedy training wins.

---

## Conventions for every item (read first)

1. **Branch off latest `master`, one focused PR per item.** `git fetch origin`
   first — this repo is worked in parallel.
2. **CI is the gate** (`.github/workflows/ci.yml`: typecheck + test + build +
   lint). Don't self-merge red.
3. **Protect the baseline.** Any change to the env, the observation key, the
   reward, or the seeded RNG stream can invalidate `test_history.md`. The safe
   pattern: **add new behavior behind a flag/param that defaults to today's
   behavior**, so "off" is byte-identical and CI only proves "nothing changed
   when off." If a change *must* alter the key, bump `OBSERVATION_KEY_VERSION`
   (`observation.ts`) — `load()` correctly discards mismatched policies.
4. **Every training item ends with a measured run logged to `test_history.md`**
   (config, `wins`, `p5`, verdict). A code change with no run is not done.
   Track wins via the raw `wins` count and `pl_p5`, never `winRate` (Finding #7).

---

## T. Training quality (ordered by leverage)

### T1 — Faster credit assignment: n-step returns / eligibility traces · ★ highest leverage, contained
**What:** replace the 1-step backup with **n-step returns** (or accumulating
eligibility traces, Q(λ)) so the terminal `winBonus` propagates along the whole
path in one episode instead of one tile per re-visit.
**Why:** this is the single change most likely to move the plateau — it attacks
Root cause B's slowest mechanism without touching the representation or the
reward. Sparse-terminal-reward tasks are exactly what n-step/λ returns are for.
**Where:** `src/rl/qlearning.ts` `update()` + the call site in
`trainingController.ts:singleStep()`. An n-step buffer of the last *n*
(obs, action, reward) tuples per episode is the smallest version; Q(λ) with a
trace map is the fuller one.
**Safety:** gate behind `nStep` (default 1) / `lambda` (default 0) hyperparams
→ off-state is byte-identical to today. New CLI knobs in `overnight-bench.ts`.
**Verify:** unit test that n=1 reproduces current updates exactly; then a
single-worker sweep `n ∈ {1,3,5,10}` (and/or `λ ∈ {0,0.5,0.9}`) vs the current
2-ghost baseline. Success = `p5 < 55` or `wins` up at equal compute.

### T2 — Discount + reward-balance sweep · cheap, no code (or tiny)
**What:** sweep the knobs that make the last pellets EV-negative —
`gamma ∈ {0.99, 0.997, 0.999}`, `winBonus ∈ {1000, 2500, 5000}`,
`deathPenalty ∈ {−100, −50}`, and a steeper pellet-escalation cap (6× → 10×).
**Why:** directly targets Root cause B's reward arithmetic. Open Question #1
flags `winBonus`/`deathPenalty` as "highest-leverage untested knob"; the γ
horizon mismatch (γ²⁹⁰≈0.05) is new from this review and just as cheap to test.
**Where:** mostly CLI sweeps (`hyperparam-sweep.sh`); raising the escalation cap
is a one-line change in `environment.ts:433` (gate behind a param if it alters
the default).
**Verify:** `hyperparam-sweep.sh` matrix; log the winning cell to history.
Cheapest item here — do it first to de-risk T3/T1 reward assumptions.

### T3 — Potential-based reward shaping for the endgame · principled densification
**What:** add a **potential-based** shaping term Φ(s) (e.g. Φ = −pelletsLeft, or
−distance-to-nearest-pellet) so `r' = r + γΦ(s') − Φ(s)`. This densifies the
"make progress toward clearing the maze" signal **without changing the optimal
policy** (Ng et al. potential-based shaping is policy-invariant — unlike the ad
hoc escalation, which distorts it).
**Why:** the current escalation is a non-potential hack that biases values; a
proper Φ gives a dense, unbiased gradient toward the win and complements T1.
**Where:** new shaping term in `environment.ts step()` reward assembly
(~`environment.ts:464`), behind a `shaping` flag (default off → baseline-safe).
**Verify:** prove invariance with a test (shaped vs unshaped greedy policy match
on a fixed toy rollout), then a sweep vs baseline.

### T4 — Decouple exploration-optimism from the greedy policy · fixes the eval gap
**What:** stop optimism from polluting evaluated values: (a) bootstrap unseen
next-states from **0, not `optimisticInit`** in the *target* (`qlearning.ts:151`)
while keeping optimistic init only for *action selection*; and/or (b) in eval,
break Q-ties **toward the most-visited action** (visit counts already exist,
`qlearning.ts:64`) or toward `nearestPelletDir`, instead of randomly.
**Why:** directly explains the train-wins-but-greedy-eval-0 gap (Root cause B,
last bullet). A cleaner greedy argmax may recover much of the existing policy's
latent skill for free.
**Safety:** both behind flags defaulting to current behavior; (a) changes the
learning target so it needs a key-independent A/B, not a byte-identical claim.
**Verify:** re-evaluate an *existing* trained policy from `bench-out/` with the
new eval tie-break (no retrain needed for (b)) — fastest possible signal.

### T5 — Less-aliased state: add a coarse Pac-Man region to the key · attacks Root cause A, contained
**What:** add a low-cardinality **Pac-Man maze-region** field (e.g. 3×3 = 9
zones, or quadrant=4) to the observation key so the agent can at least tell
*where in the maze* it is — cutting the worst of the aliasing without exploding
the state space.
**Why:** the contained, tabular-friendly half of Root cause A. Pure key growth:
19.5M × 9 ≈ 175M theoretical, but populated states stay far smaller.
**Where:** `observation.ts` (new field + `observationKey` term), **bump
`OBSERVATION_KEY_VERSION`** (old policies discarded — correct).
**Verify:** confirm key round-trips (`stringToObservationKey`), then a from-
scratch sweep vs baseline. Watch Q-table size / states-per-second for blowup.

### T6 — Deep Q-network over the raw grid (DQN/CNN) · the real ceiling-breaker, large
**What:** replace hand-features with a function approximator that ingests the
**full board** (pellet map + walls + ghost positions as grid planes) — a small
CNN DQN with replay + target network.
**Why:** the principled fix for Root cause A and the only path the evidence
supports past the ~2.5% ceiling. The linear-FA experiments (Finding #10) already
proved hand-features + linear models top out ~3× below tabular — capacity, not
tuning, is the wall. A board-seeing model is the next capacity tier.
**Caution:** largest item by far; real research effort (training stability,
replay, target nets, JS/WASM perf for the conv). Scope explicitly before
starting; likely its own multi-PR track. Keep tabular as the shipped baseline
throughout.
**Verify:** beat the 2-ghost tabular baseline (`avgScore ~960`, `p5 ~55`,
2.5% wins) on the same eval harness.

---

## I. Infra to make the above measurable

### I1 — Single-worker reproducible baseline + fast learning smoke
**What:** a deterministic single-worker run that reproduces a known win rate
(Open Questions #2/#3: "is the federated merge doing more than the policy
reflects?"), plus a CI-bounded short-learning smoke that asserts win-rate /`p5`
**doesn't regress** below a pinned floor.
**Why:** every T-item needs a trustworthy, fast A/B reference; right now the only
signal is multi-hour 32-worker runs. Also catches reward/key regressions in CI.
**Where:** extend `scripts/short-learning-sweep.sh`; add a small assertion
harness. Keep it under the CI time budget (bounded episodes, not minutes).

### I2 — Sweep ergonomics: expose `nStep`, `lambda`, `gamma`, shaping as first-class CLI knobs
**What:** thread the new T1/T2/T3 knobs through `overnight-bench.ts` and the
sweep scripts so experiments are config-only, no code edits per run.
**Why:** keeps the experiment loop fast and `test_history.md` honest (one knob
per cell). `lambda` is currently hard-coded 0 in the bench (Finding #10 note).

---

## Future / Maybe

Deprioritized — not on the critical path to better training.

### C1 — Maze editor + import/export (D2.6) · ⏸ parked
In-UI editor to draw/edit mazes and import/export them as JSON
(`validateMaze()` already exists in `src/mazes/mazes.ts` to reuse). A real
product/design effort (grid editing, palette, persistence; several PRs) with
**no bearing on training quality** — parked until the agent actually wins.

### Linear function-approximation agent · ⏸ parked (Finding #10)
Continuous features + α sweeps left it ~3× below tabular and never winning. Only
revisit *after* T6 makes the case for function approximation; if so, the move is
a richer model (T6's CNN), not more linear-feature tuning. Don't spend more time
on the linear path in isolation.

---

## Recommended order

**T2** (cheap reward/γ sweep — de-risks everything) → **T4** (eval fix, may be a
free win on existing policies) → **T1** (n-step/λ — highest-leverage code change)
→ **I1/I2** (lock in fast measurement) → **T3** (potential shaping) → **T5**
(coarse position key) → **T6** (DQN — the real ceiling-breaker, its own track).

---

## Quick-reference: file map

| Area | Path |
|------|------|
| Env + rewards + EnvParams + escalation | `src/env/environment.ts` |
| Observation / state key | `src/env/observation.ts` |
| Tabular agent (update, optimism, act) | `src/rl/qlearning.ts` |
| Linear agent (parked) | `src/rl/linearQlearning.ts` |
| Training loop + eval | `src/rl/trainingController.ts` |
| Reward presets | `src/rl/rewardPresets.ts` |
| Ghost AI | `src/ghosts/ghostAi.ts` |
| Bench / sweep / merge CLIs | `scripts/*` |
| Empirical log (READ FIRST) | `test_history.md` |
