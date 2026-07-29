# Development Roadmap — Training Quality

The structural/refactor backlog (A1–A5, B1, B2) is **done**. The historical
tabular 2-ghost track plateaued near 2.5% greedy-eval win rate and `p5 ≈ 55`.
D8's action-conditioned linear features and D9's target network broke through
that ceiling. T7's far-pellet direction then raised a matched five-seed,
four-panel mean from 25.54% to **35.17% linear eval wins** (seed means
33.72–36.79%). This roadmap is focused entirely on **making training better
and validating gains reproducibly**.

Empirical history + baselines: **`test_history.md`** (read its *Findings* and
*Current State* first). Refactor-era history: `archive/DEEP_DIVE_2026-05-30.md`.
The teachable reasoning trail—including failed experiments and superseded
conclusions—is maintained in **`ENGINEERING_JOURNAL.md`**.

---

## 2026-07-21 correctness follow-up

The full review is in `CODE_REVIEW_2026-07-21.md`. Its high-priority findings
are now resolved:

- GUI and headless training share algorithm-specific defaults.
- Policy files are validated before their encoded ghost count is applied.
- Speeds above one resolve pellets and collisions at every traversed tile.

The fixes include regression coverage and preserve the existing one-tile
cross-over capture semantics.

---

## Diagnosis (from the 2026-06-27 training-code review)

Two root causes, both visible in the code, explain the plateau. Every item
below targets one of them.

### Root cause A — the state representation aliases away the maze
`observationKey()` (`src/env/observation.ts:334`) packs only: local wall mask,
*direction to the nearest reachable pellet*, two nearest-ghost zone+heading
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

### T1 — Faster credit assignment: n-step returns · ✅ completed, no promotion
**Implementation:** both agents now buffer transitions behind a positive
integer `nStep` hyperparameter (default 1). The buffer emits the discounted
n-step target and flushes every short terminal suffix without bootstrapping.
`overnight-bench.ts` exposes `nStep`; `scripts/t1-nstep-sweep.sh` runs the
config-only screen. The existing linear `lambda` is L2 regularization, not an
eligibility-trace control, so no misleading Q(λ) knob was added.
**Result:** at the promoted T2/T7 configuration, seed 7, 2,000 episodes, and
four 50-game panels, n=1 achieved 36.0% mean greedy wins (20.0% worst panel),
n=3 31.5% (22.0%), n=10 29.5% (20.0%), and n=5 0.0% (0.0%; `pl_p5=223.3`).
No candidate exceeded the baseline mean or produced a better endgame tail, so
`nStep=1` remains the default and no five-seed confirmation was warranted.

### T2 — Discount + reward-balance sweep · ✅ completed
**What:** swept the knobs that make the last pellets EV-negative —
`gamma ∈ {0.99, 0.997, 0.999}`, `winBonus ∈ {1000, 2500, 5000}`,
`deathPenalty ∈ {−100, −50}`, and a steeper pellet-escalation cap (6× → 10×).
**Result:** the seed-7, 36-cell screen selected `gamma=0.997`,
`winBonus=1000`, `deathPenalty=-50`, and a 10× escalation cap (36.0% mean
versus 33.5% baseline). Matched five-seed/four-panel confirmation at 2,000
episodes improved mean greedy wins **33.25% → 37.17%**; every seed improved and
the minimum worst-panel mean rose **29.5% → 32.5%**. The winning linear gamma
and shared reward defaults are promoted in the GUI, bench, and environment.
**Implementation:** `pelletEscalationMax` is now an environment/reward-preset
parameter; `overnight-bench.ts` exposes all T2 knobs; and
`scripts/t2-reward-sweep.sh` makes the 36-cell screen config-only.

### T3 — Potential-based reward shaping · ✅ completed, no promotion
**Implementation:** the environment now optionally adds
`γΦ(s') − Φ(s)`, with `Φ(s) = -scale · pelletsLeft / totalPellets` and zero
potential on every terminal state. `potentialShapingScale=0` keeps the baseline
byte-for-byte unchanged; `potentialShapingGamma` must match learner γ when
enabled. The telescoping property is unit-tested, and the bench exposes
`shapingScale`/`shapingGamma` through `scripts/t3-potential-shaping-sweep.sh`.
**Result:** on the promoted linear/T2 baseline (seed 7, 2,000 episodes, four
50-game panels), scale 0 and 25 tied at 36.0% mean / 20.0% worst panel; scale
100 fell to 33.0% mean despite a 30.0% worst panel; scale 250 fell to 32.5% /
20.0%. No scale improved mean greedy wins, so shaping remains disabled and no
five-seed confirmation is warranted.

### T4 — Decouple exploration-optimism from the greedy policy · ✅ completed
**What:** stop optimism from polluting evaluated values: (a) bootstrap unseen
next-states from **0, not `optimisticInit`** in the *target* (`qlearning.ts:151`)
while keeping optimistic init only for *action selection*; and/or (b) in eval,
break Q-ties **toward the most-visited action** (visit counts already exist,
`qlearning.ts:64`) or toward `nearestPelletDir`, instead of randomly.
**Why:** directly explains the train-wins-but-greedy-eval-0 gap (Root cause B,
last bullet). A cleaner greedy argmax may recover much of the existing policy's
latent skill for free.
**Safety:** training calls retain random tie-breaking by default; only evaluation
selects the deterministic mode. Part (a) changes the learning target, so it
still needs a key-independent A/B rather than a byte-identical claim.
**Result:** the deterministic pellet-directed tie-break improved *tabular*
greedy average score 44% (799.7 → 1152.1) on the same policy and evaluation
seeds, so it is the tabular evaluation default. A later controlled linear A/B
found the opposite: deterministic `pellet` ties reduced mean greedy wins from
27.39% to 21.33% by locking exact ties into cycles. Linear therefore defaults
to `random`; both agents expose their own `defaultEvalTieBreak`, and the GUI,
trainer, and bench honor it. The unseen-state bootstrap half remains a separate
optional experiment with no promoted default.

### T5 — Less-aliased state: coarse Pac-Man region · ✅ completed, no promotion
**Implementation:** `pacRegionGrid=3` assigns Pac-Man to one of nine row-major
regions and appends it to the tabular observation key. `pacRegionGrid=1` emits
region 0 and is the baseline default. Key version v12 correctly invalidates
old policies; key string round-trips and 3×3 geometry are unit-tested. The
bench exposes the grid and `scripts/t5-pac-region-sweep.sh` compares both
configurations from scratch.
**Result:** at seed 7, 20,000 episodes, and four 50-game panels, neither grid
won a greedy evaluation game. Grid 3 improved mean `pl_p5` **128.4 → 88.8**,
but grew the Q-table **32,008 → 54,981** states (+72%) and its best signal was
only one training win. This diagnostic tail improvement is insufficient to
promote an incompatible, larger key without a greedy-win or eval-score gain.
Keep `pacRegionGrid=1`; no five-seed confirmation is warranted.

### T7 — Extend the pellet horizon · ✅ completed
**What:** `PELLET_SEARCH_RADIUS = 12` (`observation.ts`) previously bounded the
BFS that produced `nearestPelletDir`/`nearestPelletDist`. When the last few
pellets sat farther than 12 tiles, the direction returned its "none" sentinel
(4) and the distance returned radius+1 — **every pellet feature saturated at
once**, so the agent lost the pellet direction in precisely the states that
decide the win.
**Why:** measured, not theorized. D11 (Finding #14) drove the agent into a
stable attractor of 0 wins with exactly 2 pellets left on every checkpoint,
while its *training* win rate hit an all-time high — ε-greedy noise still finds
the last pellets, the greedy policy never does. This also explains the
long-standing `p5 ≈ 55` tabular plateau and why "the agent farms 75% and stops"
keeps recurring in this log.
**Where:** `PELLET_SEARCH_RADIUS` and `bfsNearestPellet` in `observation.ts`.
**Decision (2026-07-29):** do not start by raising the radius. That would also
change `PELLET_DIST_MAX`, rescale the existing linear value feature, and
confound the test. Keep radius 12 as the fast path and, only on a miss,
continue the **same** BFS until it finds the nearest reachable pellet. Return
the far pellet's direction and true depth, while leaving the current
`min(dist, 13) / 13` feature normalization intact. This changes only the
missing direction signal; it does not revive D11's per-direction BFS or add
features. A reachable nonterminal board should then use sentinel direction 4
only if no pellet is reachable. This semantic change bumps
`OBSERVATION_KEY_VERSION` and `FEATURE_SCHEMA_VERSION`.

**Result (2026-07-29):** every predeclared v4 gate passed.

- Single-seed 60k-episode screen: **27.75% → 39.46%** mean greedy wins,
  pellet-left median 43.4 → 34.2, overall/endgame sentinel rates
  15.72%/58.13% → 0%, and no throughput loss.
- Matched confirmation, five training seeds × 20k episodes × four held-out
  panels: pooled mean **25.54% → 35.17%** (+9.63 points); seed means
  24.22–27.99% → **33.72–36.79%**; minimum worst-panel mean
  22.61% → **32.06%**; mean checkpoint p5 17.95% → **30.75%**.
- Training wins rose 9.65% → 68.69%. This agrees with greedy eval rather than
  hiding a regression, but remains secondary evidence.

The fallback is now the default observation behavior.
`OBSERVATION_KEY_VERSION` is 11 and `FEATURE_SCHEMA_VERSION` is 6.

**D11 correction:** a reconstructed 12-feature D11 probe still converged to
`0 wins, p5 = 2` with the fallback and zero sentinel observations. Post-26k
greedy win rate was 0% in the fallback cell. The horizon was a real and
high-leverage baseline defect, but it was **not the sole cause** of D11's
collapse; correlated features and linear TD dynamics remain implicated. Do
not restore that feature set unchanged.

### T6 — Full-grid CNN Double DQN · next research track
**Decision:** the simpler reward, credit-assignment, and compact-key paths have
all been screened without a new policy win. Start a separate CNN Double-DQN
research track; do not replace the promoted linear agent or its CI smoke.
**Initial architecture:** a fixed-board tensor with six planes (walls, regular
pellets, power pellets, Pac-Man, dangerous ghosts, edible ghosts), followed by
two 3×3 convolution blocks (16 then 32 channels), a 128-unit dense head, and
four action Q-values. Use a 50k-transition replay buffer, batch 64, Huber loss,
Adam, Double-DQN action selection, and a 2,000-update target sync. Add the
runtime only after the encoder and a deterministic replay/update unit test
exist; browser and headless bench must share the same agent.
**Why:** the active linear baseline is now 37.17% mean greedy wins and 32.5%
minimum worst panel after T2 confirmation, but its local hand-features cannot
represent maze-wide pellet layout. The T5 compact key reduced a diagnostic tail
but added states without greedy wins. Full-board spatial capacity is the next
untried lever.
**Gates:** first verify encoder planes, legal-action masking, replay sampling,
and deterministic one-batch loss reduction. Then run seed 7 learning curves at
2k/10k/50k episodes on the existing four panels. A candidate earns five-seed
confirmation only if it exceeds **37.17%** mean greedy wins without falling
below the **32.5%** worst-panel floor at equal or justified compute. Record
throughput and memory; stop the track if the CPU/headless path cannot maintain
reliable, reproducible evaluation.

---

## I. Infra to make the above measurable

### I1 — Single-worker reproducible baseline + fast learning smoke
**Status:** ✅ completed 2026-07-29.
**What:** `scripts/learning-smoke.sh` runs the promoted linear/T7 configuration
twice with one worker, fixed seed, 2,000 episodes, and four disjoint 50-game
panels. It requires identical `evals.csv` and summary output (except elapsed
wall time), then asserts at least 60/200 wins, 30% mean win rate, 18% on the
worst panel, and `pl_p5 = 0` on every panel. It runs in CI via
`npm run test:learning-smoke`.
**Why:** every T-item now has a fast, trustworthy reference that catches RNG,
key, reward, or evaluation regressions before a long experiment is started.
**Evidence:** the initial T7 calibration recorded 67/200 wins (33.5% mean),
20.0% worst panel, and zero `pl_p5`; a second identical run produced
byte-identical evaluation and summary data. The smoke was rerun after T2
promotion and remains reproducible at 72/200 wins (36.0% mean).

### I2 — Sweep ergonomics: expose `nStep`, `lambda`, `gamma`, shaping as first-class CLI knobs
**Done so far:** `evalPanels`, the T2 controls (`gamma`, `winBonus`,
`deathPenalty`, `pelletEscalationMax`), T1's `nStep`, and T3's
`shapingScale`/`shapingGamma` are first-class bench/sweep arguments. `lambda`
remains the linear agent's separately named L2 regularization parameter.
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

### Linear function-approximation agent · ▶ active (D8/D9)
Finding #10 described the old state-only feature model and is superseded by
D8's action-conditioned features. D9's target network stabilized that model.
The 2026-07-26 five-seed confirmation averaged 27.55% eval wins with only
0.16 percentage points of seed-to-seed standard deviation. The later soak
proved duration is not a lever, and T7's matched confirmation established the
new 35.17% mean baseline. I1/I2 measurement lock-in is next, not another soak
or α sweep.

**Soak result (2026-07-28):** ran, stopped at 85% (6h50m), analyzed. Learning
is flat across 6.5M episodes; the agent converges by episode ~2,000. It missed
the mean (21.0% vs ≥32%) and worst-panel (19.4% vs ≥25%) targets and met only
the p5 one (16.0%). Its real yield was catching a 6.4-point evaluation
regression from 9b0a880 that four prior documented runs had been scored
against — see Finding #13 and the 2026-07-28 journal entry. The linear
evaluation tie-break is back to `random`, restoring the ~27.5% baseline.

**Soak harness (2026-07-27):** every number in `test_history.md`
was scored on one hardcoded 200-maze panel (`evalEpisodes` games seeded
`1_000_000 + i`), so a repeatable mean there does not prove the policy
generalizes off that panel. `overnight-bench.ts` now takes `evalPanels=<a,b,…>`
(default `1000000` → prior behavior, verified byte-identical) and writes one
`evals.csv` row per panel with `panel` preserved in column 13.
`scripts/run-soak.sh` runs the seeds as independent processes — `merge-
policies.ts` averages tabular Q-tables and has no linear branch, so there is
nothing to federate — and reports `meanWinRate` / `worstPanelMean` /
`checkpointP5`, the three numbers the success bar is stated in.

---

## Recommended order

~~**Long-soak D9 linear**~~ (done 2026-07-28 — flat across 6.5M episodes;
duration is not a lever, see Finding #13) → ~~**feature capacity**~~ (attempted
2026-07-28, both variants regressed — see Finding #14) → ~~**T7 far-pellet
direction**~~ (done 2026-07-29, +9.63 points pooled; D11 not rescued) →
~~**I1**~~ (deterministic learning smoke, done 2026-07-29) → ~~**T2**~~
(reward/γ sweep, done 2026-07-29) → ~~**T1**~~ (n-step screen, no promotion,
done 2026-07-29) → ~~**T3**~~ (potential shaping screen, no promotion, done
2026-07-29) → ~~**T5**~~ (3×3 tabular key screen, no promotion, done
2026-07-29) → **T6** (full-grid CNN Double DQN research track).

**Standing constraint from the soak:** the linear agent converges in ~2,000
episodes. Benchmark it in minutes. If a proposed change is argued to need hours
of training to show its effect, that argument has to explain why this result
does not apply to it.

---

## Quick-reference: file map

| Area | Path |
|------|------|
| Env + rewards + EnvParams + escalation | `src/env/environment.ts` |
| Observation / state key | `src/env/observation.ts` |
| Tabular agent (update, optimism, act) | `src/rl/qlearning.ts` |
| Linear agent (active) | `src/rl/linearQlearning.ts` |
| Training loop + eval | `src/rl/trainingController.ts` |
| Reward presets | `src/rl/rewardPresets.ts` |
| Ghost AI | `src/ghosts/ghostAi.ts` |
| Bench / sweep / merge CLIs | `scripts/*` |
| Empirical log (READ FIRST) | `test_history.md` |
