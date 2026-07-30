# Test History

A running log of training experiments, configurations, and results — so future
sessions can pick up where we left off without re-litigating settled questions.

**How to use this file:**
- Read [Current State](#current-state) first to see where the agent is today.
- Skim [Findings](#findings) to avoid repeating mistakes.
- Use [Test Runs](#test-runs) to compare a new run against historical baselines.
- Append a new entry to [Test Runs](#test-runs) after every meaningful experiment.

---

## Current State

**Observation key version:** v11 (`v11:wallMask:pelletDir:gc0:gh0:gc1:gh1:lastAction:pelletsBucket:powerBucket`)
— v10 made `wallMask` tunnel-aware; v11 resolves the direction to reachable
pellets beyond radius 12. ⚠️ Policies below v11 no longer load (`load()`
discards them on key-version mismatch) and must be retrained.

### 2-Ghost Tabular — Historical Baseline

| | |
|---|---|
| **Best policy on disk** | `bench-out/20260516-224305-2g-curric07/policy-merged.json` |
| **Q-table size** | ~218k states (merged across 32 workers) |
| **Eval `p5` (best chunk avg)** | **54.4 pellets remaining** (out of ~218) |
| **Best single eval game** | 12.8 pellets remaining (worker-01 of curric07 run) |
| **Best eval win rate** | 2.5% (5/200 in a single eval pass) |
| **Status** | Retained as the tabular baseline; linear is the active confirmation track |

### 2-Ghost Linear — Active Confirmation Track

| | |
|---|---|
| **Latest run** | `bench-out/20260729-010944-t7-fallback-confirm/` |
| **Seeds** | 7, 1007, 2007, 3007, 4007 (20k episodes each) |
| **Training wins** | 68,692 / 100,000 episodes (**68.69% pooled**) |
| **Mean eval win rate** | **35.17%** (seed means 33.72–36.79%) |
| **Worst held-out panel** | **32.06%** minimum across seed-level worst panels |
| **Checkpoint fifth percentile** | **30.75%** for every seed |
| **Status** | T7, I1, and T2 confirmed; T1/T3/T5 screened with no promotion; T6 is next |

### 3-Ghost — Paused

| | |
|---|---|
| **Best policy on disk** | `bench-out/ab-3a-20260516/policy-latest.json` |
| **Q-table size** | ~133k states (single worker) |
| **Best eval `p5`** | 35 pellets remaining (single worker) |
| **Best training wins** | 4,019 in 1 hour (1 worker, curriculum=0.2) |
| **Greedy eval wins** | **0** — never reached threshold |
| **Status** | Shelved until 2-ghost is solidly solved. Transfer-learning from 2g policy is the plan. |

### 4-Ghost — Untouched

| | |
|---|---|
| **Best policy on disk** | `bench-out/_archive/pre-ab-tests/run5-4ghosts/policy-latest.json` (stale) |
| **Status** | Only a 25-min stale run exists. Not in active development. |

**Active reward preset (default):**
- `pelletReward=5` × pellet-escalation (1×→10× as pellets clear)
- `powerPelletReward=20` × pellet-escalation
- `deathPenalty=-50`
- `stepPenalty=-0.1`
- `survivalReward=0`
- `ghostEatReward=30` (×combo)
- `winBonus=1000`

**Active hyperparameter defaults (shared by GUI and overnight bench):**
- Tabular: `alpha=0.1  gamma=0.99`
- Tabular exploration: `eps=0.5  epsDecay=0.999997  epsMin=0.20`
- Tabular endgame: `endgameEpsilon=0.25  endgameBucketThreshold=1`
- Linear: `alpha=0.02  gamma=0.997`
- Linear exploration: `eps=0.3  epsDecay=0.9995  epsMin=0.05`
- Linear stabilization: `targetSyncSteps=2000`
- `optimisticInit=50` (tabular Q-init)
- Bench: `endgameCurriculum=0.90  evalEpisodes=200  evalEvery=2000  maxSteps=1000`

---

## Configuration Knobs Reference

| Knob | Where | Default | Range tested | What it does |
|---|---|---|---|---|
| `endgameCurriculum` | CLI | 0.90 | 0, 0.2, 0.5, 0.7, 0.9 | P(start episode in 10-25% pellets) |
| `endgameEpsilon` | CLI | 0.25 tabular / 0 linear | 0, 0.25, 0.4 | ε floor when in late-game bucket |
| `endgameBucket` | CLI | 1 | 1 | bucket ≤ this triggers endgameEps |
| `winBonus` | env preset | 1000 | 200, 1000 | reward for clearing all pellets |
| `optimisticInit` | hyper | 50 | -1, 50 | initial Q-value for unseen state-actions |
| `maxSteps` | CLI | 1000 | 400, 800, 1000 | episode timeout |
| `evalEpisodes` | CLI | 200 | 30, 50, 200 | greedy eval games per pass |

---

## Code Change Log

Reverse-chronological. Each commit hash is in `git log` for full diffs.

### Infrastructure / observability
| Hash | Change | Why |
|---|---|---|
| `25dc5d4` | Validate `loadPolicy=` up-front; nullglob the merge step | 32 silent crashes if path typo'd |
| `da23704` | Timestamped+labeled top-level folders, fail-fast on collision (Options A + B) | Old runs got mixed across executions |
| `9e1ddde` | Parallel federated training (run-parallel.sh + merge-policies.ts) | Use all 32 threads |
| `1321043` | Per-eval pelletsLeft quintile histogram | distinguish "lucky run" from consistent close finishes |
| `ebda317` | Bump default `evalEpisodes` 30 → 200 | SE was ~100 pts, larger than typical learning gain |
| `ac7c178` | Add `pelletsLeft`, `termReason`, `stdScore`, `wins` columns | original 0-win diagnosis was blind |
| `443f952` | Bench reapplies CLI hypers after `agent.load()` | Bug: load() overrode CLI ε/α/γ |

### Agent / environment
| Hash | Change | Why |
|---|---|---|
| _(deep-dive-audit branch, D3.2)_ | **Frightened ghosts may now reverse to flee** (flee over `legal`, not the reverse-filtered `candidates`) | ⚠️ **Behavior change — affects baseline.** Widens Pac's ghost-eat window after a power pellet so `ghostEatReward` stays learnable; a cornered edible ghost no longer stalls. Eval/training numbers from before this commit are not directly comparable for power-pellet-heavy play. |
| `6fa8952` | Add `powerPelletsLeftBucket` to obs (key v7, 3 buckets) | Agent had no signal about power-pellet panic-button availability |
| `212e472` | State-conditional ε floor (`endgameEpsilon`) | Concentrate exploration in late-game (Priority 3b) |
| `1b53afb` | Env-level endgame-curriculum reset (`clearPelletsTo`) | Force exposure to endgame states (Priority 3a) |
| `7ebbac3` | Bump default `maxSteps` 400 → 800 | 400 made winning physically impossible |
| `6fe2bdf` | Add `pelletsRemainingBucket` to obs (key v6, 5 buckets) | Same obs in opening vs endgame is wrong |
| `a2cc282` | Reward reshape: `winBonus=1000`, drop `survivalReward`, pellet-escalation 1×→6× | Agent was paid to loiter, not to win |
| `4687cce` | Optimistic Q-init (`-1` → `50`) | Pessimistic init caused premature commitment |
| `7260980` | Slow ε decay (`0.999`→`0.99999`), raise floor (`0.05`→`0.15`) | ε hit floor by ep 4.6k — agent locked into mediocre policy |

---

## Findings

Things we now consider settled. Don't waste time re-testing these unless something fundamental changes.

1. **Win rate of 0 was *not* an unsolvable problem.** It required ALL of: optimistic init, slower ε decay, higher ε floor, win bonus 5×, pellet-escalation reward, endgame curriculum, larger state space (pelletsRemainingBucket + powerPelletsLeftBucket), and `maxSteps=800`. Removing any one of these may regress to 0% wins.

2. **`load()` does not preserve CLI overrides — bench reapplies them.** If you write a new caller of `QLearningAgent.load()` outside the bench, remember this. (`src/App.tsx` *intentionally* keeps the saved hyper; bench overrides.)

3. **The `default` reward preset was a survival preset in disguise.** `survivalReward=0.02 × 150 steps = +3` reward just for being alive, dwarfing the win bonus of 200. Now `survivalReward=0` and `winBonus=1000` with pellet-escalation. Don't reintroduce survivalReward without re-testing.

4. **`maxSteps=400` makes winning physically impossible.** A maze has ~280 pellets, and an optimal path is ~290 steps. 400 leaves no slack for ghost evasion. Use `≥800`.

5. **Pessimistic Q-init (`-1`) was bad.** After the first positively-rewarded action in a state, the agent stopped exploring other actions. Optimistic init (`50`) is the default. Tests assume this; pass `optimisticInit: -1` explicitly if you need the old behavior.

6. **`endgameEpsilon=0.4` *hurts* when combined with curriculum.** The ab-3ab run showed combining 3a+3b yielded 670× *fewer* wins than 3a alone. The hypothesis is `endgameEpsilon` thrashes the late-game policy the curriculum is trying to teach. Don't enable both without justification.

7. **Eval `winRate` rounds to 0 below the noise floor.** Always look at the `wins` column (raw count) and `minPelletsLeft` / `p5` — they show signal long before `winRate` shows above 0.000.

8. **The `run-overnight.sh` auto-detect-seed-across-bench-out behavior was a bug**, not a feature. It silently mixed two separate executions. Current behavior: only auto-pick seed from the *current* top-level folder; cross-experiment seeds must be passed explicitly.

9. **Folder convention:** `bench-out/<YYYYMMDD-HHMMSS>-<desc>/{run*,worker-*}/`. Both runners honor it. Old-style flat folders (`bench-out/run1-baseline/`) are pre-2026-05-16 and live in `bench-out/_archive/`.

10. **Historical (superseded by Finding #12): the state-only linear agent was far behind tabular.** After D5.9 gave the linear agent continuous pellet/ghost distances (`nearestPelletDist`, `nearestGhostDists`) instead of re-discretized buckets, a 5-min `algorithm-compare.sh` run (seed 7, `stepPenalty=-0.02`, `endgameCurriculum=0.90`, linear `alpha=0.01`) still showed: tabular greedy eval **avgScore 958** (eats down to 133 pellets left) vs linear **avgScore 107**, dying in ~36 steps with near-zero score variance — a degenerate policy that barely moves. The continuous representation was necessary but not sufficient. A follow-up **α sweep** (2026-06-17, 6 configs × 5 min, seed 7; see Test Runs) confirmed α was not the missing lever: viable α values clustered near ~300 peak eval while α=0.3 diverged. D8 later identified the structural issue: every action saw the same state-only feature vector.

11. **Greedy/eval tie-breaking matters — `random` ties throw away tabular policy quality (roadmap T4).** `QLearningAgent.act()` historically broke ties between equal-max Q-values *randomly*. Because optimistic init leaves unvisited slots at 50, aliased/under-trained states have many ties, so under ε=0 eval the greedy policy partly degrades to a random walk. A deterministic **`pellet` tie-break** (steer a tied choice toward `nearestPelletDir`, else most-visited) lifts greedy **avgScore +44%** (799.7 → 1152.1) on the *same* Q-table and eval seeds, no retraining (2026-06-27; fresh 3-min v9 single-worker policy, 95k states, 200 eval games; `visits` mode was ~neutral at 773.9). `pellet` is the tabular GUI/headless evaluation default; exploratory training still uses random ties. Finding #13 supersedes the earlier shared-default conclusion for the linear agent, which defaults to `random`. A short 3-worker v9 smoke confirmed the federated path uses the tabular default, but was too compute-limited to establish a learning improvement. **Next:** T4(a) (bootstrap unseen next-states from 0, not optimisticInit) remains a separate, unvalidated experiment.

12. **Action-conditioned features made linear Q-learning the leading 2-ghost agent.** D8 replaced the structurally inadequate `w_a·f(s)` state-only representation with shared action-conditioned features `w·f(s,a)`. D9 added a 2,000-update target network to stabilize bootstrap targets. A corrected 2026-07-26 seed-7 comparison averaged 27.7% eval wins, and the follow-up five-seed run confirmed **27.55%** (seed means 27.42–27.84%, seed std 0.16 points, 95% t-interval 27.35–27.76%). Across 763,924 episodes it recorded **79,320 training wins (10.38%)**. Mean final evaluation was 29.30%; 99.3–99.7% of each seed's checkpoints had `p5=0`. Individual checkpoints still occasionally dipped as low as 1.5%, so a longer soak should measure tail stability even though the cross-seed mean is repeatable. **(Soak run 2026-07-28 — see Finding #13; the tail does *not* improve with training time.)**

13. **The linear agent converges in ~2,000 episodes, and `pellet` tie-breaking costs it 6 points.** Two results from the 2026-07-28 soak (5 seeds × ~6.5M episodes, stopped at 85%):
    - **Training duration is not a lever.** Win rate is flat end to end. Seed 7 by decile: 20.86% (ep 2k–648k) → 21.28% (ep 5.8M–6.5M); first-100 vs last-100 checkpoints across all five seeds differ by −0.30 to +0.58 points. The agent is converged by its first checkpoint at episode 2,016. Cap linear runs at minutes, not hours.
    - **Finding #11's `pellet` default does not transfer from the tabular agent to the linear one.** 9b0a880 (2026-07-27) pointed the linear agent at it on the premise that continuous Q-values never tie. Measured: **1.7% of multi-action decisions are exact ties** (`features[0]` bias and `features[3]` pellet-distance are action-independent and cancel in the argmax; most of the rest are binary). Deterministic tie resolution cycles. A/B at seed 7, 8 min, tie-break the only variable: **`random` 27.39% vs `pellet` 21.33%** mean eval wins, max checkpoint 37.0% vs 27.0%, avgLen 362.4 vs 396.0, training win rate identical (10.27% vs 10.28%). The linear default is now `random`; the tabular default stays `pellet`. Each agent declares its own `defaultEvalTieBreak`.
    - **Held-out panels were not where the surprise was.** Pooled over ~16k checkpoints each: panel 1000000 = 21.3%, 2000000 = 21.7%, 3000000 = **19.4%**, 4000000 = 21.8%. The historical single panel was never badly unrepresentative.

14. **The endgame blind spot is the pellet horizon, and it is measured.** D11's feature work (2026-07-28) failed in a way that localized the problem precisely. Adding a second pellet-distance feature drove the agent into a stable attractor: 0 wins with **exactly 2 pellets left**, on every checkpoint from ~26k episodes onward, while its *training* win rate hit an all-time high of 12.06%. `PELLET_SEARCH_RADIUS=12` means the last pellets are beyond the BFS horizon, so every pellet-distance feature returns its "none in range" sentinel (1.0) and carries **no gradient in exactly the states that decide the win**. The agent is not failing to learn the endgame; it cannot see it. At ε=0.05 random exploration still stumbles into the last pellets, which is why training wins rose while greedy eval collapsed to ~1%. **Consequence:** fix the horizon before adding any further pellet-direction/distance feature — otherwise more features simply add weight mass to a saturated signal. This is the concrete, measurable form of Root cause A ("the state representation aliases away the maze").

15. **Far-pellet direction was a large baseline lever, but not the complete D11 diagnosis.** A matched five-seed/four-panel comparison at 20k episodes raised pooled greedy wins **25.54% → 35.17%** (+9.63 points), seed means 24.22–27.99% → 33.72–36.79%, minimum worst-panel mean 22.61% → 32.06%, and mean checkpoint p5 17.95% → 30.75%. The bounded baseline used sentinel direction 4 on 15.30% of all eval decisions and 57.12% of bucket-0 decisions; continuing the same BFS removed both rates entirely. Training wins rose 9.65% → 68.69%. However, a documented reconstruction of D11's 12-feature superset still collapsed with the fallback: post-26k greedy win rate was **0%** and 64/65 checkpoints had `p5=2`, despite zero sentinel observations. This corrects Finding #14's single-cause wording: the horizon hid a real action signal and was worth fixing, but correlated feature/TD dynamics also caused the D11 attractor.

---

## Test Runs

Organized by ghost count, reverse-chronological within each section. Each entry:
config, top-level stats, what it told us.

**Quick index:**
- [2-Ghost runs](#2-ghost-runs) — 13 runs, active development track
- [3-Ghost runs](#3-ghost-runs) — 5 runs, paused at 0 greedy wins
- [4-Ghost runs](#4-ghost-runs) — 1 stale run
- [Mixed / Pre-fix](#mixed--pre-fix-runs) — pre-2026-05-16 layout, archived

---

### 2-Ghost runs

#### 2026-07-30 — `cnn-runner-update-smoke` — ⚠️ CPU throughput gate failed

- **Goal:** verify the isolated T6 CNN runner can execute a real replay update
  and report its runtime metrics before any learning-curve claim.
- **Config:** one episode, one max step, random action (`eps=1`), one batch-1
  update, no evaluation; pure-JS TensorFlow.js CPU backend.
- **Result:** the runner emitted `summary.json` with one finite Huber loss and
  one update, but achieved only **1.1 environment steps/sec**. The no-update
  runner smoke reached 1,169.7 steps/sec, isolating gradient computation as the
  blocker.
- **Verdict:** ⚠️ runner correct; CPU backend is not viable for the T6 curves.
  Do not compare policy quality yet. Artifacts: `bench-out/cnn-runner-smoke/`
  and `bench-out/cnn-runner-update-smoke/`.

#### 2026-07-29 — `t5-pac-region-screen` — ❌ no region-key promotion

- **Goal:** T5: test whether a 3×3 Pac-Man region resolves enough tabular
  observation aliasing to improve from-scratch greedy evaluation.
- **Config:** tabular two-ghost runs, seed 7, 20,000 episodes, four 50-game
  panels, with identical endgame curriculum and `pacRegionGrid={1,3}`.
- **Result:** grid 1: 32,008 Q states, 0 training/greedy wins, mean
  `pl_p5=128.375`; grid 3: 54,981 states (+72%), one training win, 0 greedy
  wins, mean `pl_p5=88.825`.
- **Verdict:** ❌ despite a better pellet-tail diagnostic, no greedy-win or
  eval-score gain justifies an incompatible larger key. Retain grid 1. Artifact:
  `bench-out/20260729-225926-t5-pac-region-screen/`.

#### 2026-07-29 — `t3-potential-shaping-screen` — ❌ no shaping promotion

- **Goal:** T3: test policy-invariant pellet-progress shaping without changing
  observations or base rewards.
- **Config:** seed 7, 2,000 episodes, four 50-game panels, promoted linear/T2
  settings, and `Φ(s)=-scale·pelletsLeft/totalPellets` with
  `shapingGamma=0.997`; varied `scale={0,25,100,250}` only.
- **Result:** scale 0: **36.0%** mean / 20.0% worst panel / 27.0% training
  wins; scale 25: 36.0% / 20.0% / 27.3%; scale 100: 33.0% / **30.0%** /
  28.05%; scale 250: 32.5% / 20.0% / 28.45%.
- **Verdict:** ❌ no scale improved mean greedy wins; keep shaping disabled and
  skip five-seed confirmation. Artifact:
  `bench-out/20260729-202137-t3-potential-shaping-screen/`.

#### 2026-07-29 — `t1-nstep-screen` — ❌ no n-step promotion

- **Goal:** T1: test whether multi-step returns improve terminal reward credit
  assignment at the promoted linear/T2 baseline.
- **Config:** seed 7, 2,000 episodes, four 50-game panels, with only
  `nStep={1,3,5,10}` varied. All cells used linear `gamma=0.997`,
  `deathPenalty=-50`, `pelletEscalationMax=10`, `alpha=0.02`, target sync
  2,000, and endgame curriculum 0.90.
- **Result:** n=1: **36.0%** mean / 20.0% worst panel / 27.0% training wins;
  n=3: 31.5% / **22.0%** / 20.7%; n=10: 29.5% / 20.0% / 6.7%; n=5: 0.0% /
  0.0% / 14.4%, with mean `pl_p5=223.3`.
- **Verdict:** ❌ no candidate improved mean greedy wins or the pellet tail;
  retain `nStep=1` and skip five-seed confirmation. Artifact:
  `bench-out/20260729-201510-t1-nstep-screen/`.

#### 2026-07-29 — `t2-reward-screen` + five-seed confirmation — ✅ defaults promoted

- **Goal:** T2: test whether a longer discount horizon, stronger terminal
  rewards, lower death cost, or steeper late-pellet reward makes endgame clears
  more valuable to the promoted linear/T7 agent.
- **Screen:** 36 cells at seed 7, 2,000 episodes, four 50-game panels:
  `gamma={0.99,0.997,0.999}`, `winBonus={1000,2500,5000}`,
  `deathPenalty={-100,-50}`, `pelletEscalationMax={6,10}`. The best cell was
  `0.997/1000/-50/10`: **36.0%** mean versus **33.5%** baseline.
- **Confirmation:** the baseline and candidate each ran seeds
  `{7,1007,2007,3007,4007}`, 2,000 episodes, four 200-game panels. Baseline
  mean was **33.25%** (seed range 30.75–37.0%; minimum worst panel 29.5%);
  candidate mean was **37.17%** (37.0–37.87%; minimum worst panel 32.5%). All
  five candidates beat their paired baseline seed.
- **Verdict:** ✅ Promote linear `gamma=0.997`, shared `deathPenalty=-50`, and
  `pelletEscalationMax=10`; retain `winBonus=1000`. Artifacts:
  `bench-out/20260729-193350-t2-reward-screen-final/`,
  `bench-out/20260729-193450-t2-baseline-confirm/`, and
  `bench-out/20260729-193501-t2-candidate-confirm/`.

#### 2026-07-29 — `i1-learning-smoke` (2 × 2,000 episodes, seed 7) — ✅ reproducible CI gate

- **Goal:** Lock the promoted T7 linear baseline into a fast deterministic
  regression check before the next tuning item.
- **Config:** two identical single-worker runs: `algorithm=linear ghosts=2
  seed=7 episodes=2000 endgameCurriculum=0.90 stepPenalty=-0.02 alpha=0.02
  targetSyncSteps=2000`, with four disjoint 50-game evaluation panels.
- **Result:** both `evals.csv` files and summaries (excluding elapsed wall
  time) were byte-identical. The final evaluation recorded **67/200 wins**,
  **33.5%** mean win rate, **20.0%** worst-panel win rate, and `pl_p5=0` on all
  four panels.
- **Verdict:** ✅ `npm run test:learning-smoke` now enforces conservative floors
  of 60 wins, 30.0% mean, 18.0% worst panel, and `pl_p5=0`; CI runs it on every
  change. It guards baseline integrity, not future measured improvements.

#### 2026-07-29 — `t7-{bounded,fallback}-confirm` (2 × 5 seeds × 20k episodes) — ✅ fallback promoted

- **Goal:** Matched confirmation of the passing far-pellet-direction screen.
- **Config:** seeds `{7,1007,2007,3007,4007}`, `algorithm=linear`, two ghosts,
  `episodes=20000`, `endgameCurriculum=0.90`, `stepPenalty=-0.02`,
  `evalEvery=2000`, four panels × 200 games. The only variable was whether the
  radius-12 BFS stopped or continued to the nearest reachable pellet.
- **Result:**

  | metric | bounded | fallback |
  |---|---:|---:|
  | pooled mean eval WR | 25.54% | **35.17%** |
  | seed mean range | 24.22–27.99% | **33.72–36.79%** |
  | minimum worst-panel mean | 22.61% | **32.06%** |
  | mean checkpoint p5 | 17.95% | **30.75%** |
  | training WR | 9.65% | **68.69%** |
  | overall / bucket-0 sentinel | 15.30% / 57.12% | **0% / 0%** |

- **Verdict:** ✅ Promote. `OBSERVATION_KEY_VERSION` 10→11 and
  `FEATURE_SCHEMA_VERSION` 5→6.
- **Artifacts:** `bench-out/20260729-011217-t7-bounded-confirm/` and
  `bench-out/20260729-010944-t7-fallback-confirm/`.

#### 2026-07-29 — `t7-ab` (4 cells × 60k episodes, seed 7) — ✅ v4 / ❌ D11 rescue

- **Goal:** Screen v4 and test whether the far-direction fallback rescues the
  D11 `0 wins, p5=2` attractor.
- **Config:** one historical panel × 200 games, `evalEvery=500`; cells were v4
  bounded/fallback and reconstructed-D11 bounded/fallback. The D11 source was
  never committed or staged, so the rescue uses the recorded effective
  12-feature superset, not a byte-identical recovery.
- **Result:** v4 mean eval WR **27.75% → 39.46%** and median pellets left
  43.4 → 34.2. D11 bounded/fallback mean WR was 1.02%/2.16%; post-26k it was
  0.015%/0%, with the `p5=2` attractor in both cells. Fallback sentinel rate was
  zero.
- **Verdict:** v4 screen passed. D11 rescue failed, falsifying the horizon as
  the sole cause of that feature set's collapse.
- **Artifacts:** `bench-out/20260729-t7-ab/`.

#### 2026-07-28 — `d11-features` (7 runs × 8 min) — ❌ both variants regressed, reverted

- **Goal:** Cut the 1.7% exact-tie rate (Finding #13) by giving the linear agent action-conditioned features: per-action pellet distance, pellet on the destination tile, dead-end and escape-breadth indicators.
- **Config:** seed 7 (plus 5 seeds for attempt 1), `algorithm=linear ghosts=2 durationMin=8 endgameCurriculum=0.90 stepPenalty=-0.02 evalEpisodes=200 evalEvery=500`, random tie-break throughout.
- **Result:**

  | features | α | mean eval WR | max | <5% | train WR |
  |---|---|---:|---:|---:|---:|
  | v4 baseline | 0.02 | **27.39%** | 37.0% | 1 | 10.27% |
  | v5 attempt 1 (replaced f1/f3) | 0.02 | 22.63% | 45.0% | 18 | 11.07% |
  | v5 attempt 1 | 0.01 | 23.59% | 36.0% | 5 | 10.90% |
  | v5 attempt 2 (superset, 12 feat) | 0.02 | **0.99%** | 32.0% | 163 | **12.06%** |
  | v5 attempt 2 | 0.01 | 0.60% | 27.0% | 162 | 11.66% |

  Attempt 1 across five seeds: 22.63/21.88/22.51/22.69/21.60% (mean 22.2%) — reproducible, not seed noise.
- **Verdict:** ❌ Reverted to v4 features. Attempt 2 converges to a stable **0-win, `p5=2` attractor** — it clears the maze except the last two pellets and never finishes. Cause: `PELLET_SEARCH_RADIUS=12` means the final pellets sit beyond the horizon, so pellet-distance features saturate at their 1.0 sentinel and carry no gradient in exactly the states that decide the win. Two saturating features instead of one made "wander safely" outrank "find the last pellet". See Finding #14.
- **Kept from this work:** the tunnel-aware `wallMask` fix (a real bug — the mask probed raw `pac.x + dx` while `canMove()` wraps first, so a legal tunnel move was encoded as a wall), `OBSERVATION_KEY_VERSION` 9→10, `FEATURE_SCHEMA_VERSION` 4→5, plus the `evalTieBreak` CLI knob and per-eval `tie%` reporting.
- **Post-revert verification:** v4 features + tunnel fix, seed 7 → **27.79%** mean eval wins (263 checkpoints), back at baseline.

#### 2026-07-28 — `linear-soak` (5 seeds × 6h50m, stopped at 85% of 8h)

- **Goal:** Roadmap "long-soak D9 linear" — test whether longer training raises the mean and removes the low checkpoints. Targets: ≥32% mean, ≥25% worst held-out panel, ≥15% checkpoint p5.
- **Config:** `./scripts/run-soak.sh durationMin=480`; seeds `{7,1007,2007,3007,4007}` as independent processes, `algorithm=linear`, two ghosts, `endgameCurriculum=0.90`, `stepPenalty=-0.02`, `alpha=0.02`, target sync 2000, `evalPanels=1000000,2000000,3000000,4000000`, 200 games per panel every 2,000 episodes.
- **Result:** ~6.5M episodes and ~3,200 checkpoints per seed.

  | seed | episodes | train WR | mean eval WR | worst panel | ckpt p5 |
  |---:|---:|---:|---:|---:|---:|
  | 7 | 6,487,296 | 10.54% | 21.02% | 19.35% | 16.0% |
  | 1007 | 6,556,964 | 10.53% | 20.98% | 19.38% | 16.0% |
  | 2007 | 6,548,512 | 10.54% | 21.11% | 19.46% | 16.0% |
  | 3007 | 6,570,467 | 10.51% | 21.09% | 19.48% | 16.0% |
  | 4007 | 6,552,860 | 10.54% | 21.06% | 19.41% | 16.0% |

- **Verdict:** ❌ on mean and worst-panel, ✅ on p5 only. **Learning is flat** — see Finding #13. The run's real yield was diagnostic: matched against the 2026-07-26 baseline on the same seeds, panel, and episode range (<160k), it scored 6.4 points lower on *every* seed with training unchanged, which isolated the 9b0a880 evaluation-tie-break regression.
- **Note:** the processes were SIGKILLed, so no `summary.json` was written; all numbers are recomputed from `evals.csv`/`episodes.csv`. `run-soak.sh` now recovers from this case.
- **Artifacts:** `bench-out/20260727-235405-linear-soak/`.

#### 2026-07-28 — `ab-tiebreak` (2 × 8 min, seed 7)

- **Goal:** Isolate whether 9b0a880's evaluation tie-break explains the soak's 6.4-point shortfall against the baseline.
- **Config:** `algorithm=linear ghosts=2 seed=7 durationMin=8 endgameCurriculum=0.90 stepPenalty=-0.02 evalEpisodes=200 evalEvery=500`, varying only the new `evalTieBreak` knob.
- **Result:**

  | tie-break | ckpts | mean eval WR | min | max | avgLen | train WR |
  |---|---:|---:|---:|---:|---:|---:|
  | `pellet` | 195 | 21.33% | 1.5% | 27.0% | 396.0 | 10.28% |
  | **`random`** | 205 | **27.39%** | 3.5% | **37.0%** | 362.4 | 10.27% |

- **Verdict:** ✅ Confirmed. `random` reproduces the baseline (27.39 vs 27.50), `pellet` reproduces the soak (21.33 vs 21.31), and training win rates match to four decimals — the effect is entirely in evaluation. Linear default reverted to `random`.
- **Artifacts:** scratch run; numbers recorded here and in the journal.

#### 2026-07-26 — `linear-multiseed` (5 seeds × 8 min)

- **Goal:** Determine whether the D8/D9 linear gain repeats beyond seed 7.
- **Config:** Seeds `{7,1007,2007,3007,4007}`, two ghosts,
  `algorithm=linear`, `endgameCurriculum=0.90`, `stepPenalty=-0.02`,
  production hyperparameters (`alpha=0.02`, target sync 2000), 200 greedy
  evaluation games every 500 episodes.
- **Result:**

  | seed | episodes | train win rate | mean eval wins | last-30 | final |
  |---:|---:|---:|---:|---:|---:|
  | 7 | 158,222 | 10.31% | 27.50% | 26.40% | 27.50% |
  | 1007 | 158,886 | 10.39% | 27.84% | 28.02% | 37.00% |
  | 2007 | 148,042 | 10.32% | 27.48% | 27.48% | 30.00% |
  | 3007 | 148,869 | 10.49% | 27.53% | 28.48% | 30.00% |
  | 4007 | 149,905 | 10.41% | 27.42% | 26.98% | 22.00% |

  Aggregate mean eval win rate was **27.55%**; seed-to-seed standard deviation
  was **0.16 percentage points** and the five-seed 95% t-interval was
  **27.35–27.76%**. Pooled training win rate was **10.38%**.
- **Verdict:** ✅ The gain is repeatable across training seeds and is not a
  seed-7 artifact. Checkpoint-level instability remains (minimum individual
  checkpoint 1.5%), but it barely changes the per-seed run means.
- **Artifacts:** `bench-out/20260726-044028-linear-multiseed/`.

#### 2026-07-26 — `linear-vs-tabular` (8 min each, corrected environment)

- **Goal:** Re-run the D8/D9 comparison after the 2026-07-21 correctness fixes
  and determine whether learning performance improved.
- **Config:** `algorithm-compare.sh durationMin=8`, seed 7, two ghosts,
  `endgameCurriculum=0.90`, `stepPenalty=-0.02`, production algorithm defaults;
  200 greedy evaluation games every 500 training episodes.
- **Result:**

  | algorithm | episodes | train wins | mean eval wins | final eval | mean eval score |
  |---|---:|---:|---:|---:|---:|
  | Tabular | 309,916 | 1,023 (0.33%) | 0.0% | 0/200 | 773.24 |
  | **Linear** | 174,614 | **18,048 (10.34%)** | **27.7%** | **61/200 (30.5%)** | **3899.51** |

  Linear checkpoint win-rate std/min/max was 5.7% / 3.5% / 37.0%; its
  last-30 mean was 29.9% and every checkpoint had `p5=0`.
- **Verdict:** ✅ Strong improvement over both the pre-D8 linear agent and D9's
  reported 23.0% checkpoint mean (27.7%, +4.7 percentage points). The last-30
  mean improved from 25.4% to 29.9% (+4.5 points). Because this is one
  deterministic seed, confirm with multiple seeds before changing the shipped
  algorithm default.
- **Artifacts:** `bench-out/20260726-031523-linear-vs-tabular/`.

#### 2026-07-25 — `t4-pellet-default-v9` (3 workers × 5 min, from scratch)

- **Goal:** finish Roadmap T4(b): make `pellet` the shared greedy-evaluation
  default, verify exploratory training still uses random ties, and exercise the
  default through a fresh federated v9 train/merge/eval cycle.
- **Config:** `./scripts/run-parallel.sh -j 3 desc=t4-pellet-default-v9
  durationMin=5 ghosts=2 evalEpisodes=200`; default tabular hyperparameters
  (`alpha=0.1`, `gamma=0.99`, `endgameCurriculum=0.90`, `endgameEps=0.25`),
  seeds 7/1007/2007. The container exposed only 3 CPUs, so this is a
  **15-worker-minute smoke**, not a full 32-worker baseline replacement.
- **Training result:** 193,776 aggregate episodes, **3 wins** (2/0/1 by
  worker), and 121,121 states in the merged policy.
- **Merged-policy greedy eval:** 200 deterministic full-maze games with the
  `pellet` default: **0 wins**, **`pl_p5=113.65`**, **avgScore=884.26**
  (min pellets left 16).
- **Verdict:** ✅ the v9 federated pipeline trains, merges, and evaluates with
  the new default. The result does **not** prove better learning: the tie-break
  changes evaluation action selection, not training, and this from-scratch run
  used far less compute than the historical baseline. No training/reward
  defaults were changed. A full-duration v9 federated run remains necessary.

#### 2026-06-27 — `t4-eval-tiebreak` (eval-only A/B, no retrain)

- **Goal:** Roadmap T4 — does a deterministic greedy tie-break recover policy
  quality that `random` ties were discarding at eval time?
- **Config:** One fresh **v9** single-worker policy (3 min, seed 7, default preset,
  α=0.1, `endgameCurriculum=0.90`, `endgameEps=0.25`, ghosts=2 → 158k episodes,
  95k states, 39 train wins). Same policy + same eval seeds (`0xE0A1`, 200 games),
  only the `evaluate(..., tieBreak)` mode varies.
- **Result:**

  | tie-break | avgScore | avgLength | wins/200 |
  |---|---|---|---|
  | `random` (baseline) | 799.7 | 137.2 | 0 |
  | `visits` | 773.9 | 136.6 | 0 |
  | **`pellet`** | **1152.1** | 157.7 | 0 |

- **Verdict:** ✅ `pellet` tie-break = **+44% greedy avgScore** for free (no
  retrain). Confirms the eval-degradation diagnosis (Root cause B). 0 wins across
  all modes is expected — a 3-min policy is far too weak for a full clear; the
  score/length lift is the signal. See [Findings #11](#findings).
- **Next:** flip the eval default to `pellet`, then re-measure once a real
  federated policy is retrained to v9 (the old best-on-disk policies are v7/v8 and
  no longer load). Pair with T4(a) and T1.

#### 2026-06-17 22:10 — `linear-alpha-sweep` (6 × 5 min, seed 7)

- **Goal:** Finding #10 follow-up — does any learning rate make the linear agent (post-D5.9 continuous features) competitive?
- **Config:** 6 linear workers, `alpha ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}`, seed 7, `stepPenalty=-0.02`, `endgameCurriculum=0.90`, `endgameEps=0.25`, `evalEpisodes=100`. (λ not swept — hard-coded 0 in the bench.)
- **Result (best eval row per worker):**

  | α | peak eval avgScore | fewest pelletsLeft | note |
  |---|---|---|---|
  | 0.001 | 153 | 246 | too slow |
  | 0.003 | 312 | 225 | viable |
  | 0.01 | 277 | **202** | viable (current default) |
  | 0.03 | **324** | 221 | viable |
  | 0.1 | 293 | 223 | viable |
  | 0.3 | 20 | 286 | **diverged** |
  | *tabular ref* | *961* | *94* | *(5-min compare run)* |

  All training-win counts were **0**.
- **Verdict:** ❌ α is not the lever. Viable band α∈[0.003,0.1] all cluster ~300 peak eval (single-seed noise); 0.001 too slow, 0.3 diverges. Best linear (~324) is still ~3× below tabular (~961) and never wins. Default α=0.01 left unchanged (in-band). See [Findings #10](#findings).
- **Next (deferred):** the linear ceiling is model-bound — richer features (raw distance maps, ghost edibility/heading) and/or L2 (needs a `lambda` CLI knob added to the bench) before the linear path is worth more time.

#### 2026-06-17 17:27 — `linear-vs-tabular` (5 min each, post-D5.9)

- **Goal:** Validate D5.9 (#19) — does giving the linear agent *continuous* pellet/ghost distances make it competitive with tabular?
- **Config:** `algorithm-compare.sh durationMin=5` (seed 7, `stepPenalty=-0.02`, `endgameCurriculum=0.90`, `endgameEps=0.25`; tabular α=0.1, linear α=0.01).
- **Result:** ❌ Linear still far behind.
  - Tabular: 178k episodes, 84 train wins, mean score (last 1k) **215**; greedy eval **avgScore 958**, avgLength 152, minPelletsLeft 133.
  - Linear: 243k episodes, **0** train wins, mean score **34**; greedy eval **avgScore 107**, avgLength **36**, minPelletsLeft 271, stdScore 3.3 (degenerate — barely moves).
- **Verdict:** Continuous features are a correct representation fix but not sufficient. See [Findings #10](#findings). Defaults unchanged — tabular stays the baseline.
- **Next (deferred):** a dedicated linear-tuning sweep (α, feature richness) before reconsidering the linear path.

#### 2026-05-16 22:43 — `2g-curric07` (45 min)

- **Goal:** Test if curriculum knob still has headroom past 0.5.
- **Config:** `-j 32 durationMin=45 ghosts=2 endgameCurriculum=0.7` loaded from `20260516-193418-2g-aggressive/policy-merged.json`
- **Result:** ✅ p5 dropped to **54.37** (chunk 10), meeting ≤55 threshold.
  - 1,225 greedy eval wins (**1,633/hr** — new high)
  - Best single eval: 5/200 wins, p5=12.8 (worker-01)
  - Q-states merged: 217,881
  - Wins/chunk declining 210 → 89 (interpretation: harder eval distribution under new training, not policy degradation)
- **Verdict:** Green on p5. Diminishing-but-real returns from curriculum (0.2→0.5→0.7 yielded −9, −4 pts each step).
- **Next:** 4-hour soak at curriculum=0.7 before pivoting to reward shaping.

#### 2026-05-16 19:34 — `2g-aggressive` (2 hr)

- **Goal:** Test if aggressive curriculum (0.5) breaks the 4h-baseline plateau at p5≈66.
- **Config:** `-j 32 durationMin=120 ghosts=2 endgameCurriculum=0.5` loaded from overnight-2g
- **Result:** 🟡 Yellow.
  - p5: 68 → 58 by chunk 2, plateau at 58-60 (improvement vs baseline ~66)
  - **2,434 greedy eval wins** (1,217/hr — 2.4× baseline rate)
  - Best single eval p5: **16.9** (huge improvement from baseline's 49)
  - Q-states: 216,239
- **Verdict:** Real progress (2.4× wins/hr, −7 pt p5) but plateaued mid-run.
- **Next:** Try curriculum=0.7 for 45 min to test if knob still has headroom.

#### 2026-05-16 13:13 — `overnight-2g` (4 hr)

- **Goal:** Long-soak from smoke-1h merged policy. The "let-it-cook" run.
- **Config:** `-j 32 durationMin=240 ghosts=2 endgameCurriculum=0.2` loaded from smoke-1h
- **Result:** 🟡 Solid baseline, plateaued.
  - 1,995 greedy eval wins across 4 hours (~500/hr)
  - p5 plateaued at ~66 (not actively decreasing)
- **Verdict:** Curriculum=0.2 is fully saturated. Try aggressive curriculum.

#### 2026-05-16 10:56 — `smoke-1h` (1 hr) — 🎯 **First greedy wins ever**

- **Goal:** 1-hour smoke test before overnight. Checking the new folder structure + script fixes.
- **Config:** `-j 32 durationMin=60 ghosts=2 endgameCurriculum=0.2` loaded from `20260516-101633-parallel`
- **Result:** 🟢 **Breakthrough run.** First time ever achieving non-zero greedy eval wins.
  - **40,247 training wins** (vs 1,896 prior)
  - **69 greedy eval wins** across 26 of 32 workers
  - 31 of 32 workers reached `minPelletsLeft ≤ 1`
  - p5: 91.6 → 75.3
  - Q-states merged: 175,611
- **Verdict:** Crossed the threshold. Greedy wins by chunk *accelerating* through the hour.
- **Next:** Overnight 8h.

#### 2026-05-16 10:16 — `parallel` (20 min) — first parallel test

- **Goal:** Smoke-test the new run-parallel.sh with 32 workers.
- **Config:** `-j 32 durationMin=20 endgameCurriculum=0.2` (note: `ghosts=2` by default — NOT directly comparable to the earlier 3-ghost ab tests)
- **Result:**
  - 4.55M episodes, **1,896 training wins**, 0 greedy wins
  - Best minPellets in an eval: 31
  - p5: 210 → 124 (40% reduction, monotonic descent across all 10 chunks)
  - Q-states merged: 116,805
- **Verdict:** Federated parallel training works. Curve still descending at the end of 20 min.
- **Next:** Continue to 1-hour smoke.

---

### 3-Ghost runs

#### 2026-05-16 ~09:51 — `ab-3ab` (60 min) — ⚠️ Combination *hurt*

- **Goal:** Test combining 3a + 3b.
- **Config:** Single-worker, `endgameCurriculum=0.2 endgameEps=0.4 endgameBucket=1` + ghosts=3
- **Result:** **Only 6 training wins** (vs 4,019 for 3a alone) — combining made things drastically worse.
- **Lesson:** `endgameEpsilon` forces 40% random actions in late-game, which destroys the policy the curriculum is teaching.
- **Action:** Do NOT enable both flags together. (Generalized into Findings #6.)

#### 2026-05-16 ~08:51 — `ab-3b` (60 min)

- **Goal:** Isolate Priority 3b (state-conditional ε floor).
- **Config:** Single-worker, `endgameEps=0.4 endgameBucket=1` + ghosts=3
- **Result:** **0 training wins**, p5=60.
- **Verdict:** 3b alone doesn't drive learning. Random thrashing in endgame ≠ learning.

#### 2026-05-16 ~07:51 — `ab-3a` (60 min) — 🎯 First training wins (single-worker)

- **Goal:** Isolate Priority 3a (endgame curriculum).
- **Config:** Single-worker, `endgameCurriculum=0.2` + ghosts=3
- **Result:** **4,019 training wins** (0.21% rate). Best eval p5: 35.
- **Verdict:** 3a is the real exploration knob. 3b is a distraction.
- **Note:** Current best 3-ghost policy lives at `bench-out/ab-3a-20260516/policy-latest.json`.

#### 2026-05-15 evening — `run1`/`run2`/`run3`/`run6` (sub-runs of initial overnight)

- **Goal:** First test of the 5 fixes from the "implement fixes" commit batch (commits `7260980` through `ac7c178`).
- **Config:** Single-threaded `run-overnight.sh` sequential runs. `ghosts=3` for run1/run2/run3/run6.
- **Result:** **0 training wins, 0 greedy wins**. Best minPelletsLeft 44 (run1), 61 elsewhere.
- **Verdict:** Initial fix batch wasn't enough on its own. Needed `1b53afb` (curriculum), `212e472` (state-conditional ε), and `6fa8952` (powerPelletsLeftBucket).
- **Note:** Archived under `bench-out/_archive/pre-ab-tests/`. Pre-curriculum, pre-powerPelletBucket.

#### Pre-2026-05-15 — Original 75-min run (the audit baseline)

- **Goal:** Understand why score plateaus at ~441 and win rate stays at 0%.
- **Config:** Single thread, `ghosts=3 epsDecay=0.9995 epsMin=0.05 maxSteps=400 winBonus=200` (original defaults).
- **Result:** 2.07M episodes, **0 wins**, mean score 441 (plateaued after first ~10k episodes).
- **Verdict triggered:** The full set of fixes described in the [Code Change Log](#code-change-log) section.

---

### 4-Ghost runs

#### 2026-05-15 evening — `run5-4ghosts` (25 min)

- **Goal:** Cross-ghost-count generalization test (4 ghosts vs a 3-ghost-trained policy).
- **Config:** Single-threaded, `ghosts=4` loading a 3-ghost policy → `numGhosts mismatch` warning.
- **Result:** 828k episodes, **0 wins**. Best minPelletsLeft 70.
- **Verdict:** Confirmed `numGhostsEncoded` mismatch matters — most observations were Q-table misses.
- **Note:** Archived under `bench-out/_archive/pre-ab-tests/run5-4ghosts/`. Not in active development.

---

### Mixed / Pre-fix runs

#### 2026-05-15 evening — `run4-2ghosts` (25 min) — first 2-ghost data point

- **Goal:** Cross-ghost-count generalization test (2 ghosts vs a 3-ghost-trained policy).
- **Config:** Single-threaded, `ghosts=2` loading a 3-ghost policy.
- **Result:** 817k episodes, **0 wins**. Best minPelletsLeft 69.
- **Verdict:** Like run5-4ghosts, cross-count loading was a miss-fest.
- **Note:** Archived. Predates `endgameCurriculum`. Not the same lineage as the current 2-ghost track.

---

## Open Questions

Things worth investigating, but not yet pursued.

1. **Will reward shaping (`winBonus=2500-5000`, `deathPenalty=-50`) break the current p5≈55 plateau?** Highest-leverage untested knob.
2. **Does federated Q-table averaging help or hurt vs each worker training in isolation?** We've always merged; never compared to a single-worker baseline at equivalent compute.
3. **Is there a single-worker test that reproduces the 2.5% eval win rate?** The federated training might be doing more work than the resulting policy reflects.
4. **What's the win-rate ceiling for the 2-ghost case?** We haven't pushed any single policy past 2.5% in eval. May be a fundamental limit of the state representation.
5. **How does the current policy transfer to 3 ghosts?** Loading a `numGhostsEncoded=2` policy with `ghosts=3` triggers a console warning but does load. Unknown if useful.

---

## Folder Layout Reference

```
bench-out/
├── <YYYYMMDD-HHMMSS>-<desc>/         # one top-level per experiment
│   ├── run1-baseline/                # ← run-overnight.sh sub-runs
│   │   ├── policy-latest.json
│   │   ├── episodes.csv
│   │   ├── evals.csv
│   │   └── summary.json
│   ├── run2-resume/  …  run6-overnight/
│   ├── worker-00/                    # ← run-parallel.sh sub-runs
│   │   ├── bench.log
│   │   ├── episodes.csv
│   │   ├── evals.csv
│   │   ├── policy-latest.json
│   │   └── summary.json
│   ├── worker-01/  …  worker-31/
│   ├── policy-merged.json            # Q-table averaged across workers
│   └── summary.txt                   # per-worker stats table
└── _archive/                         # pre-2026-05-16 layout (flat run*/ folders)
```

**CSV schemas:**
- `episodes.csv`: `episode,score,length,epsilon,qTableSize,stepsPerSec,pelletsLeft,termReason`
- `evals.csv`: `episode,avgScore,stdScore,avgLength,winRate,wins,minPelletsLeft,pl_p5,pl_p25,pl_p50,pl_p75,pl_p95`
- `summary.json`: includes `config` block, `trainingWins`/`trainingWinRate`, `qTableSize`, score stats.

**Key fields for analysis:**
- `wins` (eval) — raw greedy-eval win count. Use this, not `winRate`.
- `pl_p5` — 5th-percentile pelletsLeft across the 200 eval games. Tighter signal than `minPelletsLeft`.
- `trainingWins` — total wins during ε-greedy training. Leading indicator before greedy wins appear.
- `termReason` — `won` / `died` / `timeout`. Almost always `died`.

---

## Updating This File

After each significant run:
1. Update the matching [Current State](#current-state) sub-section (2-Ghost / 3-Ghost / 4-Ghost) if best policy / best p5 changed.
2. Append a new dated entry to the matching ghost-count section under [Test Runs](#test-runs) (reverse-chronological within section).
   - If you start a new ghost-count track, add a new sub-section header.
3. If a finding generalizes across ghost counts or to future agents, add to [Findings](#findings).
4. If a code change happened, add to [Code Change Log](#code-change-log) with the commit hash.
5. Move resolved items out of [Open Questions](#open-questions).

The point of this file: when picking up after a break, the answer to "what have we tried for N ghosts?" should be one section read, not a re-derivation from `git log` + `bench-out/`.
