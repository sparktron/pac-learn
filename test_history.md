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

**Observation key version:** v9 (`v9:wallMask:pelletDir:gc0:gh0:gc1:gh1:lastAction:pelletsBucket:powerBucket`)
— v8 added per-ghost heading codes (`gh0`/`gh1`); v9 realigned `pelletDir` to the
DIRECTIONS action order. ⚠️ **All "best policy on disk" entries below are v7/v8 and
no longer load** (`load()` discards on key-version mismatch) — they must be
retrained before they can be evaluated or resumed.

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
| **Latest run** | `bench-out/20260726-031523-linear-vs-tabular/linear` |
| **Training wins** | 18,048 / 174,614 episodes (**10.34%**) |
| **Mean eval win rate** | **27.7%** across 332 × 200-game checkpoints |
| **Last-30 eval win rate** | **29.9%** |
| **Final eval** | 61/200 wins (**30.5%**), avgScore 3956.85, p5=0 |
| **Status** | D8/D9 gain confirmed on seed 7; multi-seed confirmation is next |

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
- `pelletReward=5` × pellet-escalation (1×→6× as pellets clear)
- `powerPelletReward=20` × pellet-escalation
- `deathPenalty=-100`
- `stepPenalty=-0.1`
- `survivalReward=0`
- `ghostEatReward=30` (×combo)
- `winBonus=1000`

**Active hyperparameter defaults (shared by GUI and overnight bench):**
- Tabular: `alpha=0.1  gamma=0.99`
- Tabular exploration: `eps=0.5  epsDecay=0.999997  epsMin=0.20`
- Tabular endgame: `endgameEpsilon=0.25  endgameBucketThreshold=1`
- Linear: `alpha=0.02  gamma=0.99`
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

11. **Greedy/eval tie-breaking matters — `random` ties throw away policy quality (roadmap T4).** `QLearningAgent.act()` historically broke ties between equal-max Q-values *randomly*. Because optimistic init leaves unvisited slots at 50, aliased/under-trained states have many ties, so under ε=0 eval the greedy policy partly degrades to a random walk. A deterministic **`pellet` tie-break** (steer a tied choice toward `nearestPelletDir`, else most-visited) lifts greedy **avgScore +44%** (799.7 → 1152.1) on the *same* Q-table and eval seeds, no retraining (2026-06-27; fresh 3-min v9 single-worker policy, 95k states, 200 eval games; `visits` mode was ~neutral at 773.9). The tie-break is a flag on `act()`/`evaluate()` (default `random` → baseline-safe). **Next:** make `pellet` the eval default and re-measure the real federated policies once retrained to v9; possible further lift from T4(a) (bootstrap unseen next-states from 0, not optimisticInit).

12. **Action-conditioned features made linear Q-learning the leading 2-ghost agent.** D8 replaced the structurally inadequate `w_a·f(s)` state-only representation with shared action-conditioned features `w·f(s,a)`. D9 added a 2,000-update target network to stabilize bootstrap targets. In the corrected 2026-07-26 eight-minute seed-7 comparison, linear averaged **27.7% eval wins** across 332 checkpoints (std 5.7%, min 3.5%, max 37.0%, last-30 mean 29.9%) and recorded **18,048/174,614 training wins (10.34%)**. Tabular recorded 0 eval wins and 1,023/309,916 training wins (0.33%) under the same environment/reward configuration. Against D9's reported 23.0% checkpoint mean and 25.4% last-30 mean, this is +4.7 and +4.5 percentage points respectively. This is strong single-seed evidence, not yet a multi-seed confidence interval.

---

## Test Runs

Organized by ghost count, reverse-chronological within each section. Each entry:
config, top-level stats, what it told us.

**Quick index:**
- [2-Ghost runs](#2-ghost-runs) — active tabular/linear comparison track
- [3-Ghost runs](#3-ghost-runs) — 5 runs, paused at 0 greedy wins
- [4-Ghost runs](#4-ghost-runs) — 1 stale run
- [Mixed / Pre-fix](#mixed--pre-fix-runs) — pre-2026-05-16 layout, archived

---

### 2-Ghost runs

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
