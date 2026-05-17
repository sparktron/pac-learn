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

| | |
|---|---|
| **Best policy on disk** | `bench-out/20260516-224305-2g-curric07/policy-merged.json` |
| **Trained for** | 2-ghost Pac-Man (`numGhostsEncoded=2`) |
| **Q-table size** | ~218k states (merged across 32 workers) |
| **Eval `p5` (best chunk avg)** | **54.4 pellets remaining** (out of ~218) |
| **Best single eval game** | 12.8 pellets remaining (worker-01 of curric07 run) |
| **Best eval win rate** | 2.5% (5/200 in a single eval pass) |
| **Status** | Curriculum knob saturating — reward shaping is the next likely lever |

**Observation key version:** v7 (`v7:wallMask:pelletDir:gc0:gc1:lastAction:pelletsBucket:powerBucket`)

**Active reward preset (default):**
- `pelletReward=5` × pellet-escalation (1×→6× as pellets clear)
- `powerPelletReward=20` × pellet-escalation
- `deathPenalty=-100`
- `stepPenalty=-0.1`
- `survivalReward=0`
- `ghostEatReward=30` (×combo)
- `winBonus=1000`

**Active hyperparameter defaults (overnight-bench.ts):**
- `alpha=0.2  gamma=0.99`
- `eps=0.5  epsDecay=0.99999  epsMin=0.15`
- `optimisticInit=50` (Q-init)
- `endgameEpsilon=0` (off by default — only enable via CLI for 3b ablations)
- `endgameBucketThreshold=1` (only used if endgameEpsilon set)
- `evalEpisodes=200  evalEvery=2000  maxSteps=800`

---

## Configuration Knobs Reference

| Knob | Where | Default | Range tested | What it does |
|---|---|---|---|---|
| `endgameCurriculum` | CLI | 0 | 0, 0.2, 0.5, 0.7 | P(start episode in 10-25% pellets) |
| `endgameEpsilon` | CLI | 0 | 0, 0.4 | ε floor when in late-game bucket |
| `endgameBucket` | CLI | 1 | 1 | bucket ≤ this triggers endgameEps |
| `winBonus` | env preset | 1000 | 200, 1000 | reward for clearing all pellets |
| `optimisticInit` | hyper | 50 | -1, 50 | initial Q-value for unseen state-actions |
| `maxSteps` | CLI | 800 | 400, 800 | episode timeout |
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

---

## Test Runs

Reverse-chronological. Each entry: config, top-level stats, what it told us.

### 2026-05-16 22:43 — `2g-curric07` (45 min)

- **Goal:** Test if curriculum knob still has headroom past 0.5.
- **Config:** `-j 32 durationMin=45 ghosts=2 endgameCurriculum=0.7` loaded from `20260516-193418-2g-aggressive/policy-merged.json`
- **Result:** ✅ p5 dropped to **54.37** (chunk 10), meeting ≤55 threshold.
  - 1,225 greedy eval wins (**1,633/hr** — new high)
  - Best single eval: 5/200 wins, p5=12.8 (worker-01)
  - Q-states merged: 217,881
  - Wins/chunk declining 210 → 89 (interpretation: harder eval distribution under new training, not policy degradation)
- **Verdict:** Green on p5. Diminishing-but-real returns from curriculum (0.2→0.5→0.7 yielded −9, −4 pts each step).
- **Next:** 4-hour soak at curriculum=0.7 before pivoting to reward shaping.

### 2026-05-16 19:34 — `2g-aggressive` (2 hr)

- **Goal:** Test if aggressive curriculum (0.5) breaks the 4h-baseline plateau at p5≈66.
- **Config:** `-j 32 durationMin=120 ghosts=2 endgameCurriculum=0.5` loaded from overnight-2g
- **Result:** 🟡 Yellow.
  - p5: 68 → 58 by chunk 2, plateau at 58-60 (improvement vs baseline ~66)
  - **2,434 greedy eval wins** (1,217/hr — 2.4× baseline rate)
  - Best single eval p5: **16.9** (huge improvement from baseline's 49)
  - Q-states: 216,239
- **Verdict:** Real progress (2.4× wins/hr, −7 pt p5) but plateaued mid-run.
- **Next:** Try curriculum=0.7 for 45 min to test if knob still has headroom.

### 2026-05-16 13:13 — `overnight-2g` (4 hr)

- **Goal:** Long-soak from smoke-1h merged policy. The "let-it-cook" run.
- **Config:** `-j 32 durationMin=240 ghosts=2 endgameCurriculum=0.2` loaded from smoke-1h
- **Result:** 🟡 Solid baseline, plateaued.
  - 1,995 greedy eval wins across 4 hours (~500/hr)
  - p5 plateaued at ~66 (not actively decreasing)
- **Verdict:** Curriculum=0.2 is fully saturated. Try aggressive curriculum.

### 2026-05-16 10:56 — `smoke-1h` (1 hr) — 🎯 **First greedy wins**

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

### 2026-05-16 10:16 — `parallel` (20 min)

- **Goal:** Smoke-test the new run-parallel.sh with 32 workers.
- **Config:** `-j 32 durationMin=20 endgameCurriculum=0.2` (note: `ghosts=2` by default — NOT comparable to the earlier ab-3a which used ghosts=3)
- **Result:**
  - 4.55M episodes, **1,896 training wins**, 0 greedy wins
  - Best minPellets in an eval: 31
  - p5: 210 → 124 (40% reduction, monotonic descent across all 10 chunks)
  - Q-states merged: 116,805
- **Verdict:** Federated parallel training works. Curve still descending at the end of 20 min.
- **Next:** Continue to 1-hour smoke.

### 2026-05-16 ~09:51 — `ab-3ab` (60 min) — ⚠️ Combination *hurt*

- **Goal:** Test combining 3a + 3b.
- **Config:** Single-worker, `endgameCurriculum=0.2 endgameEps=0.4 endgameBucket=1` + ghosts=3
- **Result:** **Only 6 training wins** (vs 4,019 for 3a alone) — combining made things drastically worse.
- **Lesson:** `endgameEpsilon` forces 40% random actions in late-game, which destroys the policy the curriculum is teaching.
- **Action:** Do NOT enable both flags together.

### 2026-05-16 ~08:51 — `ab-3b` (60 min)

- **Goal:** Isolate Priority 3b (state-conditional ε floor).
- **Config:** Single-worker, `endgameEps=0.4 endgameBucket=1` + ghosts=3
- **Result:** **0 training wins**, p5=60.
- **Verdict:** 3b alone doesn't drive learning. Random thrashing in endgame ≠ learning.

### 2026-05-16 ~07:51 — `ab-3a` (60 min) — 🎯 First training wins (single-worker)

- **Goal:** Isolate Priority 3a (endgame curriculum).
- **Config:** Single-worker, `endgameCurriculum=0.2` + ghosts=3
- **Result:** **4,019 training wins** (0.21% rate). Best eval p5: 35.
- **Verdict:** 3a is the real exploration knob. 3b is a distraction.

### 2026-05-15 evening — `run1`–`run6` (initial overnight)

- **Goal:** First test of the 5 fixes from the "implement fixes" commit batch (commits `7260980` through `ac7c178`).
- **Config:** 6 different runs via `run-overnight.sh`, sequential, single-threaded.
  - Note: This bench-out got muddled because the script auto-detected stale seed policies across executions. Now archived under `bench-out/_archive/`.
- **Result:** **0 training wins, 0 greedy wins** across 1.13M total episodes.
- **Best minPelletsLeft (eval):** 44 in `run1`, 61 in others. Agent reliably collected ~75% of pellets, then died.
- **Verdict:** Initial fixes weren't enough on their own. Needed the further additions in `1b53afb`, `212e472`, `6fa8952`.
- **Note:** This was *before* `endgameCurriculum` and `powerPelletsLeftBucket` existed.

### Pre-2026-05-15 — Original 75-min run (the audit baseline)

- **Goal:** Understand why score plateaus at ~441 and win rate stays at 0%.
- **Config:** Single thread, `ghosts=3 epsDecay=0.9995 epsMin=0.05 maxSteps=400 winBonus=200` (original defaults).
- **Result:** 2.07M episodes, **0 wins**, mean score 441 (plateaued after first ~10k episodes).
- **Verdict triggered:** The full set of fixes described in the [Code Change Log](#code-change-log) section.

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
1. Update [Current State](#current-state) if best policy / best p5 changed.
2. Add a new dated section to [Test Runs](#test-runs) (reverse-chronological).
3. If a finding generalizes, add to [Findings](#findings).
4. If a code change happened, add to [Code Change Log](#code-change-log) with the commit hash.
5. Move resolved items out of [Open Questions](#open-questions).

The point of this file: when picking up after a break, the answer to "what have we tried?" should be one read of this doc, not a re-derivation from `git log` + `bench-out/`.
