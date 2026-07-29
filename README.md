# 👾 AI Pac-Man Lab

Browser-based Pac-Man + in-browser Q-learning training lab. No backend, no build-time dependencies beyond Node.

Engineering decisions, failed experiments, and lessons learned are recorded in
[`ENGINEERING_JOURNAL.md`](ENGINEERING_JOURNAL.md). Detailed metrics remain in
[`test_history.md`](test_history.md), with current priorities in
[`ROADMAP.md`](ROADMAP.md).

## 📋 Recent updates

### ✅ Correctness review follow-up (2026-07-21)

- Centralized algorithm-specific hyperparameter defaults so GUI and headless runs agree
- Made policy loading validate compatibility before synchronizing the configured ghost count
- Resolved pellet collection and collisions after every tile at movement speeds above 1
- Recorded the full findings and evidence in [`CODE_REVIEW_2026-07-21.md`](CODE_REVIEW_2026-07-21.md)

### 🎨 UI Refactoring & Component Extraction (A5 initiative)
- Extracted reusable React components: **EnvironmentPanel**, **ConfigurationPanel**, **TelemetryPanel**
- Extracted custom hooks: **useGameEnv** (environment setup), **useTrainingLoop** (training state)
- Added React Testing Library infrastructure + App smoke tests for component reliability
- Improved code maintainability and reusability across the UI

### 🛡️ Type Safety & Code Quality
- Promoted `Action` to a nominal (branded) type for compile-time safety (D1.5)
- Added ESLint with flat config + automated CI lint checks (D9.2)
- Proper validation in `TrainingController.evaluate()` — now throws on invalid episode counts instead of returning NaN

### 🎮 Gameplay & Rendering Enhancements

- **Vertical tunnel support** — mazes can now opt-in to tunnel navigation top-to-bottom
- **Aspect ratio tile sizing** — canvas rendering now scales tiles to fit both container axes for better responsive design
- **Action-conditioned linear features + target network** (D8/D9) — the
  2026-07-26 five-seed confirmation averaged 27.55% eval wins
- **Agent-specific greedy evaluation** — tabular exact ties steer toward
  pellets; linear exact ties stay random because a controlled A/B measured a
  6.1-point regression from deterministic tie-breaking
- **Tunnel-correct observations** — the local wall mask now agrees with legal
  movement at tunnel mouths
- **Held-out evaluation panels and resilient soak reporting** — headless runs
  can report per-panel results, tie rates, and recover summaries after a kill
- **Far-pellet direction fallback** — the radius-12 BFS now continues only
  when necessary, removing reachable endgame sentinel states; matched
  five-seed/four-panel runs improved mean greedy wins from 25.54% to 35.17%
- **T2 reward/discount defaults** — linear γ=0.997, death penalty −50, and a
  10× late-pellet multiplier improved a matched five-seed mean from 33.25% to
  37.17%
- **T1 n-step return screen** — n=3, 5, and 10 did not beat the promoted
  one-step baseline at equal compute, so `nStep=1` remains the default
- **T3 potential-based shaping screen** — scales 25, 100, and 250 did not
  improve mean greedy wins, so shaping remains disabled by default
- **T5 coarse-position screen** — a 3×3 tabular key reduced the pellet tail
  but produced no greedy wins and increased table size 72%, so it remains off

The far-pellet fallback leaves the existing distance-feature normalization
capped at 13, so its gain comes from restoring the missing action direction
rather than rescaling the linear value feature. The next research item is
a separate full-grid CNN Double-DQN research track; the promoted linear agent
remains the baseline. See [the roadmap](ROADMAP.md) for its architecture and
acceptance gates.

### 📖 Documentation Improvements
- Recorded linear α sweep findings — α is not the main learning lever (Finding #10)
- Added linear vs. tabular learning comparison analysis
- Archived deep-dive audit logs as historical records

## 🚀 Quick start

```bash
npm install
npm run dev
```

Open the local Vite dev URL (usually `http://localhost:5173`).

---

## 🎮 Playing the game (Human mode)

1. Select **Human** in the Mode dropdown.
2. Use **arrow keys** to move Pac-Man.
3. Click **Reset** at any time to restart the current episode with the selected seed.

### 📋 Game rules

- 🟡 **Pellets** — eat all pellets to win the level and earn a win bonus.
- ⭐ **Power pellets** — larger, pulsing orange orbs in the maze corners. Eating one makes all ghosts edible (they turn blue) for a limited time.
- 👻 **Ghosts** — each ghost has a distinct color (red, pink, blue, orange, purple, green). Contact with a non-edible ghost kills Pac-Man and ends the game.
- 😋 **Eating ghosts** — while ghosts are edible, Pac-Man can eat them for bonus points. A combo multiplier rewards eating multiple ghosts per power pellet (1x, 2x, 3x, 4x).
- 📊 **Scoring** — points come from pellets (+5), power pellets (+20), eating ghosts (+30 x combo), and clearing all pellets (win bonus +1000). All values are configurable.

### 🎨 Canvas legend

| Symbol | Meaning |
|--------|---------|
| Colored outlines | Walls (color varies per maze) |
| 🟡 Small yellow dots | Regular pellets |
| ⭐ Pulsing orange orbs | Power pellets |
| 👻 Colored ghost shapes with eyes | Ghosts (flash white near timer expiry) |
| 💛 Yellow wedge | Pac-Man |

Score, pellets remaining, and current step count are displayed below the canvas.

---

## 🗺️ Mazes

### 🖼️ Static mazes

Four hand-designed mazes are included:

| Maze | Size | Wall color |
|------|------|------------|
| Classic | 28×31 | 🔵 Blue |
| Arena | 21×17 | 🟣 Purple |
| Corridors | 17×13 | 🟢 Green |
| Vertical Loop | 13×13 | 🟦 Teal |

### ✨ Procedural mazes

Five procedurally generated mazes are available out of the box (`Procedural #100` through `#104`). Each is built with a recursive-backtracker algorithm with extra loop-opening passes to create Pac-Man-friendly layouts, plus a central ghost house. Wall colors are randomly assigned per seed.

The generator can produce unlimited unique mazes — see `generateMaze(seed)` in `src/mazes/mazes.ts`.

---

## 🤖 Watching the AI play (AI controlled mode)

1. Select **AI controlled** in the Mode dropdown.
2. The current Q-table policy runs at ~120 ms/step.
3. If no policy has been trained or loaded yet, the agent acts randomly.

Switching to AI mode automatically stops any running training loop.

---

## 🎓 Training workflow

1. **⚙️ Configure** environment parameters in the right-hand panel (maze, ghost count, speeds, rewards, etc.).
2. **🌱 Set seed** — determines pellet layout and ghost/pac start positions for each episode.
3. Click **▶️ Training** — launches a `requestAnimationFrame` training loop.
   - Adjust **steps/frame** at any time; the loop picks up changes immediately.
   - Adjust **renderEveryNSteps** to control how often the canvas refreshes during training (higher = faster throughput).
   - The green **● TRAINING — episode N** badge in the header shows training is active.
4. Click **⏸️ Pause** to stop the loop without resetting the Q-table or stats.
5. **💾 Save policy** downloads the current policy as JSON (`policy-<timestamp>.json`).
6. **📂 Load** restores a compatible policy and synchronizes its encoded ghost count.
7. **🗑️ Reset Q** clears the policy and stats.

The GUI and headless bench share algorithm-specific hyperparameter defaults
from `src/rl/hyperDefaults.ts`, so equivalent runs start from the same tabular
or linear baseline.

### 🧪 Reproducible learning smoke

Run the deterministic T7 regression check locally:

```bash
npm run test:learning-smoke
```

It trains the promoted linear baseline twice for 2,000 episodes, verifies that
the learning outputs match, then checks conservative four-panel win-rate and
pellet-tail floors. CI runs the same smoke on every change.

### 💡 Tips for faster learning

- Start with 2 ghosts, Classic maze, default rewards.
- Set **steps/frame** to 50–200 and raise **renderEveryN** for higher throughput.
- Watch the **Moving avg score** chart; it should trend upward after a few hundred episodes.
- The tabular default uses **epsilonDecay** = 0.999997; the linear default uses
  0.9995 with a lower exploration floor.

### 🧵 Parallel training + merge smoke test

Run a short two-worker local smoke test to verify independent training outputs can be merged:

```bash
npm run test:parallel-merge
```

For real runs, launch multiple workers and merge their `policy-latest.json` outputs into one policy:

```bash
./scripts/run-parallel.sh -j 8 durationMin=60 desc=overnight-parallel
```

Outputs are written under `bench-out/<timestamp>-<desc>/`, with one `worker-*` folder per worker and a final `policy-merged.json` at the top level.

Your 4-worker 420-minute run is a good sign: training wins were about 1.2% per worker, and 87% of merged states were shared by at least two workers. The next likely bottleneck is greedy-policy quality/generalization, not raw throughput. Run a short sweep before another overnight job:

```bash
npm run sweep:short -- durationMin=20 desc=short-learning-sweep
```

This compares the current baseline against lower exploration floor, lighter endgame curriculum, lighter step penalty, and lighter reverse penalty variants. Sort `bench-out/<timestamp>-short-learning-sweep/report.tsv` by `bestEvalWinRate`, then rerun the best one with `./scripts/run-parallel.sh -j 4 durationMin=420 ...`.

---

## ⚙️ Environment parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numGhosts` | 2 | 👻 Number of ghosts (1–6) |
| `numPacmen` | 1 | 💛 Number of Pac-Man clones (extra clones move randomly) |
| `ghostSpeed` | 0.95 | ⚡ Fractional tiles/step. 0.5 = moves every other step; 2 = 2 tiles/step |
| `pacmanSpeed` | 1.0 | ⚡ Same scale as `ghostSpeed` |
| `pelletDensity` | 1.0 | 🟡 Fraction of open cells that spawn a pellet |
| `enablePowerPellets` | true | ⭐ Spawn power pellets at maze-defined corner positions |
| `powerPelletDuration` | 20 | ⏱️ Steps ghosts remain edible after power pellet |
| `captureRules` | tile | 🎯 `tile` = same cell; `touch` = manhattan distance ≤ 1 |
| `maxEpisodeSteps` | 1000 | ⏰ Hard episode timeout |
| `illegalMoveMode` | stay | 🚫 `stay` = ignore illegal key; `noop` = take random legal move |

### 💰 Reward shaping

| Key | Default | Notes |
|-----|---------|-------|
| `pelletReward` | 5 | 🟡 Per pellet eaten |
| `powerPelletReward` | 20 | ⭐ Per power pellet eaten |
| `deathPenalty` | -50 | 💀 Captured by a non-edible ghost |
| `stepPenalty` | -0.1 | ⏱️ Per-step cost to discourage idling |
| `survivalReward` | 0 | 💚 Per-step bonus while alive |
| `ghostEatReward` | 30 | 😋 Base reward for eating an edible ghost (multiplied by combo) |
| `winBonus` | 1000 | 🏆 Bonus for clearing all pellets |
| `pelletEscalationMax` | 10 | 📈 Final-pellet reward multiplier (ramps from 1×) |
| `potentialShapingScale` | 0 | 🧭 Optional policy-invariant pellet-progress potential (off by default) |
| `potentialShapingGamma` | 0.997 | 🧭 Discount in the potential equation; match learner γ when enabled |

---

## 📦 Build for production

```bash
npm run build
```

Output lands in `dist/`. Host on any static server (GitHub Pages, Netlify, Cloudflare Pages, etc.).

```bash
npm run preview   # local preview of the built dist
```

---

## ✅ Running tests

```bash
npm test
```

Three test suites: maze collision, observation determinism, Q-value update.

---

## 🏗️ Architecture

```
src/
  engine/       Core types (Direction, Vec2) and seeded PRNG
  env/          PacmanEnvironment (reset/step/observe) + observation encoding
  ghosts/       Ghost AI strategies: classic, heatmap, hybrid
  rl/           QLearningAgent + TrainingController
  render/       CanvasRenderer (walls, pellets, ghosts, Pac-Man, heatmap overlay)
  ui/           LineChart component
  mazes/        Static maze definitions + procedural maze generator
```

---

## 🔧 Extending the project

### 👻 Add a ghost AI type
1. Add a new literal to `GhostAIType` in `src/ghosts/ghostAi.ts`.
2. Add a branch in `chooseGhostMove`.
3. The UI dropdown will pick it up automatically.

### 🗺️ Add a maze
1. Define a string grid in `src/mazes/mazes.ts` (1 = wall, 0 = open).
2. Call `parse(id, name, rows, wallColor)` and add it to `STATIC_MAZES`.
3. Or use `generateMaze(seed, width, height, wallColor)` for procedural mazes.
4. Select it via the Maze dropdown.

---

## ⚠️ Known limitations

- 🧠 Q-table observation is compact but lossy (5×5 wall mask + nearest pellet direction + clamped ghost offsets). A neural DQN would generalise better.
- 👥 Extra Pac-Man clones (numPacmen > 1) move randomly and do not collect pellets — a cooperative multi-agent extension is scaffold-ready.
- ⏱️ Ghost edibility timer does not reset between episodes if Pause is used mid-episode (resets on the next `env.reset()` call).

---

## 📜 Changelog

### Latest (2026-06-21)
- **#29**: Fix `evaluate()` episode count validation — throws on non-positive or non-integer counts instead of returning NaN
- **#56**: Extract `ConfigurationPanel` component — consolidates training config UI
- **#55**: Extract `EnvironmentPanel` component — consolidates environment parameter UI
- **#54**: Extract `TelemetryPanel` component — consolidates telemetry/stats display
- **#53**: Extract `useTrainingLoop` hook — centralizes training loop state management
- **#52**: Extract `useGameEnv` hook — wraps environment setup and live parameter application

### May 2026 (A5 Component Initiative)
- **#51**: Add React Testing Library infrastructure + App smoke tests
- **#50**: Document linear α sweep findings (Finding #10)
- **#49**: Document linear vs. tabular learning comparison
- **#48**: Add continuous distance features for linear agent (D5.9)
- **#47**: Add vertical tunnel support for mazes (A3)
- **#46**: Implement aspect ratio tile sizing for responsive rendering (D6.7)
- **#45**: Set up ESLint with flat config + CI lint checks (D9.2)
- **#44**: Promote `Action` type to nominal type for type safety (D1.5)

### April–May 2026
- **#43**: Archive deep-dive audit logs (historical record)
- **#42**: Audit and test cross-over collision detection (D4.2)
- **#41**: Add per-ghost configurable personality (A2, D3.11)
- **#40**: Add scatter/chase phase indicator HUD (A1, D3.11)
- **#39**: Add ROADMAP.md for next-steps planning
- **#38**: Implement Cruise Elroy (late-game Blinky speed boost, default off)
- **#37**: Add shared wrap in `bfsPelletDir` + configurable phase durations (D4.7, D4.8)
- **#36**: Extract shared reward presets module — dedup App/bench/test code (D5.11)
- **#35**: Use `node:` specifiers in vite.config (D9.6)
- **#34**: Extract bench metrics to tested src/ module (D8.4)
- **#33**: Add algorithm selector (tabular/linear), URL revoke, safe localStorage (D7.8–D7.10)
- **#32**: Fix stale tile on maze change; scale pellets; face pac mouth (D6.10–D6.13)
- **#31**: Normalize linear features, shared key decode, peekMaxQ (D5.10, D5.12–D5.14)
- **#30**: Fix double death penalty in collision loop (D4.9)
