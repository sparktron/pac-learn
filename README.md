# AI Pac-Man Lab

Browser-based Pac-Man + in-browser Q-learning training lab. No backend, no build-time dependencies beyond Node.

## Quick start

```bash
npm install
npm run dev
```

Open the local Vite dev URL (usually `http://localhost:5173`).

---

## Playing the game (Human mode)

1. Select **Human** in the Mode dropdown.
2. Use **arrow keys** to move Pac-Man.
3. Click **Reset** at any time to restart the current episode with the selected seed.

The canvas shows:
- Blue tiles — walls
- Yellow dots — regular pellets
- Orange circles — power pellets (eat one to make ghosts edible / blue)
- Red circles — ghosts (turn blue when edible)
- Yellow wedge — Pac-Man

Score, pellets remaining, and current step count are displayed below the canvas.

---

## Watching the AI play (AI controlled mode)

1. Select **AI controlled** in the Mode dropdown.
2. The current Q-table policy runs at ~120 ms/step.
3. If no policy has been trained or loaded yet, the agent acts randomly.

Switching to AI mode automatically stops any running training loop.

---

## Training workflow

1. **Configure** environment parameters in the right-hand panel (maze, ghost count, speeds, rewards, etc.).
2. **Set seed** — determines pellet layout and ghost/pac start positions for each episode.
3. Click **Start training** — launches a `requestAnimationFrame` training loop.
   - Adjust **steps/frame** and **turbo** at any time; the loop picks up changes immediately.
   - Adjust **renderEveryNSteps** to control how often the canvas refreshes during training (higher = faster throughput).
   - The green **TRAINING — episode N** badge in the header shows training is active.
4. Click **Pause** to stop the loop without resetting the Q-table or stats.
5. Click **Single step** to advance exactly one environment step (useful for debugging).
6. Click **Evaluate** to run 20 greedy-policy episodes and display avg score / length / win rate.
7. **Save policy** downloads the Q-table as JSON (`policy-<timestamp>.json`).
8. **Load policy** restores a previously saved JSON file.
9. **Reset Q** clears the Q-table and stats.

### Tips for faster learning

- Start with 1 ghost, Classic maze, default rewards.
- Set **steps/frame** to 50–200 and enable **turbo** for ~10× throughput.
- Watch the **Moving avg score** chart; it should trend upward after a few hundred episodes.
- Decay **epsilon** toward 0 via **epsilonDecay** ≈ 0.999 (default) or lower for faster exploitation.

---

## Environment parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numGhosts` | 1 | Number of ghosts (1–6) |
| `numPacmen` | 1 | Number of Pac-Man clones (extra clones move randomly) |
| `ghostSpeed` | 0.95 | Fractional tiles/step. 0.5 = moves every other step; 2 = 2 tiles/step |
| `pacmanSpeed` | 1.0 | Same scale as `ghostSpeed` |
| `pelletDensity` | 1.0 | Fraction of open cells that spawn a pellet |
| `enablePowerPellets` | true | Spawn 4 power pellets at maze corners |
| `powerPelletDuration` | 20 | Steps ghosts remain edible after power pellet |
| `captureRules` | tile | `tile` = same cell; `touch` = manhattan distance ≤ 1 |
| `maxEpisodeSteps` | 400 | Hard episode timeout |
| `illegalMoveMode` | stay | `stay` = ignore illegal key; `noop` = take random legal move |

### Reward shaping

| Key | Default | Notes |
|-----|---------|-------|
| `pelletReward` | 5 | Per pellet eaten |
| `deathPenalty` | -100 | Applied when captured by a non-edible ghost |
| `stepPenalty` | -0.1 | Per-step cost to discourage idling |
| `survivalReward` | 0.02 | Per-step bonus while alive |
| `ghostEatReward` | 30 | For eating an edible ghost |

---

## Build for production

```bash
npm run build
```

Output lands in `dist/`. Host on any static server (GitHub Pages, Netlify, Cloudflare Pages, etc.).

```bash
npm run preview   # local preview of the built dist
```

---

## Running tests

```bash
npm test
```

Three test suites: maze collision, observation determinism, Q-value update.

---

## Architecture

```
src/
  engine/       Core types (Direction, Vec2) and seeded PRNG
  env/          PacmanEnvironment (reset/step/observe) + observation encoding
  ghosts/       Ghost AI strategies: classic, heatmap, hybrid
  rl/           QLearningAgent + TrainingController
  render/       CanvasRenderer (walls, pellets, ghosts, Pac-Man, heatmap overlay)
  ui/           LineChart component
  mazes/        Built-in maze grid definitions
```

---

## Extending the project

### Add a ghost AI type
1. Add a new literal to `GhostAIType` in `src/ghosts/ghostAi.ts`.
2. Add a branch in `chooseGhostMove`.
3. The UI dropdown will pick it up automatically.

### Add a maze
1. Define a string grid in `src/mazes/mazes.ts` (1 = wall, 0 = open).
2. Call `parse(id, name, rows)` and add it to `MAZES`.
3. Select it via the Maze dropdown.

---

## Known limitations

- Q-table observation is compact but lossy (5×5 wall mask + nearest pellet direction + clamped ghost offsets). A neural DQN would generalise better.
- Extra Pac-Man clones (numPacmen > 1) move randomly and do not collect pellets — a cooperative multi-agent extension is scaffold-ready.
- Ghost edibility timer does not reset between episodes if Pause is used mid-episode (resets on the next `env.reset()` call).
