# AI Pac-Man Lab

Browser-based Pac-Man + in-browser RL training lab (no backend required).

## Run locally

```bash
npm install
npm run dev
```

Open the local Vite URL.

## Build

```bash
npm run build
```

Output is in `dist/`.

## Static hosting

Host the `dist/` folder on any static host (GitHub Pages, Netlify, Cloudflare Pages, your own domain like `dylansparks.com`).

## Features

- Playable Pac-Man clone via arrow keys.
- RL environment (`reset`, `step`, `getLegalActions`, `setParams`) decoupled from rendering.
- Tabular Q-learning baseline (epsilon-greedy + decay).
- Deterministic seed-based simulation.
- Multiple mazes (3 built in).
- Configurable ghost count and AI type (`classic`, `heatmap`, `hybrid`).
- Add multiple Pac-Man clones (random behavior baseline for extra clones).
- Levers for reward shaping, episode length, training speed, and physics-ish settings.
- Save/load Q-table policy JSON.
- Live charts for score/length/epsilon + ghost heatmap overlay.

## Training workflow

1. Set seed + environment levers.
2. Click **Start training** or **Single step**.
3. Monitor score/length/epsilon charts.
4. Click **Evaluate** for greedy policy evaluation.
5. **Save policy** to download JSON.
6. **Load policy** to restore JSON.

## Architecture

- `src/env`: Environment and observation encoding.
- `src/engine`: Core directional types and deterministic PRNG.
- `src/ghosts`: Ghost strategy implementations.
- `src/rl`: Q-learning and training control.
- `src/render`: Canvas rendering.
- `src/ui`: lightweight charting components.
- `src/mazes`: Built-in maze definitions.

## Add a new ghost type

1. Extend `GhostAIType` in `src/ghosts/ghostAi.ts`.
2. Add behavior logic in `chooseGhostMove`.
3. Add UI control for selecting type per ghost.

## Add a new maze

1. Add a string grid in `src/mazes/mazes.ts`.
2. Add it to `MAZES` with id/name.
3. Select it in the UI Maze dropdown.

## Limitations / next steps

- Tabular observation encoding is compact but lossy.
- Extra Pac-Man clones currently use random baseline policy.
- DQN interface is scaffold-ready but not implemented yet.
- Future: self-play, hall-of-fame policy league, richer multi-agent reward partitioning.
