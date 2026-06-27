# Development Roadmap

Next steps after the 2026 deep-dive audit (Sections 1–9, all merged). This doc is
written to be **picked up cold** — a fresh session should be able to execute any
item below without prior chat context.

- Full finding history + per-section change logs: **`archive/DEEP_DIVE_2026-05-30.md`** (archived 2026-06-16 — historical record).
- Training-run history + baseline policies: **`test_history.md`**.
- The audit shipped via PRs #30–#38 (all CI-green, merged to `master`).

---

## ⚙️ Conventions for every item (read first)

1. **Branch off latest `master`, one focused PR per item.** `git fetch origin`
   first — this repo is worked in parallel and `master` moves.
2. **CI is the gate.** `.github/workflows/ci.yml` runs `npm run typecheck`
   (covers `src/` *and* `scripts/` via `tsconfig.scripts.json`) + `npm test` +
   `npm run build` on every PR. Don't self-merge red. If you can't run the
   toolchain locally, push and watch the PR check.
3. **Protect the training baseline.** Changes to the environment, the observation
   key, or the seeded RNG stream can silently invalidate the policies/results in
   `test_history.md`. The safe pattern used throughout the audit: **add new
   behavior behind a flag/param that defaults to today's behavior**, so the
   "off" state is byte-identical and CI only has to prove "nothing changed when
   off." If a change *must* alter the observation key, bump
   `OBSERVATION_KEY_VERSION` (env/observation.ts) — `load()` discards mismatched
   policies, which is correct.
4. **Pure logic → `src/`, not `scripts/`.** Scripts are typechecked but keep
   their IO/glue there; testable logic belongs in `src/` with a `*.test.ts`.
5. Keep finding IDs (`D3.11`, etc.) in commit messages + test names so the
   chain stays greppable against the deep-dive doc.

---

## A. Ready to implement (contained, CI-verifiable)

### A1 — Scatter/chase phase visualization (D3.11) · ✅ DONE (PR #40)
**What:** show whether ghosts are in chase or scatter phase in the UI.
**Why:** the phase already drives ghost targeting (`env.isScatterPhase()`), but
it's invisible to the user — hard to interpret ghost behavior without it.
**Where:** `src/render/canvasRenderer.ts` (a small HUD indicator or border tint),
or a status chip in `src/App.tsx` (the maze-stage `hud-chip` area, ~line 530).
**Approach:** read `env.isScatterPhase()` in the render/draw path; render a small
"CHASE"/"SCATTER" chip or tint. If drawing on canvas, fold a phase flag into the
render-skip hash (see `canvasRenderer.ts` `hash`) so it repaints on flip.
**Safety:** pure presentation — no env/RNG change.
**Verify:** a renderer test asserting it doesn't throw across both phases; manual
eyeball. (`canvasRenderer.test.ts` has a mock-ctx pattern to copy.)

### A2 — Per-ghost configurable personality (D3.11) · ✅ DONE (PR #41)
**What:** let each ghost's targeting personality (Blinky/Pinky/Inky/Clyde) be set
independently instead of being fixed by `id % 4`.
**Why:** enables difficulty tuning ("4 aggressive Blinkys") and experiments.
**Where:** `src/ghosts/ghostAi.ts` (`getChaseTarget` uses `ghost.id % 4`);
`GhostState` in `src/env/environment.ts` (add an optional `role`/`personality`
field); `src/App.tsx` (per-ghost selector UI).
**Approach:** add an optional `personality` to `GhostState`; `getChaseTarget`
uses `ghost.personality ?? (ghost.id % 4)`. Populate it in `reset()` from a new
`EnvParams` field (array or default).
**Safety:** default must reproduce `id % 4` exactly → baseline-safe.
**Verify:** extend `ghostAi.test.ts` (it already has Pinky/Inky/Clyde targeting
tests) to assert an overridden personality changes targeting.

### A3 — Vertical-tunnel support (D3.11 / L4) · ✅ DONE (PR #47)
**What:** allow mazes with top/bottom wraparound tunnels (currently x-only).
**Why:** `wrapPosition` (engine/types) and `bfsPelletDir`/ghost-AI all wrap x
only; a maze with a vertical tunnel silently doesn't wrap.
**Where:** `src/engine/types.ts` `wrapPosition` (add y-wrap, currently ignores
`height`); confirm all callers (`environment.ts`, `ghostAi.ts`,
`observation.ts`) pass through; add a maze that uses it in `src/mazes/mazes.ts`.
**Approach:** gate y-wrap so existing mazes are unaffected (e.g. only wrap y for
mazes that opt in, or only when the edge tile is open). **Caution:** changing
wrap for existing mazes alters movement → baseline impact. Prefer an opt-in maze
property so current mazes are untouched.
**Verify:** `mazes.test.ts` reachability + a movement test on the new maze.

### A4 — Nominal `Action` type (D1.5) · ✅ DONE (PR #44)
**What:** make the action space a nominal/branded type so direction-order bugs
(historically C3/M2) become compile errors instead of silent aliasing.
**Where:** `src/engine/types.ts` (define `Action`); thread through
`environment.ts`, `rl/*`, `App.tsx` action plumbing.
**Approach:** behavior-neutral refactor — no runtime change, just types. Do it in
one pass and lean on CI typecheck.
**Safety/Verify:** zero runtime change; green typecheck is the proof.

### A5 — `App.tsx` decomposition (refactor) · ✅ DONE (PRs #51–#56 + topbar slice 5)
**What:** break the ~1000-line `App.tsx` into hooks + presentational components.
**Why:** it's the biggest structural smell; everything else in the UI is small.
**Outcome:** `App.tsx` is now ~415 lines — a thin orchestrator that wires the
agent/trainer, the renderer/play-loop effects, and the handlers, then composes
the extracted pieces. Shipped as slices:
- RTL harness + App smoke tests (groundwork, #51) — RTL is now installed.
- `useGameEnv` hook — env + params + live-apply (slice 1, #52).
- `useTrainingLoop` hook — trainer start/stop, speed presets, Space toggle,
  structural-reset effect (slice 3, #53).
- `TelemetryPanel` / `EnvironmentPanel` / `ConfigurationPanel` components
  (slices 4a–4c, #54–#56).
- `TopBar` component — brand, status pill, key stats, action buttons (slice 5).
Each extraction was behavior-neutral and ships with a `*.test.tsx` (RTL).
**Verify:** `npm run typecheck` + `npm test` + `npm run lint` + `npm run build`
all green; manual smoke of training/reset/tab switching.

---

## B. Needs a live toolchain (do where `npm`/shell/eslint actually run)

These were deferred specifically because they can't be validated in a no-toolchain
sandbox — do them in a session where you can run commands iteratively.

### B1 — ESLint setup + triage (D9.2) · ✅ DONE (PR #45)
**What:** install + configure ESLint (typescript-eslint + react-hooks), add a
`lint` script, and triage the findings.
**Why:** `eslint-disable` directives exist in `App.tsx`/tests but **no linter is
installed** — they're dead no-ops today.
**Approach:** add deps + a flat config; run `npm run lint`; fix/triage iteratively
(this is the part that needs a live linter). Add a CI lint step **only after**
the tree is clean, so it can't fail the build with an un-triaged backlog.
**Verify:** `npm run lint` clean, then add to `ci.yml`.

### B2 — `.sh` strict mode (D8.5) · ✅ ALREADY DONE (verified 2026-06-27)
**What:** ensure `set -euo pipefail` on the orchestrator scripts.
**Finding:** the D8.5 premise was already stale — **all seven `scripts/*.sh`
already carry `set -euo pipefail`** (`hyperparam-sweep`, `run-overnight`,
`run-parallel`, `run-sweep`, `algorithm-compare`, plus the `short-learning-sweep`
and `test-parallel-merge` harnesses). The flags were added when each script was
written (e.g. `run-parallel.sh` in 9e1ddde), before this roadmap was authored.
**Verify (done):** `bash -n` clean on all four named targets, and
`scripts/test-parallel-merge.sh` (2 workers × 2 episodes through
`run-parallel.sh`) runs green under strict mode → merged policy with 119 states.
No code change required.

---

## C. Larger features (product/design decision)

### C1 — Maze editor + import/export (D2.6)
**What:** an in-UI editor to draw/edit mazes and import/export them as JSON, so
hand-designed mazes can be added without code.
**Where:** new component(s) in `src/`; `src/mazes/mazes.ts` already exports
`validateMaze()` — reuse it to validate edited/imported mazes before play.
**Approach:** biggest surface + real UX decisions (grid editing, palette,
persistence). Scope it explicitly before building; likely several PRs.
**Verify:** `validateMaze` on every edited/imported maze; reachability is the
key invariant (a maze with unreachable pellets is unwinnable).

---

## Quick-reference: file map

| Area | Path |
|------|------|
| Env + rewards + EnvParams | `src/env/environment.ts` |
| Observation / state key | `src/env/observation.ts` |
| Ghost AI | `src/ghosts/ghostAi.ts` |
| Tabular / linear agents, merge, presets | `src/rl/*` |
| Renderer | `src/render/canvasRenderer.ts` |
| UI | `src/App.tsx`, `src/uiHelpers.ts` |
| Bench / sweep / merge CLIs | `scripts/*` |
| Build / test / CI config | `tsconfig*.json`, `vitest.config.ts`, `vite.config.ts`, `.github/workflows/ci.yml` |

**Status:** A1–A5 + B1 + B2 are all shipped (B2 was already satisfied — see its
note). **Remaining:** C1 (maze editor — the one large product/design item).
