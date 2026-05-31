# AGENTS.md

Guidance for coding agents working in this repo.

## Tuning results must flow back into the defaults

When a tuning run — an overnight bench, a sweep (`npm run sweep:short`,
`scripts/run-sweep.sh`, `scripts/hyperparam-sweep.sh`), or an
`scripts/algorithm-compare.sh` comparison — produces **better learning** (higher
win rate, higher eval score, faster convergence), **update the defaults to match
that winning configuration.** Don't leave a better config living only in a run
log or a one-off CLI invocation.

When you update a default, change it in **all** the places defaults live, so the
in-app trainer and headless runs stay consistent:

- **GUI / in-app trainer:** `baseHyper` and the initial reward preset in `src/App.tsx`.
- **Headless bench:** the argument defaults and `PRESETS` in `scripts/overnight-bench.ts`.
- **Environment:** `defaultParams` in `src/env/environment.ts` (e.g. `maxEpisodeSteps`, reward shaping).

Keep the GUI defaults and the bench defaults **in sync** — "what you train in the
GUI" should match "what trains overnight." If they must diverge, say why in a
comment next to the value (the bench comments already document the empirical
basis for `alpha`, `winBonus`, `endgameEpsilon`, `endgameCurriculum`, etc.).

Briefly note the evidence (which run / what metric improved) in the commit
message or a code comment so the next agent knows the default is grounded, not a
guess.
