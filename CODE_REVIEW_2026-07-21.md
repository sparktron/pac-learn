# Full Code Review — 2026-07-21

Scope: the full application and training stack, including the environment,
observations, tabular and linear agents, controller, React hooks/components,
renderer, maze generation, policy merge logic, benchmark scripts, tests, and
the three local commits ahead of `origin/master` at review time.

## Executive summary

The repository builds cleanly and its 260 tests pass, but the review found
three high-impact correctness defects:

1. Linear-agent defaults differ between the GUI and headless bench, making
   supposedly equivalent training runs incomparable.
2. Loading a policy trained with a different ghost count changes the UI to the
   policy's count after the agent has already discarded the policy, leaving an
   empty agent while appearing successful.
3. Speeds above one tile per step process pellets and collisions only at the
   final position, allowing entities to skip interactions on intermediate
   tiles.

The review also found incomplete tunnel encoding, destructive evaluation state
restoration, misleading second-stage epsilon decay semantics, insufficient
policy/merge validation, benchmark failure handling gaps, and stale operator
documentation.

## High-priority findings

### H1. Linear GUI and benchmark defaults are different

**Files:** `src/App.tsx`, `scripts/overnight-bench.ts`

The GUI uses `alpha=0.02`, `epsilon=0.3`, `epsilonDecay=0.9995`,
`epsilonMin=0.05`, no endgame epsilon, and `targetSyncSteps=2000`. The headless
bench uses `alpha=0.01`, `epsilon=0.5`, `epsilonDecay=0.999997`,
`epsilonMin=0.20`, `endgameEpsilon=0.25`, and `targetSyncSteps=2000` for linear
runs.

This directly violates the repository invariant that GUI and headless defaults
stay synchronized. Headless linear runs inherit the tabular exploration regime
that the GUI comments say destabilizes linear TD, so GUI and overnight results
cannot be compared reliably.

**Recommended fix:** define shared tabular and linear default objects in one
module, import them from both call sites, and add a parity test.

### H2. Cross-ghost-count policy loading silently leaves an empty agent

**Files:** `src/rl/qlearning.ts`, `src/rl/linearQlearning.ts`, `src/App.tsx`

`QLearningAgent.load()` records `loadedNumGhosts`, detects the mismatch, clears
the Q-table, and returns. The UI then changes `params.numGhosts` to the retained
loaded value but never retries the discarded load. The environment therefore
switches to the right ghost count while the Q-table remains empty.

The linear path behaves differently: mismatch handling calls `reset()`, which
clears `loadedNumGhosts`, so it neither loads the policy nor synchronizes the UI.

**Recommended fix:** inspect `numGhostsEncoded` before calling `load()`, load
against that count, then synchronize the environment. Return an explicit load
result rather than inferring success from mutable fields.

### H3. Multi-tile speeds skip game interactions

**File:** `src/env/environment.ts`

Pac-Man and ghosts may move multiple tiles inside their movement loops, but
pellets and collisions are processed only after all iterations finish. At
`pacmanSpeed=2` or `ghostSpeed=2`, intermediate pellets are skipped and either
entity can pass through the other. Cross-over detection compares only the
start and final positions.

This contradicts the documented meaning of speeds above one and affects values
available in the GUI.

**Recommended fix:** process movement as tile-level microsteps and resolve
pellet collection and collisions after every transition.

## Medium-priority findings

### M1. Tunnel topology is inconsistently encoded

`encodeObservation()` checks out-of-bounds coordinates directly when building
the wall mask, so a legal tunnel move is encoded as blocked. The linear agent
then uses that incorrect mask for its wall feature and post-action ghost
geometry. Vertical-tunnel ghost distance, zone, relative offset, and heading
also wrap x only, not y.

**Recommended fix:** use shared wrapped-position/displacement helpers throughout
observation encoding and bump `OBSERVATION_KEY_VERSION` for changed tabular
states.

### M2. Evaluation destroys the current training episode

Both `TrainingController.evaluate()` and the headless `runEvalPass()` evaluate
inside the live training environment. They restore by resetting to the episode
seed, which restarts the episode instead of restoring positions, pellets,
timers, step count, curriculum state, and RNG state. Evaluation frequency can
therefore change training, and partial recordings can span a hidden restart.

**Recommended fix:** use a separate environment instance for evaluation.

### M3. `epsilonMinDecay` does not follow its documented schedule

`endEpisode()` clamps epsilon against the old floor and then lowers the floor.
Epsilon is consequently above the new floor, so floor decay pauses until
epsilon catches up. With `epsilonDecay=0.999997` and
`epsilonMinDecay=0.9999`, the advertised per-episode floor decay fires roughly
once every 34 episodes and is largely governed by `epsilonDecay`.

**Recommended fix:** define the intended effective-epsilon behavior explicitly,
track whether second-stage decay has engaged, and test unequal decay rates.

### M4. Policy merge accepts semantically incompatible inputs

`mergePolicies()` checks key version and visit-table presence but not algorithm,
maze, ghost count, optimistic initialization, or exact finite array shapes. It
then copies metadata from the first policy, so mixed inputs can produce a
mislabelled and semantically invalid merged policy.

### M5. Loaded policy JSON is trusted after shallow checks

The UI checks only a few top-level fields before casting JSON to a serialized
policy type. Agent loaders trust hyperparameters, keys, and numeric arrays.
Malformed or non-finite values can poison Q-values and hyperparameters or be
reported as a successful no-op load.

### M6. Benchmark scripts can report invalid or incomplete runs as successful

The headless bench casts enum-like arguments without validating them and
silently falls back to the default reward preset while retaining the invalid
name in summaries. `algorithm-compare.sh` and `hyperparam-sweep.sh` suppress
worker failures with `wait ... || true`, allowing partial experiments to exit
successfully.

## Low-priority finding

### L1. Changing ghost count resets a linear agent to tabular epsilon

The confirmed ghost-count reset path always assigns `baseHyper.epsilon` (0.5),
while the linear default is 0.3. The separate Reset-Q path already selects the
correct algorithm-specific default.

## Documentation and coverage gaps

- `README.md` still describes one ghost, 400 maximum steps, `winBonus=200`, and
  `survivalReward=0.02`; production uses two ghosts, 1000 steps,
  `winBonus=1000`, and zero survival reward.
- `test_history.md` and `ROADMAP.md` still describe the linear agent as parked
  and ineffective even though D8/D9 claim 20–30% eval wins.
- The D8–D10 experiments were not added to the empirical history.
- There is no test asserting GUI/bench default parity.
- Ghost-count load tests cover agent discard behavior but not the complete UI
  workflow.
- Speed tests do not cover pellets or collisions at speeds above one.
- Tunnel tests cover movement and pellet BFS, not wall-mask and ghost-geometry
  parity.
- Evaluation restoration is tested only from an episode's initial state.

## Validation performed during review

- `npm test -- --reporter=verbose` — 23 files, 260 tests passed.
- `npm run typecheck` — passed.
- `npm run lint` — passed.
- `npm run build` — passed.
- `bash -n scripts/*.sh` — passed.
- `git diff --check` — passed.

The highest-priority implementation order is H1, H2, then H3.
