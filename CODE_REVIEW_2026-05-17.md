# Weekly Code Review — 2026-05-10 → 2026-05-17

Scope: 53 commits + working-tree changes across `src/env`, `src/rl`, `src/ghosts`,
`src/App.tsx`, `src/render`, `src/ui`, `src/main.tsx`, `scripts/`, `vite.config.ts`.

Findings below are real bugs only (no style nits). Priority ranking at the bottom.

---

## CRITICAL — fix before next training run

### C1. Federated Q-table merge actively destroys learned signal
**File:** `scripts/merge-policies.ts:65-82`
Optimistic init seeds every action slot to 50. The merger averages all four
slots across every worker that observed the state, treating untried slots
(`50`) as legitimate observations. When worker A has `Q[2]=-3.4` (learned
death) and worker B never tried action 2 (`Q[2]=50` from init), the merged
result is `~23.3` — the negative signal is gone.

**Fix:** Track per-slot visit counts in `update()` (parallel `Map<key, Uint8Array(4)>`),
serialize them, and average only over visited slots. Or merge by `max` for
slots above init and `min` for slots below.

### C2. `epsilon` decay survives `serialize()` → reloaded workers don't explore
**File:** `src/rl/qlearning.ts:142, 147`, `scripts/merge-policies.ts:95`
`serialize()` writes the *current* (decayed) `hyper.epsilon`. After a long
run that's `~epsilonMin`. `load()` does `this.hyper = {...data.hyper}`,
clobbering the worker's intended starting ε. Federated workers warm-started
from a merged policy run nearly-greedy from step 1.

**Fix:** In `load()`, preserve exploration fields:
```ts
const { epsilon, epsilonDecay, epsilonMin, endgameEpsilon,
        endgameBucketThreshold } = this.hyper;
this.hyper = { ...data.hyper, epsilon, epsilonDecay, epsilonMin,
               endgameEpsilon, endgameBucketThreshold };
```
And reset `hyper.epsilon` to a sensible default in `merge-policies.ts`.

### C3. Anti-reversal mapping is wrong (up↔left, down↔right)
**File:** `src/env/environment.ts:269-273`
```ts
const lastReversed = (this.lastAction + 2) % 4 === this.secondLastAction;
```
`DIRECTIONS = ['up','down','left','right']`, so `+2 mod 4` pairs
up(0)↔left(2), down(1)↔right(3). The actual geometric opposites are
up↔down and left↔right. The entire anti-oscillation feature (commits
a4ebec6, 36c5ec7, 225f8b3) is broken: it never blocks real up↔down
oscillation and spuriously trims legal actions during legitimate
up→left→up corner-circling.

**Fix:**
```ts
const OPP = [1, 0, 3, 2];
const lastReversed   = OPP[this.lastAction]       === this.secondLastAction;
const secondReversed = OPP[this.secondLastAction] === this.thirdLastAction;
```

### C4. `evaluate()` mutates training env AND training RNG
**File:** `src/rl/trainingController.ts:134-157`
- Shares `this.rng` with `singleStep()`; even at ε=0, `act()` calls
  `random()` for tie-breaking. After eval, the training stream is shifted.
- Calls `env.reset(i+1000)` 20 times in a row but never restores the
  in-progress training episode. The next `singleStep()` resumes against an
  env mid-eval, producing a Q-update across a hidden reset boundary.

**Fix:** Use a dedicated `evalRng = new SeededRng(0xE0A1)` reseeded each
call, and either use a separate `PacmanEnvironment` for eval or
re-reset to the current training seed before returning.

### C5. `CanvasRenderer` reconstructed every frame — animations dead, layout thrash
**File:** `src/App.tsx:202-206`
`new CanvasRenderer(ctx).draw(...)` in the effect — `lastHash`/`tile`/
`frameCount` instance state is thrown away every render. The hash-skip
short-circuit at `canvasRenderer.ts:19` never fires, `tile` is
recomputed from `parentElement.clientWidth` on every redraw (visible
snap during sidebar resize), and mouth/pulse/flash animation tied to
`frameCount` jitters with whatever `tick` cadence happens.

**Fix:**
```ts
const rendererRef = useRef<CanvasRenderer | null>(null);
useEffect(() => {
  const ctx = canvasRef.current?.getContext('2d');
  if (!ctx) return;
  if (!rendererRef.current) rendererRef.current = new CanvasRenderer(ctx);
  rendererRef.current.draw(env, viewMode === 'heatmap');
}, [env, tick, viewMode]);
```

### C6. `Math.min(...values)` in `buildSparkPath` crashes on long runs
**File:** `src/App.tsx:41-42`
`Math.min(...values)` / `Math.max(...values)` with `values.length > ~125k`
throws `RangeError: Maximum call stack size exceeded` in V8. With
`timeRange = 0` ("All") and a long training run, the sparkline crashes
the entire React tree. `LineChart.tsx:73-77` has the same pattern.

**Fix:** Use a reduce:
```ts
let mn = Infinity, mx = -Infinity;
for (const v of values) {
  if (!Number.isFinite(v)) continue;
  if (v < mn) mn = v;
  if (v > mx) mx = v;
}
```

### C7. `consumeForceReverse` is a single flag consumed by ghost 0 only
**File:** `src/ghosts/ghostAi.ts:152`; flag at `src/env/environment.ts:338, 233-237`
`chooseGhostMove` calls `env.consumeForceReverse()` per ghost. The first
ghost clears the flag; ghosts 1..N see `false` and never reverse on
chase↔scatter transitions.

**Fix:** Either set a per-ghost `pendingReverse` flag in `step()` when
the phase flips, or hold the env-wide flag through the whole tick and
clear it after the loop.

---

## HIGH

### H1. `parallel` workers don't actually receive cleanup signals
**File:** `scripts/run-parallel.sh:133-141, 155`
Workers are launched via `npx vite-node ... &`. `npx` is the immediate
child; the real `node` is a grandchild. `kill -TERM $npx_pid` kills npx
and orphans node, which dies on SIGPIPE without running its
`SIGINT/SIGTERM` flush handler. The comment "each flushes its policy +
summary before exit" is false in practice — partial Q-tables since the
last `snapshotEvery` are lost.

**Fix:** Launch with `setsid` (or `exec` directly to vite-node) and
signal the process group: `kill -TERM -- -$pid`. In `cleanup`, loop
`wait` per pid rather than `wait "${pids[@]}"`.

### H2. `releaseDelay > 0` ghosts skip both movement AND `edibleTimer` tick
**File:** `src/env/environment.ts:417-436`
A delayed ghost has its entire per-tick branch short-circuited. Two
consequences:
- On houseless mazes (`inBox=false`), the collision check at `:443` does
  not skip the ghost — Pac-Man walking onto its start tile dies even
  though the ghost is "not yet released".
- If Pac-Man eats a power pellet while ghost has `releaseDelay > 0`,
  the ghost's `edibleTimer` is set to full duration but never decrements
  until release. Released ghosts spawn pre-loaded with full edibility.

**Fix:** Decrement `edibleTimer` and `releaseDelay` unconditionally each
tick; only gate *movement* (and `chooseGhostMove`) on `releaseDelay === 0`.

### H3. Param change mid-training corrupts Q-updates across reset boundary
**File:** `src/App.tsx:209-213`
```ts
useEffect(() => {
  env.setParams(params); env.reset(seed); setTick(t=>t+1);
}, [params, seed, env]);
```
Doesn't stop the rAF loop. `trainer.singleStep()` runs on the next
frame: its cached `obs` is pre-reset, its `nextObs` is post-reset, and
the Q-update is garbage. Hits any time the user changes maze/numGhosts
while training.

**Fix:** Pause training in this effect, or queue param changes for the
next episode boundary.

### H4. Footer "Load" doesn't pass `numGhosts` → Q-table key mismatch
**File:** `src/App.tsx:826-830`
Unlike `loadTrainedPolicy` (line 309-327) which passes `targetNumGhosts`,
the footer file-load calls `agent.load(JSON.parse(await file.text()))`
single-arg. If the policy was saved with a different `numGhosts` than
current, every observation key encodes differently and every lookup
misses silently.

**Fix:** Read `numGhostsEncoded` from the file and pass it (or read
`params.numGhosts` and force a switch like `loadTrainedPolicy` does).

### H5. Ghost-eat combo and reward leak between Pac-Men
**File:** `src/env/environment.ts:457-461`
`ghostsEatenCombo` is a single env-level counter incremented per eat in
the multi-Pac loop. If Pac 0 eats one ghost (combo=1, +30) and Pac 1
eats a different ghost in the same tick, Pac 1 gets `30*2=60`. Worse,
the returned `reward` is added to *Pac 0's* StepResult (env returns
`pacmen[0]`'s view), so Pac 1's eat reward poisons Pac 0's training
signal. Pellet rewards at lines 395-414 don't add to `reward` for extra
pacs — inconsistent.

**Fix:** Return per-pacman reward in StepResult, or only update `reward`
when `pacman.id === 0` (consistently across pellets *and* ghost eats).

### H6. Pinky/Inky aim at stale `pacLastDir` when Pac is wall-blocked
**File:** `src/ghosts/ghostAi.ts:107-110` (+ `environment.ts:355, 361`)
`getPacLastDir()` is only updated when Pac actually moves. With
`illegalMoveMode='stay'` (default), holding against a wall freezes
`pacLastDir`, possibly at the initial `'left'` from `reset()`. Pinky/Inky
target 4 tiles *left* of a Pac who is no longer facing left.

**Fix:** Track `pacDesiredDir` (last requested action) separately, or
fall back to the agent's last action when the move was illegal.

### H7. Death + winBonus stack on last-pellet-with-collision
**File:** `src/env/environment.ts:438-480`
Collision sets `done=true` and `+deathPenalty`. The subsequent
`if (this.pelletsLeft <= 0)` fires unconditionally, also adding
`winBonus`. Terminal Q-value estimates for late-game states see
huge net-positive rewards on what was actually a death.

**Fix:** Gate the win-bonus block on `!done`, or check collision first
and return early.

### H8. AI-watch uses hard-coded direction order and `Math.random`
**File:** `src/App.tsx:221-232`
```ts
const action = agent.act(obs, env.getLegalActions()
  .map(d => ['up','down','left','right'].indexOf(d)), Math.random);
```
- Bypasses the seeded RNG → non-deterministic vs the seed shown in UI.
- Hard-codes direction order rather than importing `DIRECTIONS`. If
  `DIRECTIONS` order ever changes, AI watch silently scrambles actions
  without any type error.

**Fix:** Import `DIRECTIONS`, reuse it, and pass `() => trainer.rng.next()`.

### H9. `numGhosts` mismatch on load() warns but loads anyway → state aliasing
**File:** `src/rl/qlearning.ts:160-173`
On mismatch, code logs a warning then runs the load loop unconditionally.
Keys that *do* coincidentally match represent semantically different
states (different ghost count → different encoding). Silent contamination
of Q-values is worse than a clean miss.

**Fix:** On mismatch, `this.q.clear(); return;` (match the version-mismatch
branch behavior).

### H10. `lastAction` blows past its allotted base for out-of-range actions
**File:** `src/env/environment.ts:328`, `src/env/observation.ts:231, 259`
`this.lastAction = action` is set before clamping. `LAST_ACTION_BASE=5`
reserves `[0..4]` (raw `[-1..3]`). Any caller passing action `4` or `-2`
makes `observationKey` overflow its slot, colliding with
`pelletsRemainingBucket` — silent Q-table corruption.

**Fix:** Clamp on assignment:
`this.lastAction = Math.max(-1, Math.min(3, action));`

### H11. `replayRecording` aliases frame objects (mutation cross-contamination)
**File:** `src/rl/trainingController.ts:184-191`
Returned positions array stores references to `frame.pacPos` /
`frame.ghostPositions`. If any consumer (e.g. canvas renderer wrap/clamp)
mutates in place, the recording is corrupted and replays diverge.

**Fix:** Shallow-clone on push.

### H12. `canvasRenderer` `getPacmen()[0]` unguarded → throws during transient param change
**File:** `src/render/canvasRenderer.ts:18, 74`
Reads `env.getPacmen()[0].pos.x` with no guard. During the brief window
between `env.reset()` clearing pacmen and re-creating them, `[0]` is
undefined and the draw call throws inside the render effect, killing
further redraws.

**Fix:** Optional-chain or early-return on missing pacman.

### H13. `merge-policies.ts` silently treats missing action slots as 0
**File:** `scripts/merge-policies.ts:69-72`
```ts
for (let i = 0; i < 4; i += 1) sum[i] += values[i] ?? 0;
```
If a worker's `values` array has length < 4, the missing slots
contribute 0 to the sum but are still divided by full worker count.
Zeros pull the merged value far below the optimistic init prior (~50).
One corrupted worker poisons the merge.

**Fix:** Per-slot counting; skip undefined slots:
`if (values[i] !== undefined) { sum[i] += values[i]; counts[i]++; }`.

### H14. `import './app.css'` in main.tsx references uncommitted file
**File:** `src/main.tsx` (working tree)
`src/app.css` is listed under "Untracked files" in `git status`. The
import will fail on any fresh checkout / CI build.

**Fix:** Either commit `src/app.css` or remove the import.

---

## MEDIUM

### M1. `clearPelletsTo` uses unstable sort comparator
**File:** `src/env/environment.ts:130-142`
```ts
all.sort((a, b) => (a.dist - b.dist) + (rand() - rand()) * 0.5);
```
`Array.prototype.sort` calls the comparator multiple times per pair;
each call rolls fresh random offsets, producing inconsistent ordering.
Non-deterministic curriculum even with seeded RNG; older V8 used to
throw on inconsistent comparators.

**Fix:** Decorate-sort-undecorate:
```ts
const dec = all.map(p => ({ p, k: p.dist + (rand()-rand())*0.5 }));
dec.sort((a, b) => a.k - b.k);
```

### M2. `nearestPelletDir` direction encoding disagrees with action space
**File:** `src/env/observation.ts:8-9, 96-101`
`DIRS` in observation.ts is `[up=0, right=1, down=2, left=3]` (rotational).
Action space (`types.ts:3`) is `[up=0, down=1, left=2, right=3]`. The
agent reads `nearestPelletDir=1` meaning "right" while action `1` is
"down". Confuses any heuristic policy and aliases unrelated states for
the learned Q-table.

**Fix:** Reorder observation `DIRS` to match `DIRECTIONS`, update doc.

### M3. `endEpisode` epsilon decay persists into save — see C2; same root cause.

### M4. `train()`-loop `stop()` lets up to `stepsPerFrame` more steps run
**File:** `src/rl/trainingController.ts:104-127`
No `if (!this.running) break;` inside the inner for-loop. In max-speed
mode, up to 1M steps can execute after the user clicks Stop before the
outer loop check catches up.

**Fix:** Add the break inside the for-loop, checking `myId` too.

### M5. `presetBench.test.ts` still tests γ=0.95 after production switched to 0.99
**File:** `src/rl/presetBench.test.ts:27`
Bench thresholds (lines 76-77, 87) were tuned at γ=0.95; production now
runs γ=0.99. Tests pass but no longer reflect production behavior.

**Fix:** Update gamma and re-baseline thresholds.

### M6. `qlearning.test.ts` never exercises default `optimisticInit=50`
**File:** `src/rl/qlearning.test.ts:20, 42, 66`
Every test sets `optimisticInit: -1`. Federated-merge bug (C1) and
bootstrap-with-50 corner cases have zero coverage.

**Fix:** Add a test using production defaults; add a merge test asserting
behavior when worker A visited action 2 and worker B visited action 0.

### M7. `evaluate()` shared env: also see C4 — restoration is part of fix.

### M8. Spacebar handler captures stale closures
**File:** `src/App.tsx:250-262`
Empty deps with an eslint-disable comment. Today's `startTraining`/
`stopTraining` only touch refs/memos so it works — but the disable hides
the next regression.

**Fix:** Stash latest functions in a ref, or include them in deps.

### M9. Numeric input → `NaN`/`0` into params
**File:** `src/App.tsx:550, 555, 573, 578, 620, 625…`
`onChange={e => setParams(p => ({...p, numGhosts: Number(e.target.value)}))}`
— clearing the field gives `0`; typing `-` alone gives `NaN`. NaN
propagates into reward calcs.

**Fix:** Guard `Number.isFinite(n)` before committing, or hold a string
buffer.

### M10. Movement-MA recomputed full-history every render
**File:** `src/App.tsx:339, 341, 873`
`movingAverage(scores, 20)` is O(n·w). On a 100k-episode run that's 2M
ops per render. Combined with C6 the UI freezes.

**Fix:** Memoize or rolling-sum.

### M11. Top "Reset" button leaves training stats stale
**File:** `src/App.tsx:401-403` vs `818-822`
Top reset clears env but not `trainer.stats`. Episode counter HUD chip
keeps the old episode count against a fresh env. Footer "Reset Q" does
clear stats. Two buttons disagree about what "reset" means.

**Fix:** Either align behavior or rename one.

### M12. Human/AI-watch death loops the same seed forever
**File:** `src/App.tsx:228, 242`
`if (result.done) env.reset(seed)` reuses the constant `seed` for every
death, so post-death episodes are identical Groundhog Day. Defeats the
purpose of "watch the AI play."

**Fix:** Use a fresh seed per reset (`Math.random()*1e6|0` or
`trainer.rng.int(...)`).

### M13. `vite.config.ts` middleware walks `bench-out/` sync, no try/catch
**File:** `vite.config.ts:25-47`
`statSync(p)` throws if a worker deletes/renames a file mid-walk →
returns 500. No caching, so each request re-walks potentially thousands
of files synchronously, blocking the dev server.

**Fix:** Wrap `statSync` in try/catch; memoize the result for a few seconds.

### M14. `findPowerPelletPositions` can return duplicates on tight mazes
**File:** `src/mazes/mazes.ts:87-95`
Each corner runs `findOpenNear` independently with no shared "already-
chosen" set. On small procedural mazes two corners can resolve to the
same tile. `power[][]` is idempotent so `powerPelletsLeft` count stays
correct, but `powerPelletPositions.length` disagrees, breaking any
per-position iteration.

**Fix:** Dedupe by `(x,y)` after the map.

### M15. `parse()` never assigns `ghostHouseExit` (latent)
**File:** `src/mazes/mazes.ts:52-72`
Today's parsed mazes (`m2`, `m3`) have no `2` tiles, so this latent.
But anyone adding a `2`-tile maze via `parse()` will get broken ghost
release with no warning.

**Fix:** Either error if any `2` exists without explicit `ghostHouseExit`,
or auto-detect (first open tile above the topmost `2`).

### M16. `MAZES.length=8` hard-coded
**File:** `src/mazes/mazes.ts:280-296`
Works today (3 static + 5 procedural). Adding one static maze causes
`Object.defineProperty(MAZES, 3, ...)` to override it. Breaks silently.

**Fix:** Compute `length = STATIC_MAZES.length + PROC_SEEDS.length`,
offset procedural getters by `STATIC_MAZES.length`.

### M17. `overnight-bench.ts` sps column is per-report not per-episode
**File:** `scripts/overnight-bench.ts:294, 297`
`stepsSinceReport / elapsedSinceReport` is divided by ~60s and written
to *every* episode row in the window — every episode in a report
window gets the same monotonically-growing SPS. Column is meaningless.

**Fix:** Track per-episode start time/steps, or drop the column.

### M18. `overnight-bench.ts:218-226` non-atomic policy write
SIGKILL mid-`writeFileSync` on a multi-MB policy truncates the file.
`merge-policies.ts:42-44` silently drops the worker on parse failure
(see M19), so federated merge can complete with N-K workers and no
warning.

**Fix:** Write to `policy-latest.json.tmp` then `renameSync`.

### M19. `merge-policies.ts` silently skips unparseable policies
**File:** `scripts/merge-policies.ts:42-44`
Failure only aborts if *all* parse failures. Better: require
`--allow-partial` to merge incomplete sets.

### M20. `overnight-bench.ts:68-72` no validation on numeric CLI args
`num('alpha','foo')` returns `NaN`. NaN poisons every Q-update silently;
scores stay flat with no error.

**Fix:** Validate `Number.isFinite(v)` and abort with a clear message.

### M21. `computeDelta` accepts tiny negative `prev`
**File:** `src/App.tsx:56-64`
Guards `prev === 0` exactly; `prev = -0.001` produces astronomical
percentages displayed as "▲ 999999.9%".

**Fix:** Compare `Math.abs(prev) < epsilon`.

### M22. `run-sweep.sh:109-127` `reap_finished` interleaves with `cleanup`
Both iterate `JOB_GROUP` and both `unset`; concurrent execution in the
trap leaves TSV rows with empty `group`/`run_id` columns or shifted
fields.

**Fix:** Snapshot keys before iterating, and call `append_report` *before*
`unset`.

---

## LOW / latent

- **L1.** `getLegalActions` reversal — see C3.
- **L2.** Ghost-eaten respawn sets `releaseDelay=0` (re-emerges immediately).
  Possibly intentional for RL; classic Pac-Man waits.
  (`src/env/environment.ts:465`)
- **L3.** Cross-over collision detection ignores ghosts that were eaten
  earlier in same tick by another pac (multi-pac edge case).
  (`src/env/environment.ts:456-470`)
- **L4.** Tunnel wraparound is x-only in `nextPosition` — latent until a
  maze uses vertical tunnels. (`src/ghosts/ghostAi.ts:15-20`,
  `src/env/environment.ts:250-256`)
- **L5.** `LineChart.hexToRgb` regex rejects 3-digit hex `#0f0` (silently
  produces 0,0,0). (`src/ui/LineChart.tsx:16`)
- **L6.** `requestFullscreen` uses `document.querySelector('.maze-body')`
  — brittle if a second `.maze-body` ever appears. (`src/App.tsx:435-438`)
- **L7.** `comparisonTrainer` allocated in `useMemo` and never used.
  (`src/App.tsx:194-196`)
- **L8.** `observationKeyToString` hard-codes `v7:` prefix with no
  version stored in the key — future bumps can't tell mixed-version
  keys apart. (`src/env/observation.ts:275-289`)
- **L9.** `loadTrainedPolicy` ordering: `setParams` → `agent.load` →
  reset effect fires next render. Canvas can briefly show old maze
  during the swap. (`src/App.tsx:309-327`)
- **L10.** `overnight-bench.ts:67` silently skips flag-only args
  (`--clean`) — footgun if anyone passes shell-script flags directly.
- **L11.** `overnight-bench.ts:97` `powerPellets=anything-but-false` is
  true. `powerPellets=0` enables them.
- **L12.** `overnight-bench.ts:392` burst loop overshoots `maxEpisodes` by
  up to ~5000 steps — visible on small smoke tests.
- **L13.** Hard-coded tab counts ("8", "13", "4") in App tabs — silently
  lie when fields change. (`src/App.tsx:507-511`)

---

## Test coverage gaps that hide active bugs

- **`engine/collision.test.ts`** has ONE test (wall blocks movement).
  Zero coverage of: ghost-pac touch, cross-over swap, `inBox` skip,
  multi-pac collision, eating edible ghosts, power-pellet activation.
  Bugs H2, H5, H7, L3 all invisible to tests.
- **`ghosts/ghostAi.test.ts`** has no tests for `removeReverse`,
  `consumeForceReverse` integration, edible-flee behavior, in-box BFS,
  Inky vector targeting, Clyde 8-tile switch, `lastDir` mutation.
  Bug C7 invisible.
- **`env/environment.test.ts`** has no anti-reversal test → bug C3
  shipped uncaught. No assertion on reward in death+win edge case → H7
  shipped uncaught.
- **`rl/qlearning.test.ts`** never exercises production default
  `optimisticInit=50` (every test overrides to `-1`). Bug C1 invisible.
- **`rl/presetBench.test.ts`** uses γ=0.95 despite production switch to
  γ=0.99 — see M5.
- **`env/observation.test.ts:84-89`** only checks `gc1=0` vs `gc1=1` for
  the lastAction collision test. Doesn't probe `lastAction=4` (bug H10).
- **`mazes.test.ts`** checks pellet positions are open, but not unique
  — bug M14 invisible.

---

## Recommended fix order

1. **C1, C2** — federated training is fundamentally broken; everything
   you ran this week comparing federated vs single-worker is suspect.
2. **C3, C7** — gameplay-correctness bugs that affect every training
   run. Cheap to fix.
3. **C4** — your eval-vs-training numbers in the UI are not measuring
   what you think.
4. **C5, C6** — UI stability for long runs. C6 is a hard crash.
5. **H1, H2** — federated worker losses + houseless-maze ghost death
   traps.
6. **H3, H7, H10** — silent state corruption during normal use.
7. **H4, H5, H6, H8, H9, H11–H14** — high-impact correctness.
8. Address test gaps alongside the corresponding fixes so regressions
   don't recur.
