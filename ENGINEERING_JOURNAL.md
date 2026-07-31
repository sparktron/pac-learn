# Engineering Journal

This is the narrative record of how AI Pac-Man Lab is being engineered and
improved. It is meant to teach the process: what we believed, what we changed,
what failed, what the evidence showed, and why the next decision followed.

Use `test_history.md` for detailed experiment tables and historical metrics.
Use `ROADMAP.md` for current priorities. Use this journal for the reasoning
that connects them.

## Journal rules

1. Append an entry after every meaningful code change, investigation, or
   training experiment.
2. Record the hypothesis before interpreting the result.
3. Include failures, regressions, and discarded runs. Do not preserve only the
   successful path.
4. Separate observations from interpretations.
5. Link the exact commit, output directory, report, or test that supports the
   conclusion.
6. State the resulting decision: adopt, reject, retry, or leave unresolved.
7. Do not silently rewrite old conclusions. Add a dated correction when later
   evidence supersedes them.

## Entry template

```markdown
### YYYY-MM-DD — Short title

**Context:** Why this work was undertaken.

**Hypothesis:** The falsifiable belief being tested.

**Change / experiment:** Exact code or configuration change.

**Validation:** Tests, seeds, duration, evaluation set, and relevant artifacts.

**Result:** Measurements and observed behavior.

**Failures / surprises:** What broke, underperformed, or invalidated an earlier
assumption.

**Decision:** Adopt, reject, revise, or investigate further.

**Lesson:** The reusable engineering principle.
```

---

## Historical journal

### 2026-05-15 — Zero wins became an observability problem

**Context:** Early runs reported no useful learning signal.

**Hypothesis:** The agent might be making progress that the top-level win rate
did not expose.

**Change / experiment:** The benchmark gained raw win counts, termination
reasons, score variance, pellets remaining, and pellet-count percentiles.
Evaluation size later increased from 30 to 200 episodes.

**Result:** Runs that all displayed a rounded `0.000` win rate had materially
different near-win behavior. `p5`, minimum pellets remaining, and raw wins
showed movement before the rounded rate did.

**Failures / surprises:** The original metric made distinct policies look
equally unsuccessful. Small evaluation sets had score error larger than many
of the improvements being compared.

**Decision:** Keep raw wins and pellet percentiles as primary learning metrics.
Do not use rounded win rate alone to select a configuration.

**Lesson:** Improve the measurement system before changing the learning
algorithm. An invisible improvement is operationally indistinguishable from no
improvement.

### 2026-05-16 — Reward shaping and curriculum produced the first wins

**Context:** The original agent was rewarded for surviving and often could not
finish within the episode limit.

**Hypothesis:** Larger terminal value, no survival incentive, optimistic
initialization, and deliberate endgame exposure would make winning learnable.

**Change / experiment:** The work increased `winBonus` to 1000, removed
`survivalReward`, added late-pellet escalation, raised the episode limit,
introduced optimistic Q initialization, slowed epsilon decay, and added an
endgame curriculum.

**Result:** The two-ghost tabular track produced its first greedy wins and
eventually reached approximately 2.5% best evaluation win rate with
`p5 ≈ 55`.

**Failures / surprises:** A 400-step episode limit made completion effectively
impossible. Pessimistic initialization caused early commitment. Combining a
high endgame epsilon floor with curriculum hurt rather than helped.

**Decision:** Retain the win-seeking reward preset, optimistic initialization,
longer episodes, and curriculum. Treat exploration and curriculum as interacting
knobs rather than independently beneficial features.

**Lesson:** Reward design, horizon, initialization, and data distribution form
one learning system. Improving one component can fail if another still makes
the desired behavior unreachable or irrational.

### 2026-06-17 — Tuning alpha could not rescue the old linear agent

**Context:** Continuous pellet and ghost distances had been added to the linear
function approximator, but it remained far behind tabular.

**Hypothesis:** The learning rate might be the missing lever.

**Change / experiment:** Six five-minute runs swept
`alpha ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}`.

**Result:** Viable values clustered around roughly 300 peak evaluation score,
about three times below tabular, with no wins. `alpha=0.001` learned too slowly
and `alpha=0.3` diverged.

**Failures / surprises:** A broad alpha sweep could not change the performance
class of the model.

**Decision:** Stop tuning alpha and inspect representational structure.

**Lesson:** Hyperparameter tuning cannot repair a model that cannot express the
required policy.

### 2026-06-27 — Deterministic greedy tie-breaking recovered hidden quality

**Context:** Optimistic Q initialization left many equal-valued actions, so a
nominally greedy evaluation still behaved randomly in undertrained states.

**Hypothesis:** Directing ties toward the nearest pellet would reveal policy
quality already present in the Q-table.

**Change / experiment:** The same policy and evaluation seeds were measured
with random, visit-count, and pellet-directed tie-breaking.

**Result:** Pellet-directed ties improved average greedy score from 799.7 to
1152.1, a 44% gain without retraining. Visit-count tie-breaking was roughly
neutral.

**Failures / surprises:** The evaluation policy, not only the learned values,
was suppressing measured performance.

**Decision:** Use pellet-directed greedy evaluation.

**Lesson:** Evaluation behavior is part of the experimental apparatus. Verify
that “greedy” really means deterministic exploitation.

### 2026-07-02 — Action-conditioned features fixed the linear model

**Context:** The old model used separate action weights over the same
state-only features: `Q(s,a)=w_a·f(s)`.

**Hypothesis:** The model could not express “take the action that approaches a
pellet” or “avoid the action that approaches a ghost” because features did not
describe the candidate action's consequence.

**Change / experiment:** D8 replaced the model with shared
action-conditioned features, `Q(s,a)=w·f(s,a)`, including post-action pellet
and ghost geometry.

**Result:** An eight-minute two-ghost run reached 15–27% wins per evaluation
checkpoint, versus roughly 0–1.5% for tabular under the comparison setup.

**Failures / surprises:** Performance oscillated from 0% to 27% rather than
converging smoothly.

**Decision:** Adopt action-conditioned features and investigate bootstrap
stability.

**Lesson:** Features should encode the effect of an action, not merely describe
the state beside an action label.

### 2026-07-02 — A target network reduced linear TD collapse

**Context:** Linear approximation, bootstrapping, and off-policy Q-learning
formed the classic deadly triad.

**Hypothesis:** Freezing bootstrap weights for a short interval would reduce
the moving-target feedback loop.

**Change / experiment:** D9 added a target weight vector synchronized every
2,000 updates.

**Result:** Mean checkpoint win rate improved from 15.8% to 23.0%, standard
deviation fell from 9.0 to 5.8 points, minimum improved from 0% to 3%, and the
last-30 mean improved from 14.6% to 25.4%.

**Failures / surprises:** Hard target synchronization improved stability but
did not eliminate checkpoint-level oscillation.

**Decision:** Keep `targetSyncSteps=2000` as the linear default.

**Lesson:** Stabilization can be as important as representational capacity when
using bootstrapped function approximation.

### 2026-07-03 — Tabular epsilon-floor decay helped non-sparse metrics

**Context:** A permanent 20% exploration floor disrupted rare endgame
trajectories, but lowering it early had previously removed wins.

**Hypothesis:** Decaying the floor only after epsilon reached it would preserve
early exploration and improve late exploitation.

**Change / experiment:** D10 added optional `epsilonMinDecay` and
`epsilonMinFloor`.

**Result:** In short comparisons, mean evaluation score improved from 996 to
1308 and mean `p5` pellets remaining improved from 145 to 113. Wins remained
too sparse to interpret.

**Failures / surprises:** The implemented floor schedule advances only when
epsilon catches the lowered floor, so its effective cadence differs from the
simple “decay every episode” description.

**Decision:** Keep the feature gated and treat the scheduling semantics as
unresolved before making it a production default.

**Lesson:** When the target metric is sparse, use supporting metrics—but do not
promote a change beyond what those metrics actually prove.

### 2026-07-21 — Full review found correctness defects in the experiment stack

**Context:** Learning numbers are meaningful only if GUI, headless training,
policy loading, and environment transitions implement the same problem.

**Hypothesis:** A full-stack review would uncover experiment-invalidating drift
that unit-level work had missed.

**Change / experiment:** The application, environment, agents, UI, scripts,
tests, and documentation were reviewed. See
[`CODE_REVIEW_2026-07-21.md`](CODE_REVIEW_2026-07-21.md).

**Result:** Three high-priority defects were confirmed and fixed:

- GUI and headless linear defaults had drifted.
- Cross-ghost-count policy loading could discard a policy while appearing to
  load it.
- Speeds above one skipped intermediate pellets and collisions.

**Failures / surprises:** The first microstep collision fix broke the existing
one-tile cross-over timing test by terminating before the ghost completed its
move. The full test suite caught it; the implementation was revised to resolve
intermediate contacts immediately while preserving final-tile cross-over
semantics.

**Decision:** Centralize defaults, return explicit policy-load results, and
resolve high-speed interactions at atomic tile boundaries.

**Lesson:** Regression tests encode behavioral contracts that are easy to lose
inside a locally “more correct” rewrite. Run the complete suite, not only the
new regression tests.

### 2026-07-26 — Corrected single-seed benchmark improved over D9

**Context:** The correctness fixes needed a learning-quality check.

**Hypothesis:** The stabilized action-conditioned linear agent would retain or
improve its D9 performance after the environment fixes.

**Change / experiment:** Linear and tabular agents ran side-by-side for eight
minutes with seed 7 and 200-game evaluation passes.

**Validation:** Artifacts:
`bench-out/20260726-031523-linear-vs-tabular/`.

**Result:** Linear averaged 27.7% evaluation wins, 29.9% over the final 30
checkpoints, and 30.5% in the final pass. Tabular recorded no evaluation wins.
This improved over D9's 23.0% mean and 25.4% last-30 mean.

**Failures / surprises:** An earlier run started before the cross-over
regression fix was loaded and was explicitly discarded rather than mixed into
the conclusion.

**Decision:** Run a multi-seed confirmation before treating the gain as
general.

**Lesson:** Discard contaminated runs openly. A plausible number from the wrong
code version is not evidence.

### 2026-07-26 — Five seeds confirmed the linear gain

**Context:** The seed-7 result could have been favorable training randomness.

**Hypothesis:** If the improvement was structural, independent training seeds
would converge to similar evaluation performance.

**Change / experiment:** Seeds 7, 1007, 2007, 3007, and 4007 each trained for
eight minutes with the production linear configuration.

**Validation:** Artifacts:
`bench-out/20260726-044028-linear-multiseed/`.

**Result:** The aggregate mean evaluation win rate was 27.55%; seed means ranged
from 27.42% to 27.84%, with 0.16 percentage points of seed-to-seed standard
deviation. Pooled training win rate was 10.38%. Mean last-30 and final
evaluation rates were 27.47% and 29.30%.

**Failures / surprises:** Individual checkpoints still dipped as low as 1.5%.
All learned policies were evaluated on the same fixed environment seeds, so the
narrow cross-training-seed interval does not measure environment-distribution
uncertainty.

**Decision:** Treat the D8/D9 gain as repeatable across training seeds. Before
the next tuning claim, diversify held-out evaluation seed panels and investigate
the remaining checkpoint collapses.

**Lesson:** Multi-seed training confirmation and multi-environment evaluation
answer different questions. A robust experiment needs both.

### 2026-07-26 — The engineering journal became a project invariant

**Context:** Metrics and code-review reports existed, but the reasoning that
connected failures to later decisions was spread across commit messages and
several documents.

**Hypothesis:** A structured, append-only narrative would make the process
teachable and prevent future work from repeating failed approaches.

**Change / experiment:** This journal was created and seeded from the review,
commit, and benchmark evidence. `AGENTS.md` now requires future work to record
context, hypothesis, validation, results, failures, decisions, and lessons.
The README and roadmap link back to the journal.

**Validation:** Markdown and whitespace checks passed; historical measurements
were reconciled against `test_history.md`, review reports, and benchmark
artifacts.

**Failures / surprises:** Reconstructing history after the fact is less precise
than recording intent before an experiment. Some older entries can explain the
documented hypothesis only from surviving evidence, not from a contemporaneous
journal note.

**Decision:** Treat journal maintenance as part of the definition of done for
meaningful engineering and training work.

**Lesson:** The process is itself an artifact. If the reasoning is not recorded
when decisions are made, later readers inherit results without the knowledge
needed to reproduce good judgment.

### 2026-07-27 — Linear evaluation honored the shared tie-break contract

**Context:** The shared evaluation default selected pellet-directed tie-breaking,
but the linear agent accepted the option only for interface parity and ignored
it. PR #58 was also based before the latest linear feature, correctness, and
experiment-history work, so updating it produced code and documentation
conflicts.

**Hypothesis:** Applying the shared tie-break only when linear Q-values are
exactly equal would make evaluation deterministic without changing exploratory
training or the learned value function.

**Change / experiment:** `LinearQLearningAgent.act()` now selects a legal
`nearestPelletDir` for pellet-mode ties and otherwise uses the lowest tied action
for deterministic modes. Random remains the direct-call default. The PR branch
was merged with current `master`; the evolved action-conditioned feature code,
both test sets, and the newer experiment history were retained.

**Validation:** `src/rl/linearQlearning.test.ts`,
`src/rl/trainingController.test.ts`, the full test suite, typecheck, lint, and
build were run on the resolved merge.

**Result:** Zero-weight initialization and other exact ties no longer consult
RNG during deterministic linear evaluation. Training behavior remains unchanged
unless a caller explicitly requests a deterministic tie-break.

**Failures / surprises:** The earlier assumption that continuous linear
Q-values effectively never tie was false at minimum during zero-weight
initialization. Resolving the old branch also required preferring current
history over duplicated, stale run counts.

**Decision:** Adopt the shared deterministic evaluation contract for both
tabular and linear agents.

**Lesson:** Interface parity is insufficient when a control changes experiment
semantics; every implementation must honor the option, including initialization
and fallback states.

### 2026-07-27 — Held-out evaluation panels, and the soak harness they unblock

**Context:** The roadmap's next item is a long soak of the D8/D9 linear agent,
whose stated success bar includes "at least 25% on the worst held-out panel."
No held-out panel existed. `runEvalPass()` hardcoded a single evaluation set —
`evalEpisodes` games seeded `1_000_000 + i` — so every result in
`test_history.md`, including the 27.55% five-seed mean, is a score on one fixed
set of 200 mazes. The soak could not have answered the question it was for.

**Hypothesis:** Evaluating each checkpoint on several disjoint seed panels
separates a policy that generalizes from one that has fit the default panel,
and can be added without perturbing the training stream at all, because
evaluation draws only from its own `evalRng` and never from the training `rng`.

**Change / experiment:** `overnight-bench.ts` gained `evalPanels=<a,b,…>`
(default `1000000`), aborting on non-numeric bases, an empty list, or bases
closer together than `evalEpisodes` — overlapping panels would report the same
mazes as independent evidence and make "worst panel" meaningless. Each pass
writes one `evals.csv` row per panel with a trailing `panel` column, chosen
over a new file so column positions 1–12 stay valid for existing readers
(`run-parallel.sh` reads column 7 positionally). Multi-panel passes also print
a mean/worst line. `scripts/run-soak.sh` runs seeds as independent processes
and aggregates `meanWinRate` / `worstPanelMean` / `checkpointP5`.

**Validation:** A/B against the committed bench at seed 7, 600 episodes: with
default panels, `evals.csv` columns 1–12, `episodes.csv` (every column but the
wall-clock `stepsPerSec`), and the serialized policy weights are identical —
only the `savedAt` timestamp differs. A second control at matched `evalEvery`
confirmed 4 panels versus 1 leaves the training stream identical and reproduces
the default panel's rows exactly, so panel count is a pure measurement change.
Guard rails were exercised directly; 270 tests, typecheck, lint, and a
two-seed end-to-end soak smoke all pass.

**Result:** The soak's success bar is now measurable. The smoke run's numbers
are not evidence about the agent — 0.4 minutes per seed, 30-game panels — but
they confirmed the aggregation reports the intended three statistics.

**Failures / surprises:** Two pre-existing artifacts surfaced while proving
non-perturbation. `runEvalPass()`'s closing `env.reset(episodeSeed)` fires
whenever the 5,000-step burst ends mid-episode, silently discarding that
in-flight training episode — no `endEpisode()`, no CSV row, no epsilon decay —
and it also wipes the endgame-curriculum fast-forward from the episode that
follows an eval. Both are roughly one episode per `evalEvery`, negligible at
the default 2000, and both predate this change. Filed separately rather than
folded in, to keep the measurement change free of training-behavior edits.

**Decision:** Land the panels and the harness before running the soak, so the
long run produces generalization evidence rather than another number on the
panel the agent has always been scored against.

**Lesson:** A metric that has never varied is not a validated metric. Five
seeds agreeing to 0.16 percentage points measured the stability of training,
not the generality of the policy, because the evaluation set was constant
across all five. Vary what you claim to be measuring over.

### 2026-07-28 — The soak answered a different question than it asked

**Context:** The 8-hour five-seed linear soak was stopped at ~6h50m (~85%),
after ~6.5M episodes per seed. The processes were SIGKILLed, so no
`summary.json` or `summary.tsv` was written, but `evals.csv` and
`episodes.csv` were complete and every number below is recomputed from them.

**Hypothesis under test:** that longer training would raise the mean and make
the occasional low checkpoint disappear (target ≥32% mean, ≥25% worst
held-out panel, ≥15% checkpoint p5).

**Result — the hypothesis is falsified, and not narrowly.** Win rate is flat
across the entire run. Seed 7 by decile: 20.86% (ep 2k–648k) → 21.48%
(mid) → 21.28% (ep 5.8M–6.5M). First-100 versus last-100 checkpoints across
all five seeds: −0.30, +0.38, +0.58, −0.07, −0.09 percentage points. The agent
is converged by its first checkpoint at episode 2,016 — roughly thirty seconds
— and 6h50m adds nothing measurable. Final: 21.0% mean, 19.4% worst panel,
16.0% checkpoint p5. One of three targets met, and the one met is the tail.

**The larger finding — a regression the soak surfaced by accident.** Matched
against the 2026-07-26 baseline on the same seeds, the same panel (1000000),
and the same episode range (<160k), the soak scored 6.4 points lower on every
seed: 21.82/20.56/20.50/21.62/21.16% versus 27.50/27.84/27.48/27.53/27.42%.
Training was untouched — 10.54% training win rate versus 10.38% — which
localized it to evaluation. The only code change between the two runs is
9b0a880, the 2026-07-27 commit that made the linear agent honor the shared
`pellet` evaluation tie-break.

**Change / experiment:** added `evalTieBreak=<random|visits|pellet>` to
`overnight-bench.ts` so this is config-only, then ran seed 7 for 8 minutes at
`evalEvery=500` with the tie-break as the sole variable.

| tie-break | ckpts | mean eval WR | min | max | avgLen | train WR |
|---|---:|---:|---:|---:|---:|---:|
| `pellet` (post-9b0a880) | 195 | 21.33% | 1.5% | 27.0% | 396.0 | 10.28% |
| `random` (pre-9b0a880) | 205 | **27.39%** | 3.5% | **37.0%** | 362.4 | 10.27% |

`random` reproduces the baseline (27.39 vs 27.50); `pellet` reproduces the
soak (21.33 vs 21.31). Training win rates match to four decimals.

**Mechanism.** 9b0a880 rested on the premise that continuous linear Q-values
effectively never tie. Measured on the converged soak policy, **1.7% of
multi-action decisions are exact ties** (1.5% two-way, 0.2% three-way). This is
structural, not incidental: of nine features, `features[0]` (bias) and
`features[3]` (pellet distance) are action-independent by construction and
therefore cancel in the argmax, and most of the remainder are binary
indicators — two actions sharing a (blocked, is-pellet-dir, ghost-bucket,
is-reverse) profile get *exactly* equal Q. Resolving those ties
deterministically locks recurring states into cycles. The signature is visible
in three independent places: average eval episode length rises 362 → 396 while
median pellets-left *improves* 42.6 → 39.9 (it gets closer and then fails to
finish); checkpoints below 5% rise from 0.33% to 1.25%; and the worst
checkpoint of all five seeds is the identical value 0.0138 — when the policy
degenerates, the deterministic tie-break decides everything, so the eval
trajectory stops depending on the training seed at all.

**Dated correction.** The 2026-07-27 entry above concluded that adopting the
shared deterministic tie-break for both agents was correct, and recorded
"training behavior remains unchanged" — which is true and is precisely why the
cost went unnoticed. That entry stands as written; this supersedes its
decision for the linear agent only. The T4 evidence behind `pellet` (+44%
greedy score, 799.7 → 1152.1) was measured on the *tabular* agent and does
transfer to it; it does not transfer to an agent with no visit counts and a
dense, mostly-binary action-conditioned feature space.

**Change:** each agent now declares its own `defaultEvalTieBreak` — `pellet`
for tabular, `random` for linear — and `TrainingController.evaluate()`, the
bench, and App.tsx watch mode all resolve through it rather than a single
shared constant. The option remains honored on both agents; only the default
moved. A regression test pins both defaults. `run-soak.sh` now reconstructs
its aggregate from `episodes.csv` when `summary.json` is missing and reports
on SIGINT/SIGTERM, so an interrupted run is no longer unreportable.

**Failures / surprises:** the soak was designed to measure the tail and
instead caught a six-point regression that four prior documented runs had
already been silently scored against. The held-out panels turned out *not* to
be where the surprise lived — panel spread is only 19.4–21.8%, so the
historical single panel was never badly unrepresentative. The infrastructure
built for one question paid off by answering a different one.

**Decision:** revert the linear evaluation default to `random`; stop soaking
the linear agent, since it converges in ~2,000 episodes and more compute is
provably not the lever; move to capacity work.

**Lesson:** a refactor justified as "interface parity" or "correctness" still
changes results if it touches the policy, and it must be benchmarked like any
tuning change. The tell we had and ignored: the commit changed what the greedy
argmax returns. Anything that changes action selection is an experiment, not a
cleanup — and evidence for one agent is not evidence for another that happens
to share the interface.

### 2026-07-28 — D11 feature capacity: two variants, both regressions, one bug fix kept

**Context:** With duration ruled out as a lever and the tie-break regression
fixed, the roadmap's next item was feature capacity. The 2026-07-28 census had
measured 1.7% of multi-action decisions as exact Q-value ties, and inspection
showed why: of nine features, `features[0]` (bias) and `features[3]` (pellet
distance) were action-independent and cancelled in the argmax, and
`features[1]` ("moves into a wall") was constant 0 on every input `act()` ever
sees, because `act()` only scores `getLegalActionIndices()` — which is exactly
the unblocked moves.

**Hypothesis:** features that describe each candidate move distinctly — a
per-action pellet distance, a pellet on the destination tile, dead-end and
escape-breadth indicators — would cut the tie rate and raise the win rate.

**Change / experiment:** two variants, each measured at seed 7, 8 min,
`evalEvery=500`, random tie-break, against the restored v4 baseline of 27.39%.

| features | α | mean eval WR | max | <5% ckpts | train WR |
|---|---|---:|---:|---:|---:|
| v4 baseline | 0.02 | **27.39%** | 37.0% | 1 | 10.27% |
| v5 attempt 1 (*replaced* f1 and f3) | 0.02 | 22.63% | 45.0% | 18 | 11.07% |
| v5 attempt 1 | 0.01 | 23.59% | 36.0% | 5 | 10.90% |
| v5 attempt 2 (*superset*, 12 features) | 0.02 | **0.99%** | 32.0% | 163 | **12.06%** |
| v5 attempt 2 | 0.01 | 0.60% | 27.0% | 162 | 11.66% |

Attempt 1 was confirmed across five seeds: 22.63/21.88/22.51/22.69/21.60%,
mean 22.2%. Not seed noise.

**Result:** both regressions, the second catastrophically. Attempt 2 opens at
64/200 — the best first checkpoint of any run in this experiment — then
converges by ~26k episodes to a stable attractor it never leaves: **0 wins,
`p5 = 2` pellets left, every checkpoint to the end.** It clears the whole maze
except the last two pellets and cannot finish.

**Mechanism.** `PELLET_SEARCH_RADIUS` is 12. When two pellets remain they are
almost always farther than that, so every pellet-distance feature returns its
"none in range" sentinel of 1.0 and carries no gradient — the agent is blind in
exactly the states that decide the win. v4 had one such feature; attempt 2 had
two, and doubling the saturating weight mass was enough to make "wander safely
forever" outrank "go find the last pellet". The 12.06% *training* win rate
against ~1% greedy eval is the same fact from the other side: at ε=0.05 the
random 5% eventually stumbles onto the last pellets, and the learned policy
never does.

**Failures / surprises — two reasoning errors, the second worse:**

1. Attempt 1's regression was attributed to α. Lowering α did cut collapsed
   checkpoints 18 → 5, so instability was real, but it left ~4 points
   unexplained. That was a partial cause reported as a diagnosis.
2. Attempt 2 was justified with the claim that a strict superset "cannot be
   worse in representational terms". True about capacity, and irrelevant:
   adding a *correlated feature that saturates in the states that decide the
   outcome* changes what is learned. An assumption was stated as a guarantee,
   and it was wrong by 26 percentage points.

The features did do what they were built to do — tie rate fell 1.7% → 1.12%,
training win rate rose, and attempt 1 reached a 45% peak checkpoint where v4
never exceeded 37%. None of that survived contact with the endgame.

**Decision:** revert the feature set to v4 (9 features) and remove `actionInfo`
and the per-direction BFS along with their throughput cost. **Keep** the
tunnel-aware `wallMask` fix — an unrelated genuine bug found along the way, in
which `wallMask` probed raw `pac.x + dx` while the env's `canMove()` wraps
through `nextPosition()` first, so at a tunnel mouth both agents were told
"wall" about the one legal move that crosses the maze. `OBSERVATION_KEY_VERSION`
9→10 and `FEATURE_SCHEMA_VERSION` 4→5: the feature *set* is v4, but
`features[1]` now reads a corrected mask, so old weights must not load.
Verified at seed 7: **27.79%** mean eval wins, back at baseline (27.39%); the
+0.4 is within single-seed noise and the runs had different CPU contention, so
the fix is neutral-to-slightly-positive, not an improvement.

**Lesson:** "this change strictly adds information" is a statement about
representational capacity, not about what gradient descent will do with it. The
question that mattered was not *can* the model express a better policy — it
demonstrably could, at a 45% peak — but *what does this feature do in the states
that decide the outcome*. Saturating features are worst precisely where the
task is hardest, and adding a second one doubled a blind spot instead of
widening the view. Check a proposed feature's behavior at the endgame, not at
the opening.

**Follow-on (measured, not theorized):** the agent provably stalls at two
pellets when they sit beyond the search radius. The pellet *horizon*, not the
feature count, is the next target — this is the concrete form of the roadmap's
Root cause A. Any future action-conditioned pellet feature should wait until the
horizon is fixed.

---

## Current open thread

The baseline is the D8/D9 linear agent with T7 far-pellet direction at 35.17%
pooled mean evaluation wins over five training seeds and four held-out panels.

Two things are now settled and should not be re-litigated:

- **Training duration is not a lever.** The agent converges in ~2,000 episodes;
  6.5M more changed nothing. Cap linear runs at minutes. Any future claim that
  a change needs a long run to show its effect needs to explain why this
  result does not apply.
- **The single fixed evaluation panel was not the problem.** Panel spread is
  19.4–21.8%. Keep evaluating on the four panels — it is nearly free and it is
  how the worst-panel target is stated — but do not expect held-out panels to
  be where the remaining gap lives.

A third thing is now settled: **D11's feature set is not a lever.** Both
variants regressed, and the 2026-07-29 fallback rescue still collapsed with
zero sentinel observations. Do not restore it unchanged.

The pellet horizon was a real binding constraint and is now resolved. The
revised sequence:

1. ~~Add multiple held-out evaluation seed panels.~~ Done 2026-07-27
   (`evalPanels`, `scripts/run-soak.sh`); soak run and analyzed 2026-07-28.
2. ~~Enrich the action-conditioned features.~~ Attempted 2026-07-28; both
   variants regressed and were reverted. The tunnel-aware wall-encoding half
   was a real bug and was kept.
3. ~~**Extend the pellet horizon (roadmap T7).**~~ Done 2026-07-29: the
   far-direction fallback improved pooled five-seed/four-panel greedy wins
   25.54% → 35.17%. The D11 rescue still failed, so its collapse was not caused
   by the horizon alone.
4. Sweep target synchronization, L2 regularization, and TD-error clipping for
   tail stability, plus an annealed endgame curriculum and `gamma=0.995`.
5. Add gated n-step credit assignment (roadmap T1).
6. If the horizon fix plateaus, escalate capacity: roadmap T5 (coarse position
   in the key) then T6 (DQN over the board).

T7 exceeded all prior success targets: 35.17% mean evaluation wins, at least
32.06% on every seed's worst held-out panel, and 30.75% checkpoint fifth
percentile. I1 should now pin regression floors against this baseline before
the next tuning item.

### 2026-07-29 — T7 experiment decision: isolate far-pellet direction first

**Context and hypothesis:** D11 localized a plausible endgame blind spot:
`nearestPelletDir` becomes 4 and `nearestPelletDist` becomes 13 whenever the
nearest pellet is beyond the radius-12 BFS horizon. The strongest evidence is
the reverted superset's stable `0 wins, p5 = 2` attractor, but that result does
not by itself prove that a wider horizon improves the restored v4 baseline.
The falsifiable hypothesis is narrower: restoring the direction to the nearest
far pellet will improve greedy endgame routing without requiring new features.

**Decision:** preserve radius 12 as the fast path. On a miss, continue the same
BFS until the nearest reachable pellet is found. Return its direction and true
depth, but keep the existing linear distance normalization capped at 13. This
isolates the action-conditioned direction signal; changing the radius constant
would also rescale the state-only distance feature and make the result
ambiguous. Do not restore D11's per-direction BFS.

**Validation plan:** first land and run CI against the restored baseline/session
survivors. Add an eval census of sentinel-direction observations by
`pelletsRemainingBucket`, regression tests for far/near/tunnel/unreachable
pellets, and a focused observation-throughput benchmark. Then run a sequential,
fixed-episode seed-7 v4 A/B on the historical panel. Advance only if reachable
sentinels disappear, throughput remains at least 90% of baseline, mean greedy
win rate rises by at least 2 points, and pellet-left median/tail do not worsen.
A passing candidate gets the established five-seed/four-panel short
confirmation; it does not get an overnight soak because the agent converges by
about episode 2,000.

**Failure interpretation:** if v4 is neutral but the baseline census confirms
frequent endgame sentinel states, rerun the reverted D11 superset only as a
mechanism probe. The fallback must rescue the `p5 = 2` collapse. Failure to do
so falsifies the current single-cause explanation; demote T7 and investigate
the correlated-feature/TD dynamics instead of widening the search further.
Training win rate alone is explicitly not an acceptance metric.

**Result / artifacts:** planning decision only; no implementation or training
run yet. The executable gates are recorded in `ROADMAP.md`. No measured result
is claimed.

**Reusable lesson:** when a representation change and a normalization change
can be separated, test them separately. A failure mode is evidence for a
mechanism only if the proposed fix removes that failure under a controlled
comparison.

### 2026-07-29 — T7 result: far direction wins; D11 has a second failure mechanism

**Context and falsifiable hypotheses:** The predeclared T7 screen asked whether
continuing the existing BFS beyond radius 12 would improve restored v4 without
rescaling its distance feature. A second probe asked whether that change alone
would rescue D11's stable `0 wins, p5 = 2` attractor.

**Exact change / experiment:** `PELLET_SEARCH_RADIUS=12` remains the linear
normalization cap. The BFS now continues the same queue to the nearest
reachable pellet after a radius-12 miss, returning the far direction and true
depth; `min(dist, 13) / 13` is unchanged. Evaluation gained overall and
per-pellet-bucket sentinel rates. The screen used four seed-7 cells at 60k
episodes and one 200-game panel every 500 episodes: v4 bounded/fallback and a
reconstruction of D11 bounded/fallback. The original D11 source was never
committed or staged, so the probe recreated the recorded effective 12-feature
superset rather than claiming byte identity.

The passing v4 screen then advanced to a matched confirmation: bounded and
fallback, each with seeds `{7,1007,2007,3007,4007}`, 20k episodes, four disjoint
200-game panels every 2,000 episodes, `endgameCurriculum=0.90`,
`stepPenalty=-0.02`, `alpha=0.02`, random eval ties, and target sync 2,000.

**Measured result — v4:** the seed-7 screen improved mean greedy wins
27.75% → 39.46%, median pellets left 43.4 → 34.2, and sentinel rates
15.72% overall / 58.13% in bucket 0 → zero, with no observed throughput loss.
The matched five-seed confirmation repeated the result:

| metric | bounded | fallback |
|---|---:|---:|
| pooled mean eval wins | 25.54% | **35.17%** |
| seed mean range | 24.22–27.99% | **33.72–36.79%** |
| minimum worst-panel mean | 22.61% | **32.06%** |
| mean checkpoint p5 | 17.95% | **30.75%** |
| training win rate | 9.65% | **68.69%** |
| overall / bucket-0 sentinel | 15.30% / 57.12% | **0% / 0%** |

All five fallback seeds beat all five bounded seeds. All predeclared gates
passed, including the ≥2-point mean gain, tail improvement, sentinel removal,
and no material seed regression.

**Measured result — D11 rescue:** failed. Bounded/fallback all-checkpoint mean
greedy wins were 1.02%/2.16%. After episode 26k they were 0.015%/0%; the
fallback had zero sentinel observations but 64/65 checkpoints still had
`p5=2`. Training wins rose from 14.54% bounded to 42.35% fallback while greedy
eval stayed collapsed. The probe reproduced the attractor but did not rescue
it.

**Failures, surprises, correction:** the plan treated rescue as conditional on
v4 being neutral, but the user explicitly requested the mechanism test and it
was worth running despite v4's large win. More importantly, the 2026-07-28
entry's claim that the agent collapsed because it "cannot see" the last pellets
was too singular. The baseline really was blind often enough for the fallback
to add 9.63 points, but zeroing that blind state did not rescue the reconstructed
D11 features. Correlated feature weights and linear TD dynamics are a second
cause. Finding #14 remains as the original reasoning; Finding #15 records this
dated correction.

**Decision:** promote the far-direction fallback. Bump
`OBSERVATION_KEY_VERSION` 10→11 and `FEATURE_SCHEMA_VERSION` 5→6. Remove all
D11 reconstruction code after the experiment; do not restore that feature set
unchanged. Keep the sentinel census in the eval CSV as a regression metric.

**Validation and artifacts:** raw cells:
`bench-out/20260729-t7-ab/`; matched confirmations:
`bench-out/20260729-011217-t7-bounded-confirm/` and
`bench-out/20260729-010944-t7-fallback-confirm/`. The far/near/tunnel/no-pellet
observation behavior has regression coverage. Full repository validation is
recorded after the final source cleanup.

**Reusable lesson:** removing a measured blind spot can be a large production
win without explaining every failure first attributed to that blind spot.
Separate “this defect matters” from “this defect is the sole cause.”

### 2026-07-29 — I1 deterministic learning smoke pins the T7 baseline

**Context and falsifiable hypothesis:** T7 established a strong five-seed
baseline, but no CI check would detect a later change that perturbed seeded
training, the observation key, rewards, or evaluation. The hypothesis was that
a fixed single-worker linear run would converge quickly enough to serve as a
repeatable, bounded regression fixture.

**Change / configuration:** added `scripts/learning-smoke.sh` and its assertion
helper. Each invocation runs the promoted configuration twice — linear, two
ghosts, seed 7, 2,000 episodes, `endgameCurriculum=0.90`,
`stepPenalty=-0.02`, `alpha=0.02`, target sync 2,000 — and evaluates four
disjoint 50-game panels. It compares `evals.csv` byte-for-byte and summaries
after removing elapsed wall time, then checks floors of 60/200 wins, 30% mean
win rate, 18% worst panel, and `pl_p5=0` per panel. CI calls it through
`npm run test:learning-smoke`.

**Validation / measured result:** the calibration produced 67/200 wins, 33.5%
mean evaluation wins, a 20.0% worst panel, and `pl_p5=0` on all panels. The
second run matched evaluation and summary outputs exactly. The full unit suite,
typecheck, lint, and build remained green after integration.

**Decision:** I1 is complete. These are intentionally conservative regression
floors, not a claim that 33.5% is a new long-run target; subsequent tuning must
still use the five-seed/four-panel confirmation protocol.

**Reusable lesson:** a short deterministic run can protect a long-run result if
its configuration, measurements, and acceptance thresholds are all explicit.

### 2026-07-29 — T2 reward/discount sweep improves the promoted linear baseline

**Context and falsifiable hypothesis:** Despite T7 restoring the direction to
far pellets, the terminal reward was still discounted by `gamma=0.99`, late
pellets still risked a -100 death, and their shaped reward topped out at 6×.
The hypothesis was that a longer linear discount horizon and a less negative
endgame trade-off would improve greedy clears without changing the observation
or model.

**Exact change / experiment:** added first-class `winBonus`, `deathPenalty`,
and `pelletEscalationMax` bench arguments, with the cap now an environment and
reward-preset parameter. `scripts/t2-reward-sweep.sh` screened all 36 cells:
`gamma={0.99,0.997,0.999}`, `winBonus={1000,2500,5000}`,
`deathPenalty={-100,-50}`, and cap `{6,10}`. Each used seed 7, 2,000 episodes,
the restored linear/T7 settings, and four 50-game panels. The screen winner
was `gamma=0.997`, `winBonus=1000`, `deathPenalty=-50`, cap 10 (36.0% mean
greedy wins versus 33.5% bounded baseline).

**Validation / measured result:** a matched confirmation ran the five
established seeds, 2,000 episodes each, and four 200-game panels. Baseline
mean was 33.25% (30.75–37.0% by seed; 29.5% minimum worst panel). The candidate
mean was **37.17%** (37.0–37.87%; **32.5%** minimum worst panel). Every
candidate seed beat its corresponding baseline. Training win rate declined
slightly (26.80% → 26.17%), so it was not used as the promotion metric; greedy
evaluation and panel robustness improved.

**Failure / surprise:** the first screen completed all training cells but its
report writer exited because an `awk` process-substitution output had no final
newline under `set -e`. The raw artifacts were intact; a one-cell aggregation
smoke exposed the issue, and the final screen was rerun after adding the
newline. This is a harness failure, not an experiment result.

**Decision:** promote linear `gamma=0.997`, shared `deathPenalty=-50`, and
`pelletEscalationMax=10`; retain `winBonus=1000`. Update GUI, bench, and
environment defaults together. T1, gated n-step returns, is next; it must add
its own first-class CLI knob as part of I2.

The I1 smoke was rerun against the promoted defaults and stayed reproducible,
now recording 72/200 wins (36.0% mean) with the same 20.0% worst panel and
`pl_p5=0` guard.

**Reusable lesson:** a larger training-win count can coexist with a worse
greedy policy, and vice versa. Promote only against the metric the user sees:
held-out greedy evaluation, with the weakest panel visible.

### 2026-07-29 — T1 n-step returns regress at the promoted baseline

**Context and falsifiable hypothesis:** Terminal wins remain sparse even after
T2, so a short n-step backup might propagate endgame reward faster than the
one-step update without changing observations or rewards. The hypothesis was
that at least one of n=3, 5, or 10 would exceed the n=1 baseline at equal
training compute and four held-out panels.

**Exact change / experiment:** added a shared terminal-flushing n-step buffer
to tabular and linear Q-learning, exposed `nStep` in the bench, and added
`scripts/t1-nstep-sweep.sh`. The screen fixed the promoted linear/T2 settings,
seed 7, 2,000 episodes, and four 50-game panels while varying only
`nStep={1,3,5,10}`. Unit tests cover discounted terminal returns, suffix
flushing, delayed updates, and the default one-step path.

**Validation / measured result:** n=1 produced 36.0% mean greedy wins and a
20.0% worst panel. n=3 reached 31.5% and 22.0%; n=10 reached 29.5% and 20.0%;
n=5 collapsed to 0.0% with mean `pl_p5=223.3`. None improved mean wins or the
pellet tail, so none met the confirmation gate. Artifacts:
`bench-out/20260729-201510-t1-nstep-screen`.

**Decision:** retain `nStep=1` as the default and do not run a five-seed
confirmation. T1 is complete as a negative result; proceed to T3
potential-based shaping. The existing linear `lambda` remains L2
regularization, not an eligibility-trace setting.

**Reusable lesson:** n-step backups are not automatically helpful in an
off-policy linear TD system. Screen the credit-assignment horizon independently
before coupling it with reward or representation changes.

### 2026-07-29 — T3 potential-based pellet-progress shaping is neutral or worse

**Context and falsifiable hypothesis:** T1 left the one-step learner unchanged,
but endgame credit could still benefit from a dense reward that preserves the
underlying objective. The hypothesis was that a terminal-zeroed potential over
pellet progress would improve held-out greedy clears without altering the
optimal policy.

**Exact change / experiment:** added optional
`γΦ(s') − Φ(s)` to the environment, where
`Φ(s)=-scale·pelletsLeft/totalPellets`; terminal Φ is zero. The default scale
is zero, `shapingGamma` is explicit, and a unit test verifies discounted
telescoping across distinct completed toy routes. The T3 screen fixed the
promoted linear/T2 baseline, seed 7, 2,000 episodes, and four 50-game panels,
then varied only scale `{0,25,100,250}` at γ=0.997.

**Validation / measured result:** scale 0 yielded 36.0% mean greedy wins and a
20.0% worst panel. Scale 25 tied it exactly; scale 100 lowered mean to 33.0%
(though its weakest panel was 30.0%); scale 250 lowered mean to 32.5%. Training
wins rose modestly with scale (27.0% → 28.45%) but did not predict greedy
quality. Artifacts: `bench-out/20260729-202137-t3-potential-shaping-screen`.

**Decision:** retain `potentialShapingScale=0`; no candidate earns five-seed
confirmation. T3 is complete as a negative result. Move to T5 coarse Pac-Man
position, which targets the remaining observation aliasing directly.

**Reusable lesson:** policy invariance does not guarantee an optimization gain.
The shaping term can be mathematically safe while still changing the learning
dynamics unfavorably in finite-budget linear TD.

### 2026-07-29 — T5 coarse Pac-Man regions improve tail distance but not policy quality

**Context and falsifiable hypothesis:** the tabular key still aliases local
states in different maze areas. The hypothesis was that a 3×3 Pac-Man region
would reduce that aliasing enough to improve greedy evaluation at a practical
state-table cost.

**Exact change / experiment:** added `pacRegionGrid` to the environment and
bench. Grid 1 emits the baseline region 0; grid 3 packs a row-major 3×3 region
into the v12 tabular key. The screen ran two-ghost tabular training from scratch
for 20,000 episodes, seed 7, four 50-game panels, and otherwise identical
curriculum settings.

**Validation / measured result:** grid 3 cut mean held-out `pl_p5` from 128.375
to 88.825 and produced one training win, versus none for grid 1. It produced
zero greedy wins in every panel, exactly like grid 1, while populated Q states
grew 32,008 → 54,981 (+72%). Artifact:
`bench-out/20260729-225926-t5-pac-region-screen`.

**Decision:** retain `pacRegionGrid=1`; no candidate met the greedy-policy
promotion gate, so no five-seed confirmation is warranted. T5 is a negative
result with a useful diagnostic signal. The remaining roadmap option is T6:
full-grid DQN/CNN research, which needs an explicit architecture choice before
implementation.

**Reusable lesson:** reducing a diagnostic tail is not enough if the learned
greedy policy never converts it into wins. Include representation capacity and
the cost of state-space growth in the promotion decision.

### 2026-07-29 — Next strategy: isolate full-grid CNN Double DQN research

**Context:** T1 n-step returns, T3 potential shaping, and T5 coarse tabular
position were implemented and screened after the T2/T7 linear promotion. None
improved greedy policy quality enough to replace the 37.17% five-seed linear
baseline.

**Decision:** begin T6 as a separate full-grid CNN Double-DQN track, retaining
the linear agent and deterministic smoke as the production control. The first
implementation is a six-plane fixed-board encoder and a small two-convolution
network with replay, legal-action masking, Huber loss, and a target network;
the shared browser/headless agent is required before learning experiments.

**Predeclared gate:** seed-7 curves at 2k/10k/50k episodes on four held-out
panels must exceed 37.17% mean greedy wins while preserving a 32.5% worst-panel
floor before five-seed confirmation. This is a capacity experiment, so longer
compute must be reported alongside throughput and memory rather than assumed
to be comparable to the 2k-episode linear convergence budget.

### 2026-07-30 — T6 runtime decision: shared pure-JavaScript TensorFlow.js

**Context and hypothesis:** T6 needs one trainable CNN implementation for both
the Vite browser application and the Node headless benchmark. The hypothesis is
that the pure-JS TensorFlow.js package provides that shared surface without a
native binary installation path.

**Decision / validation:** added `@tensorflow/tfjs` v4.22.0 and a small runtime
wrapper that waits for backend initialization. Its unit test confirms a backend
is selected and a tensor operation executes under the existing Node test
runner. The browser build is the companion compatibility check.

**Trade-off:** `tfjs-node` can accelerate offline Node work but cannot be the
shared Vite runtime and adds platform-specific TensorFlow binaries. Use the
portable package first; record backend, steps/sec, and tensor memory in every
T6 run. Escalate to a separate optional acceleration path only if measured
headless throughput blocks the predeclared experiment gates.

### 2026-07-30 — T6 CNN Double-DQN foundation is isolated and testable

**Context and falsifiable hypothesis:** compact observations, reward changes,
and n-step backups did not improve the promoted linear policy. The hypothesis
for this first T6 slice is narrower: a full-board CNN agent can be represented
and updated deterministically in the chosen shared runtime before any expensive
environment experiment is attempted.

**Exact implementation:** added a 28×31 six-plane encoder (walls, pellets,
power pellets, Pac-Man, dangerous ghosts, edible ghosts), a fixed-capacity
copying replay buffer, legal-masked Double-DQN bootstrap helper, and a
16/32-channel 3×3 CNN with a 128-unit dense head, Huber loss, Adam, and a
target network. The production linear trainer is unchanged.

**Validation:** tests verify padding/plane placement, deterministic replay
sampling, legal-action masking with online selection plus target evaluation,
and that a repeated terminal batch lowers real CNN Huber loss. The Node CPU
backend completed the loss test but was much slower than ordinary unit tests,
which confirms the need for the predeclared throughput gate.

**Decision:** do not yet run a learning curve or change defaults. Build the
headless CNN runner next, with explicit backend, steps/sec, and tensor-memory
metrics; only then execute the 2k/10k/50k four-panel gate.

### 2026-07-30 — T6 headless runner exposes CPU-throughput blocker

**Context and falsifiable hypothesis:** the CNN primitives needed an
environment-integrated runner before a learning curve could be trusted. The
hypothesis was that the pure-JS runtime could at least execute the same
environment/agent/replay path and report whether its throughput supports the
predeclared curve.

**Exact implementation:** added `scripts/cnn-bench.ts` and `npm run bench:cnn`.
It trains only the isolated CNN agent, writes four-panel-compatible `evals.csv`,
and records backend, environment steps/sec, update count/rate, loss, and tensor
memory in `summary.json`. It supports zero-eval throughput smoke runs so runner
validation does not accidentally become a policy experiment.

**Measured result:** the no-update smoke was correct at 1,169.7 environment
steps/sec. Enabling exactly one batch-1 gradient update reduced end-to-end
throughput to **1.1 environment steps/sec** on the Node CPU backend. Artifacts:
`bench-out/cnn-runner-smoke` and `bench-out/cnn-runner-update-smoke`.

**Decision:** the runner is complete, but CPU is not viable for a 2k/10k/50k
curve. Do not claim a policy result or switch defaults. The next T6 decision is
to measure a portable accelerated backend (WASM or browser WebGL) against this
same runner contract; do not add a native-only path unless that comparison is
explicitly approved.

### 2026-07-30 — Portable T6 acceleration gate fails on WASM and WebGL

**Context and hypothesis:** Node CPU measured 1.1 environment steps/sec with a
real CNN update. The next hypothesis was that a portable TensorFlow.js backend
could make the same agent practical without introducing a native-only training
path.

**Exact experiment:** added the TensorFlow.js WASM backend and runner backend
selection before model construction. WASM was run through the identical
one-episode, batch-1 update smoke. On failure, added a query-gated browser
WebGL micro-benchmark (`?cnnWebglBenchmark=1`) that performs the same synthetic
terminal CNN update and reports backend, update rate, loss, and tensor memory.

**Measured result:** WASM failed before a result because its
`Conv2DBackpropFilter` training kernel is not registered. The interactive
browser benchmark selected WebGL successfully but measured **0.15 updates/sec**
with finite loss 0.1250 and 34 tensors, slower than the Node CPU result.

**Decision:** no portable backend supports a practical T6 learning curve in the
current runtime. Keep the 37.17% linear policy unchanged and do not run the
predeclared 2k/10k/50k curves. Any next T6 move needs explicit approval for a
native accelerator or external training workflow; it is not authorized by the
current isolated browser/headless scope.

### 2026-07-30 — T6 acceleration follow-up identifies a benchmark and model-size confound

**Context and falsifiable hypothesis:** the portable gate recorded one WebGL
training update at 0.15 updates/sec. The follow-up hypothesis is that this is a
valid cold-start failure but not enough evidence to establish sustained WebGL
throughput, because TensorFlow.js compiles WebGL shaders lazily and the current
network has a disproportionately large dense head.

**Investigation:** reviewed the benchmark timing boundary and model shapes. It
starts timing before the first update and stops after that update, so shader
compilation, initial weight upload, GPU readback, and optimization are all
amortized over one sample. The two same-padded convolutions preserve the
31×28 grid; flattening 32 channels produces 27,776 inputs to a 128-unit dense
layer. That layer has 3,555,456 parameters including bias, versus only 5,520
parameters in both convolutions and the four-action output combined. Online and
target models total 7,122,984 parameters before Adam optimizer state.

**Measured result:** no new performance number was produced. The WASM result
remains conclusive for this implementation because its registered kernel set
lacks `Conv2DBackpropFilter`. The existing WebGL result remains valid only as
cold, batch-1 latency; warmed multi-update throughput is still unknown.

**Decision:** keep linear in production and keep the policy curves paused.
Before introducing a native or external workflow, run a corrected portable
diagnostic with disposable warm-up, repeated timed updates, realistic batches,
and kernel profiling, and screen strided convolution or pooling before the
dense head. If that still misses the declared curve budget, the preferred
recovery is an optional native TensorFlow.js training entry point, followed by
external Python/GPU training only if native TensorFlow.js is operationally
unsuitable.

**Reusable lesson:** a one-iteration GPU benchmark measures initialization plus
work, not steady-state throughput. Always report cold latency and warmed
throughput separately, and inspect parameter concentration before concluding
that the backend is the sole bottleneck.

### 2026-07-30 — Corrected T6 benchmark recovers WebGL throughput; readback is now the limit

**Context and falsifiable hypothesis:** the 0.15 updates/sec WebGL result timed
one cold batch-1 update on a 7.12M-parameter online/target pair. The hypothesis
was that spatial downsampling, on-device target/loss construction, and warmed
multi-batch timing would materially improve portable throughput without
changing the six-plane observation or promoting the CNN prematurely.

**Exact implementation:** both 3×3 convolutions now use stride 2, reducing the
flattened representation from 31×28×32 (27,776 values) to 8×7×32 (1,792).
Each model has 235,540 parameters rather than 3,561,492. `trainBatch()` packs
states into preallocated `Float32Array` buffers, builds legal-masked Double-DQN
targets as tensors, and applies Huber loss only to the selected actions. It
removes three full `tensor.array()` readbacks and retains one scalar loss
readback. The development benchmark now runs one first update, two disposable
warm-ups, one `tf.profile()` update, and 30 timed updates at batches 1, 16, and
64. It reports updates/sec, samples/sec, per-kernel timing, readback time, tensor
memory, and renderer identity. An explicit WebGPU backend option runs the same
complete update. `App.tsx` gates the entire panel import behind
`import.meta.env.DEV`, so lazy benchmark code is not emitted in production.

**Validation and artifacts:** the focused CNN/runtime tests, lint, typecheck,
and production build passed. The build emitted only `index-DIUaXvDp.js` and its
CSS—no benchmark or TensorFlow.js chunk. The fresh-browser benchmark ran at
`http://127.0.0.1:5173/?cnnWebglBenchmark=1` on Chrome/ANGLE reporting
`NVIDIA GeForce RTX 4080 Laptop GPU`, not SwiftShader.

| Batch | First update | Warm updates/s | Warm samples/s | Profile kernels | Scalar readback |
|---:|---:|---:|---:|---:|---:|
| 1 | 6,263.3 ms | 8.32 | 8.3 | 10.5 ms | 119.6 ms |
| 16 | 3,116.6 ms | 8.34 | 133.4 | 22.9 ms | 108.2 ms |
| 64 | 4,270.2 ms | 8.31 | 531.9 | 11.9 ms | 106.6 ms |

A repeat after shader caching held 8.34–8.38 updates/sec and showed first-update
latency near 114–120 ms, confirming why a fresh page/backend is required for
the cold number. WebGPU backend initialization failed before the update because
the current Electron Chrome returned a null GPU adapter; therefore its gradient
compatibility remains unmeasured rather than failed.

**Failures, regressions, and surprises:** after the three large readbacks were
removed, the remaining scalar `loss.data()` synchronization dominated wall
time by roughly an order of magnitude over profiled kernels. Updates/sec stayed
nearly flat by batch while samples/sec scaled 64×. The WebGPU experiment could
not reach the intended kernel smoke on this browser. `npm install` also
reported the repository's existing audit state (nine vulnerabilities); no
automatic audit fix was applied because that would be unrelated and potentially
breaking.

**Decision:** the old 0.15 updates/sec result is superseded and WebGL is no
longer categorically blocked. Do not add native TensorFlow.js yet and do not
promote the CNN. Next measure the complete browser environment/inference/update
wall clock with batch 64, then run the smallest learning gate only if that
budget is practical. Keep linear in production until the CNN exceeds 37.17%
mean wins and the 32.5% worst-panel floor. If portable end-to-end training
misses its gate, add optional offline `tfjs-node`; reserve native GPU or an
external Python learner for a further measured need.

**Reusable lesson:** once large readbacks are removed, even a scalar metric can
serialize GPU work and dominate a JavaScript training loop. Always report both
updates/sec and samples/sec, and treat shader caches as process-global when
interpreting “cold” latency.

### 2026-07-31 — Development-toolchain audit remediation

**Context and falsifiable hypothesis:** npm audit reported nine findings (one
critical) in the Vite/Vitest development stack. The hypothesis was that a
coordinated toolchain upgrade plus compatible transitive pins would clear the
report without changing application behavior.

**Exact implementation:** upgraded Vite to 8.2.0, Vitest to 4.1.10, Vite Node
to 6.0.0, the React Vite plugin to 6.0.5, ESLint to 10.8.0, TypeScript-ESLint
to 8.65.0, and the hooks plugin to 7.1.1. Overrides pin `js-yaml@4.3.0` and
patched brace-expansion versions; Vite 8's Node 20.19+ requirement is declared.

**Validation:** full and production-only audits report zero findings. Lint,
all tests, typecheck, build, and the profiled CNN smoke passed.

**Decision:** retain the explicit versions and overrides until upstream ranges
make the pins unnecessary; this changes tooling security, not training defaults.
