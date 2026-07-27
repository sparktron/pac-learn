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

---

## Current open thread

The present baseline is the D8/D9 linear agent at 27.55% mean evaluation wins
over five training seeds. The next disciplined sequence is:

1. Add multiple held-out evaluation seed panels.
2. Correct tunnel-aware wall encoding and replace the mostly redundant blocked
   action feature.
3. Test an annealed endgame curriculum and `gamma=0.995`.
4. Sweep target synchronization, L2 regularization, and TD-error clipping for
   tail stability.
5. Add gated n-step credit assignment and richer action-conditioned features.

Success should mean better mean performance and better tails: target at least
32% mean evaluation wins, at least 25% on the worst held-out panel, and at least
15% at the checkpoint fifth percentile.
