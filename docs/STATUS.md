# Current Status

**Updated:** 2026-08-17 · **Branch:** `codex/t6-browser-trainer-smoke` · **Commit:** `375c485`

> Compiled from repository evidence on 2026-08-17. Nothing was built or run.
> This file is a **snapshot and a router** — the substance lives in the
> documents linked under Deep context, which it does not replace.

## Objective

Browser-based Pac-Man with an in-browser Q-learning training lab. No backend.
The point is training quality and reproducible evidence of improvement, not the
game.

## Current state

- Structural/refactor backlog (A1–A5, B1, B2) **done**.
- The historical tabular 2-ghost track plateaued near **2.5% greedy-eval win
  rate**, `p5 ≈ 55`. D8 (action-conditioned linear features) and D9 (target
  network) broke that ceiling.
- T7 (far-pellet direction) raised a matched **five-seed, four-panel mean from
  25.54% to 35.17% linear eval wins**, seed means 33.72–36.79%.
- End-to-end CNN trainer smoke added (2026-08-13), with inference warmed before
  smoke timing.

## Active work

CNN trainer smoke and browser trainer work on `codex/t6-browser-trainer-smoke`.
3 modified tracked files and 1 untracked pending commit.

## Next

Per [`ROADMAP.md`](../ROADMAP.md), which is scoped entirely to *making training
better and validating gains reproducibly*. Specific items are not restated here
— read the roadmap rather than trusting a summary of it.

## Blockers

None identified from repository evidence.

## Known problems

Not enumerated here. Two dated reviews exist —
[`CODE_REVIEW_2026-05-17.md`](../CODE_REVIEW_2026-05-17.md) and
[`CODE_REVIEW_2026-07-21.md`](../CODE_REVIEW_2026-07-21.md) — plus
[`SYSTEM_ASSESSMENT_2026-05-27.md`](../SYSTEM_ASSESSMENT_2026-05-27.md). Which
findings remain open is **unverified**.

## Validation state

| Check | Status |
|---|---|
| Matched five-seed, four-panel evaluation | **Real multi-seed evidence.** Recorded in `test_history.md`. This repo does not generalise from single runs. |
| CNN trainer end-to-end smoke | Added 2026-08-13. Not run in this session. |
| Test suite | Present. **Not run in this session** — the device bridge cannot execute builds. |

## Unverified

- Whether the suite passes at `375c485`.
- Which findings from the two code reviews and the system assessment are still open.
- Whether the 9.6 GB on disk (largest repo in the workspace) is tracked content
  or generated training artifacts.

## Recent decisions

Recorded as narrative rather than ADRs. [`ENGINEERING_JOURNAL.md`](../ENGINEERING_JOURNAL.md)
is the canonical reasoning trail — explicitly including failed experiments and
superseded conclusions. No `docs/decisions/` series exists; the journal serves
that role and is not being replaced by one.

## Deep context

**These documents are authoritative for their subjects. This file only points at them.**

| Topic | Document | Role (as its own authors describe it) |
|---|---|---|
| Reasoning trail | [`ENGINEERING_JOURNAL.md`](../ENGINEERING_JOURNAL.md) | "what we believed, what we changed, what failed, what the evidence showed" |
| Experiment tables and metrics | [`test_history.md`](../test_history.md) | read *Current State* first, then *Findings*, to avoid repeating settled questions |
| Current priorities | [`ROADMAP.md`](../ROADMAP.md) | training quality and reproducible validation |
| Refactor-era history | `archive/DEEP_DIVE_2026-05-30.md` | superseded era, retained |
| Reviews | [`CODE_REVIEW_2026-05-17.md`](../CODE_REVIEW_2026-05-17.md), [`CODE_REVIEW_2026-07-21.md`](../CODE_REVIEW_2026-07-21.md) | point-in-time |
| System assessment | [`SYSTEM_ASSESSMENT_2026-05-27.md`](../SYSTEM_ASSESSMENT_2026-05-27.md) | point-in-time |
| Agent instructions | [`AGENTS.md`](../AGENTS.md) | canonical |
