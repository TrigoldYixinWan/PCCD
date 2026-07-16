# G6/P8 preregistration lock blocker (Red)

Date: 2026-07-16

Branch: `day9/g6-matrix`

## Status

Execution stopped before implementing or fitting G6. No source-calib lambda
selection was run, no P7 target manifest or row was loaded, and no target
temperature/matrix scaler, bootstrap, metric, or plot was computed.

PR #7 was merged as authorized (`ba2f37d`). The merged P8 preregistration is
commit `8317f8d`, file `reports/PREREG_G6.md`.

## Red blocker

The handoff message describes P8 as preregistered and directs execution, but the
authoritative merged file is explicitly not locked:

- title: `DRAFT for human lock`;
- line 5: `Not locked until PaperGuru approves.`

The same file states that changing the parameterization, splits, budgets,
verdict rule, or overfitting guard after lock is Red. Codex cannot convert a
draft into a locked preregistration or infer outcome-sensitive choices on
PaperGuru's behalf.

## Items requiring an outcome-blind human lock

The following details must be made unambiguous before any source-lambda
selection or target access:

1. **Lambda search and ownership.** Specify the candidate lambda set (or a
   continuous selection rule), whether lambda is shared across all policies,
   and whether diagonal and full matrix scalers receive one common lambda or
   separately source-selected lambdas.
2. **Objective scaling.** Specify whether the optimized 3-way NLL is summed or
   averaged before adding L2. This changes the effective lambda. Define the
   identity penalty explicitly, e.g.
   `||W-I||_F^2 + ||b||_2^2`, including any coefficient normalization.
3. **Source LOO unit and aggregation.** Specify whether leave-one-out removes a
   source prompt (ten policy cells together) or a policy cell, and how LOO NLL
   is aggregated across folds/policies for lambda selection.
4. **Numerical stability rule.** `Clip to keep the map numerically stable` needs
   fixed bounds or an equivalent deterministic constraint before fitting.
5. **Paired P7 comparison.** Specify whether P7 per-policy-T is refit inside
   every G6 calibration-bootstrap replicate using the same sampled rows (the
   natural paired comparison) or treated as a fixed frozen point estimate.
6. **Verdict partition.** Section 5 currently overlaps:
   - PARTIAL applies when conditions (a) and (c) hold but the ECE ceiling (b)
     is missed and the method improves over P7;
   - NEGATIVE says no budget recovers to the base regime, which is also true
     whenever (b) is missed.

   Lock a mutually exclusive rule, including whether PARTIAL requires the
   paired improvement over P7 to have a 95% CI excluding zero.

The descriptive threshold for flagging a “large” target-calib/test gap should
also be fixed if the report will use a categorical overfitting flag; otherwise
the report can present the gap without a post-hoc flag.

## Required resolution

PaperGuru should commit an updated `reports/PREREG_G6.md` marked `LOCKED` with
the approval date and the points above resolved. Once that commit is pulled,
execution can resume on this branch without touching any frozen prior verdict.

## Frozen boundaries preserved

- P4/P5/P6/P7 verdicts unchanged.
- Frozen D0 critic unchanged and not loaded.
- Frozen D5 teacher labels/logits unchanged and not loaded.
- P7 TARGET-CALIB/TARGET-TEST manifests unchanged and not read.
- No teacher, critic, policy, or adapter call was made.
