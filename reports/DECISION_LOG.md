# PCCD delegated-authority decision log

Status: APPEND-ONLY from 2026-07-16. Existing entries may only be corrected by a
new entry that names the superseded entry; they must not be rewritten after a
new response, label, logit, or aggregate result is inspected.

This log begins when the project owner temporarily delegated PaperGuru's
scientific decision authority to Codex. It records changes to the research
mainline, evidentiary status, and locked protocols. It does not alter any
previously frozen result.

## DL-001 — delegated authority and integrity boundary

- Date: 2026-07-16
- Authority: project owner, explicit delegation in the research handoff
- Decision: Codex may resolve design-level (formerly Red) choices until
  PaperGuru returns, provided every change is logged here before the affected
  outcome is observed.
- Integrity boundary: "recover a positive result" means create a fair,
  independently testable opportunity for a useful claim. It does not authorize
  changing old thresholds, relabeling old outcomes, repeated confirmatory runs,
  selective reporting, or optimizing against a confirmatory test set.
- Frozen findings: P1-P7 and L1-L3 retain their reported verdicts. In
  particular, P5 remains a conclusive failure/opposite-direction result and may
  not receive a third directional reformulation; G3/P6 and P4/P7 remain failed
  under their registered criteria.
- Reversibility: the delegation may be ended by the project owner or PaperGuru,
  but decisions already followed by outcome access remain part of the audit
  trail.

## DL-002 — AAAI-first thesis and claim hierarchy

- Date: 2026-07-16
- Evidence considered before this decision: frozen reports through Day 8;
  `PCCD_AAAI_Core_Review.md`; `PCCD_mechanism.md`; Berta et al. (2025/2026)
  Structured Matrix Scaling; the existing implementation and artifact audit.
- Primary thesis: **good source-distribution calibration is not a transferable
  deployment certificate after policy-model adaptation; the resulting
  calibration drift is criterion-specific in the tested PCCD setting.**
- Primary confirmatory claims:
  1. P2-C: target adaptation increases frozen-critic miscalibration on a new,
     outcome-blind prompt lockbox and an independently trained adaptation seed.
  2. P3-C: that increase differs materially across the ten policy criteria.
- Auxiliary claim: P8-C tests whether a pre-specified, low-shot,
  target-aware structured calibrator repairs that drift without harming
  discrimination.
- Supporting/limiting findings: reference-annotator wording/order sensitivity,
  the opposite-direction P5 result, the failure of scalar divergence to meet
  the registered prediction standard, and the limits of source-only/temperature
  recalibration remain fully reported. They are not rewritten as confirmations.
- Mechanism status: nonlinear-manifold, OOD-distance, and heavy-tail accounts
  remain hypotheses unless a new pre-specified probe establishes association;
  no causal mechanism claim is authorized from the existing data.
- Rationale: this hierarchy makes the already-supported calibration-transfer
  problem the paper's center, while making repair a value-adding but
  nonessential endpoint. It is robust to a further P8 failure.

## DL-003 — old target test becomes development evidence

- Date: 2026-07-16
- Decision: the P7 `TARGET-TEST` and every D0-D6 artifact derived from the
  3,000-prompt G2 evaluation set are henceforth a **development/discovery set**
  for any new method. They remain valid for their original frozen tests, but
  cannot independently confirm P8 or the reframed P2/P3 thesis.
- Reason: P7 outcomes and subsequent method discussions have already used that
  target test. Running another selected repair on it is a registered reanalysis,
  not fresh generalization evidence.
- Consequence: a publishable positive P8 claim requires a new prompt-family
  lockbox whose split is frozen before response generation, reference labeling,
  critic scoring, or method evaluation.

## DL-004 — supersede the unexecuted G6 draft

- Date: 2026-07-16
- Decision: `reports/PREREG_G6.md` remains preserved as an unexecuted draft and
  is superseded by `reports/PREREG_CONFIRMATION.md`. The blocker recorded in
  `reports/CHANGES/2026-07-16_g6_prereg_lock_blocker.md` was correct at the time;
  the owner's later delegation supplies the previously missing human authority.
- Method correction: `diag(a) z + b` is called **vector scaling**, not structured
  matrix scaling. P8-C uses the published `probmetrics==1.3.0` Structured Matrix
  Scaling (SMS) implementation with its frozen defaults as primary and
  Structured Vector Scaling (SVS) as secondary.
- Regularization correction: source-only lambda selection is abandoned for the
  new claim because P4 already showed that source-only recalibration lacks a
  target-shift signal. SMS/SVS use their pre-specified target-calibration
  regularization and preprocessing; no target-test tuning is allowed.
- Old-P8 status: an SMS/SVS run on old P7 data may be reported only as
  registered development evidence and software validation.

## DL-005 — independent confirmation unit and one-run rule

- Date: 2026-07-16
- Decision: build one 3,000-prompt outcome-blind lockbox, split by prompt family
  into 1,000 TARGET-CALIB and 2,000 CONFIRM-TEST prompts before any generated
  response or label exists.
- Primary adapted policy: a single new D5 LoRA-r32 training run with seed
  `20260723`, otherwise byte-for-byte protocol-equivalent to the frozen D5 run
  and using the same frozen 512-pair training corpus. It is trained once.
- Secondary policy: the old D5 adapter evaluated on the new prompts, separating
  prompt generalization from training-seed generalization.
- Generation/reference protocol: D0, old D5, and new-seed D5 use seed
  `20260723`; the unchanged Qwen2.5-32B label-only protocol is described as a
  **fixed reference annotation protocol**, not unquestioned ground truth.
- One-run rule: CONFIRM-TEST is analyzed once under the locked code and
  thresholds. Failure is final for P2-C/P3-C/P8-C in this project phase. No
  replacement lockbox or altered direction is permitted after outcome access.
- Schedule: this confirmation phase explicitly extends beyond the original
  nine-day construction schedule. The extension is for independent validation,
  not for reopening frozen gates.

## DL-006 — human construct-validity audit

- Date: 2026-07-16
- Decision: before aggregate confirmation metrics are opened, freeze a blinded,
  outcome-stratified audit packet of prompt-response-policy cells. Annotators see
  only prompt, response, canonical policy, and the three-way rubric; they do not
  see domain, adapter, source, reference label, critic prediction, or confidence.
- Analysis: two independent annotators per cell, adjudication of disagreements,
  inverse-probability weighting back to the lockbox population, and
  prompt-family-clustered uncertainty. Human labels validate the reference
  protocol only; they may not fit P7/P8.
- External dependency: completion requires human annotators. Code may prepare
  and freeze the packet but cannot substitute model labels for this audit.

## DL-007 — outcome-blind capacity audit expands the lockbox

- Date: 2026-07-16
- Supersedes: the `3,000 = 1,000 + 2,000` size in DL-005; every other DL-005
  boundary remains active.
- New decision: use 4,000 families split into 500 TARGET-CALIB and 3,500
  CONFIRM-TEST, with exact source/stratum counts in
  `reports/PREREG_CONFIRMATION.md`.
- Evidence available before any new outcome: the read-only capacity audit found
  89,784 unused canonical natural prompts after excluding all 42,234 prompts
  consumed by Day-2 and the entire G2 candidate/support-selection process. GPU
  and storage cost for the additional test prompts is negligible.
- Rationale: the largest locked low-shot budget is 500, so allocating additional
  prompts to calibration provides no primary-method information. Putting them
  in the sealed test set increases precision without increasing the number of
  fitted methods, thresholds, or attempts.
- Independence qualification: all original source families have been touched
  previously and no vendor semantic-family ID exists. The protocol therefore
  freezes an exact five-word-shingle Jaccard family graph and claims lexical
  near-duplicate separation, not untouched semantic-domain independence.

## DL-008 — P8 numerical-guard implementation clarification

- Date: 2026-07-16
- Timing: frozen before construction of the new lockbox, new adapter training,
  response generation, reference labeling, critic scoring, or confirmation
  aggregate access.
- Primary-fit guard: the 1% overall and 5% per-criterion bootstrap failure
  ceilings apply to the two fits required for the primary P8 comparison,
  published SMS and the paired P7 per-criterion-temperature comparator, at the
  500-prompt budget. An unresampled failure of either primary-required method is
  `NON_EVALUABLE`. SVS remains secondary, so its failure is reported but cannot
  invalidate an otherwise evaluable SMS-versus-temperature test.
- Fixed-criterion aggregation: a bootstrap replicate missing any of the ten
  primary criterion values is excluded as a whole rather than silently
  reweighting the remaining criteria. The registered 1% overall and 5%
  per-criterion fit-failure ceilings remain the evaluability thresholds; the
  count of complete paired replicates is reported for every primary contrast.
- Reference missingness: malformed or partial reference outputs remain missing
  cells and are never imputed. Critic logits are still scored from the preserved
  prompt/response. Domain-level strict ten-key success below 99% or any
  domain-by-criterion missing rate above 1% makes the affected package
  `NON_EVALUABLE`; otherwise metrics use the available cells under the paired
  family bootstrap.
- Discrimination interval: the simultaneous criterion-level AUROC guard uses a
  two-sided studentized bootstrap max-|t| interval. Its lower limits implement
  non-inferiority; its upper limits identify the registered
  `CONTRADICTED_OR_HARM` exit.
- Published-implementation constraint: `probmetrics==1.3.0` documents that
  `max_iter` and `tol` are ignored by its default BFGS SMS implementation.
  Therefore the preregistered higher-iteration retry is not available; no
  optimizer or penalty substitution is authorized.
- Verdict vocabulary: confirmation uses exactly `NOT_REACHED`,
  `NON_EVALUABLE`, `SUCCESS`, `CONTRADICTED_OR_HARM`, `PARTIAL_SUPPORT`, and
  `NOT_ESTABLISHED`. Runs on the consumed P7 data are marked
  `DEVELOPMENT_ONLY`, not assigned a confirmation verdict.

## DL-009 — process-safe deterministic P8 bootstrap execution

- Date: 2026-07-16
- Timing: resolved on the already-consumed P7 development split before any new
  lockbox response, label, logit, calibrator fit, or aggregate result.
- Finding: the published SMS/SVS implementation succeeds serially, but PyTorch
  autograd rejects calibration fits in workers created by plain `fork` after a
  parent-process fit. The initial parallel implementation therefore produced
  software failures, not scientific measurements.
- Decision: production P8 uses the `forkserver` multiprocessing start method
  (with `spawn` only as a non-Linux fallback). Each replicate constructs one
  deterministic PCG64 generator from `SeedSequence([20260724, replicate])`,
  draws its test-family resample, then its calibration resamples in ascending
  registered-budget order. Methods within a replicate share those exact rows.
- Validation: on consumed development data, 160 replicates across all four
  budgets with 16 workers completed with zero SMS/SVS/P7 fit failures. This
  changes only process orchestration and the now-explicit seed-to-resample map;
  registered budgets, replicate count, seed, estimands, methods, and thresholds
  are unchanged.

## DL-010 — restart frozen-critic scoring after a pre-inference software failure

- Date: 2026-07-16
- Timing: after response generation and reference-integrity checks, but before
  any confirmation logit or aggregate existed.
- Finding: the first scoring launch failed on both ranks before the first
  inference batch because the logits-only loader referenced the existing
  `LABEL_TO_ID` constant without importing it. No logits file was written.
- Decision: add the missing import and a complete-label regression test, then
  restart scoring from row one. Preserve the failed launch log.
- Boundary: the frozen critic, all inputs, split manifests, inference settings,
  metrics, thresholds, and seeds remain unchanged. This is a Green software
  correction, not another scientific attempt.
