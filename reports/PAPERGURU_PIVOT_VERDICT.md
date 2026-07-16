# PaperGuru Verdict on the Aggregate-Certificate Pivot: REVISE

Reviewer: PaperGuru (independent research supervisor). Date: 2026-07-16.
Scope: literature/feasibility audit requested in
`reports/PAPERGURU_REVIEW_AGGREGATE_CERTIFICATE_PIVOT.md`. This document changes NO frozen
verdict (P2-P8, G1-G6, CORE_NOT_ESTABLISHED all remain frozen), does not revive P5, does not
call the Qwen reference human ground truth, and does not re-analyze the same lockbox.

## VERDICT: REVISE (conditional path to GO; one literature check can force NO-GO)

The pivot is scientifically honest and targets the two real AAAI blockers (single-family
external validity; teacher construct validity). But its headline phenomenon — criterion-level
cancellation + mean/worst rank reversal — sits in a NARROW gap between two mature literatures,
and cannot be approved as-is. It is approvable only after the revisions below.

## 1. Novelty map (closest work, claim-by-claim)

Guard-calibration line:
- **Liu et al. 2024/ICLR 2025, "On Calibration of LLM-based Guard Models"** (arXiv:2410.10414):
  9 guards x 12 benchmarks; overconfidence; jailbreak-induced miscalibration; LIMITED ROBUSTNESS
  TO DIFFERENT RESPONSE-MODEL OUTPUTS; temperature + contextual post-hoc calibration. Its
  granularity is per-benchmark/task. The abstract does NOT mention criterion-BETWEEN
  cancellation, mean-vs-worst-criterion RANK REVERSAL, or a criterion-level deployment
  certificate. THIS IS THE PRIMARY COLLISION and must be resolved by reading the full paper.
- harsh2026benchmarking (14 guards), machlovi2025multiperspective, opina2026calibrated (calibrated
  tiered moderation) — all per-benchmark or per-model, none reported doing within-guard
  criterion-cancellation geometry.

Multicalibration / subgroup line (establishes the ELEMENTARY fact "global calib != subgroup calib"):
- Hébert-Johnson et al. 2018 (foundational MC); Wu et al. 2024 "Bridging Multicalibration and
  OOD Generalization"; Hansen et al. 2024 "When is Multicalibration Post-Processing Necessary?";
  Collina et al. 2026 "Sample Complexity of Multicalibration"; Terrance & Wu 2024 "Multi-group
  UQ for long-form text". These make subgroup-vs-global calibration a SOLVED framing; the pivot
  cannot claim it as novelty.

Verdict on novelty: the gap is REAL but NARROW. Guard-calibration work did per-benchmark, not
within-guard per-safety-criterion cancellation + selection consequence; multicalibration did
subgroup theory, not a safety-guard deployment certificate with cross-taxonomy replication. A
paper survives ONLY with a safety-specific increment beyond both (see §4).

## 2. Data/model feasibility (supports the pivot's external-validity fix)
- Human-labelled multi-category safety data EXIST and are usable: AEGIS2.0 (NAACL 2025),
  WildGuardTest, BeaverTails, Aegis-1.0. This is a genuine upgrade over the single Qwen closed
  loop and over the (still-pending) internal human audit.
- >=3 guard families with defensible probability outputs are available (Llama Guard, ShieldGemma,
  WildGuard, Aegis). Cross-family calibration is feasible.
- HARD CAVEAT (the pivot correctly flagged): public benchmarks have NO paired base/adapted
  responses. External validation can therefore support DISTRIBUTION / RESPONSE-SOURCE / BENCHMARK
  shift claims ONLY, never a policy-adaptation claim.

## 3. Claim boundary (locked)
- Generalizable with external human data: aggregate-vs-criterion calibration disagreement under
  distribution/response-source/benchmark shift.
- Supported ONLY by PCCD paired data (and already CORE_NOT_ESTABLISHED): any policy-ADAPTATION-
  specific statement. PCCD is a transparent hypothesis-generating case study, not confirmation.

## 4. Conditions to upgrade REVISE -> GO (ALL required)
1. **Read ICLR 2025 (arXiv:2410.10414) in full** — tables, appendix, code — and answer, in a
   committed note, the pivot's §8 Q1-Q5. If it ALREADY reports per-category cancellation AND
   mean/worst rank reversal for guard calibration => this pivot is **NO-GO** (too incremental).
2. State the novelty increment as NOT "cancellation exists" but: (a) an AUDITABLE criterion-level
   deployment CERTIFICATE with simultaneous/family-wise uncertainty for safety guards; (b) empirical
   evidence that mean-vs-worst-criterion selection CHANGES guard ranking/deployment choice;
   (c) CROSS-TAXONOMY / cross-family replication of the profile. Cancellation itself is an audit
   statistic (triangle inequality), never a theorem.
3. Lock the claim boundary to distribution/response-source shift (per §3); PCCD stays a case study.
4. Register ALL new primary hypotheses (pivot §7 H-Aggregate/H-Cancellation/H-Rank/H-Generalization)
   with thresholds, exclusions, support minima, multiplicity control, and taxonomy-mapping
   sensitivity, BEFORE inspecting any external outcome.

## 5. Minimum acceptance package (smallest persuasive-for-AAAI set)
- >=2 human-labelled benchmarks with different native taxonomies (e.g. AEGIS2.0 + WildGuardTest).
- >=3 guards spanning >=2 families and preferably >=2 scales, with defensible probability scores.
- >=2 registered source->target shifts per benchmark.
- Primary outputs per criterion/domain: ECE + simultaneous max-|t| CIs, worst-criterion drift W,
  heterogeneity H, cancellation C; mean-vs-worst guard RANKING comparison (the decision-relevant
  result); Brier/log-loss + adaptive-ECE sensitivity; class support + N/A prevalence beside every ECE;
  prevalence/support standardization sensitivity.
- An auditable "Criterion Calibration Stress-Test" protocol released as the artifact.

## 6. Compute estimate (order of magnitude, on the 2x RTX PRO 6000 box)
- Scoring only (no training): 3 guards x 2 benchmarks x ~5-30k items x forward passes. Guards are
  ~2-8B; batched inference ~ single-digit GPU-hours per guard-benchmark. Total well under ~1 GPU-day.
- No policy adaptation, no critic training on the external path -> cheap. The cost is protocol +
  careful per-taxonomy scoring/label alignment, not FLOPs.

## 7. Stop conditions (terminate the pivot)
- ICLR 2025 (or any 2024-2026 work found on full read) already reports the SAME criterion-level
  cancellation + mean/worst selection consequence for guards. -> NO-GO.
- Usable human datasets lack adequate per-category positive/negative support for calibration.
- Guard outputs cannot be turned into defensible probabilities.
- The effect disappears under human labels, support control, or alternative calibration metrics.
- The story would need post-hoc taxonomy grouping or selective domain reporting to survive.

## 8. Proposed titles
- Conservative: "Criterion-Level Calibration Reliability of LLM Safety Guards under Distribution Shift".
- Ambitious: "Aggregate Calibration Is Not a Safety Certificate: Criterion-Specific Reliability
  Shifts in LLM Guard Models" (only defensible AFTER cross-guard human-data replication + rank-reversal).

## 9. Next action (before any GO)
Produce `reports/ICLR2025_COLLISION_AUDIT.md`: full-paper read of arXiv:2410.10414 answering §8 Q1-Q5
with quoted evidence, plus a scan of harsh2026benchmarking / machlovi2025 / multicalibration-for-LLM
works for the same contribution. If the gap survives, draft `reports/PREREG_EXTERNAL_GUARD.md` with the
four registered hypotheses and the locked claim boundary. Only then does REVISE become GO.

## 10. One-line recommendation
The mean-degradation thesis is dead and must stay dead; the aggregate-certificate pivot is the best
remaining direction, but it is GO only if a full read of ICLR 2025 confirms the criterion-cancellation
/ mean-worst-rank-reversal gap is open AND the paper commits to a safety-specific auditable-certificate
increment on human-labelled, multi-guard data with a distribution-shift (not adaptation) claim boundary.
