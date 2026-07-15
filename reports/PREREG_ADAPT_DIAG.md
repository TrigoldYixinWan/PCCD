# PCCD — Post-Adaptation Heterogeneity Diagnostic (single point, before locking G1/G2)

Pre-registered 2026-07-16 by PaperGuru (human-approved: "see how post-adaptation
heterogeneity manifests before finalizing the G1 framework"). This is a DIAGNOSTIC to
decide whether per-policy heterogeneity — which L3 did NOT find on the base D0 critic — is
instead a property of the ADAPTED-distribution degradation, as our hypotheses (P2/P5/P6)
predict. It produces NO gate verdict; it informs the locked G1 reframing and the full G2
pre-registration that follow.

## Why this diagnostic (the L3 reinterpretation being tested)
D0 results: P1 strongly supported (mean ECE 2.9%, base well-calibrated); L3 FAILED
(ten-head violated-F1 all 0.75-0.95, CV=0.081, whole CI below 0.15). Reading: the base
critic is UNIFORMLY good across policies — which is exactly what a well-calibrated base
critic (P1) should look like. Our thesis does NOT claim the base critic is heterogeneous;
it claims degradation UNDER LOCAL ADAPTATION is per-policy heterogeneous and FN-asymmetric
(P2/P3/P5/P6). So heterogeneity should appear in the D0->D_k DELTA, not in the D0 level.
This diagnostic tests that directly at ONE adaptation point before we commit the framework.

## The pipeline this exercises (not yet implemented — build minimally)
The study design (plan_9day.md) is: a POLICY model (Qwen2.5-7B, an INDEPENDENT instance —
same base family as the critic backbone but a separate model that generates text) is
adapted along D0..D6; it GENERATES responses to a fixed prompt set; the FROZEN D0 critic
scores those responses; we compare the critic's per-policy behavior on the adapted
output distribution vs the base (D0) distribution. Steps to build for ONE point:
1. Adapt the policy model at a single mid-grid point: **D3 = LoRA r=8** (plan_9day.md grid).
   Adaptation target/data: a small alignment fine-tune (e.g. DPO/SFT on a held-out slice)
   sufficient to shift the output distribution measurably; record exactly what it is.
2. Generate policy responses on a FIXED held-out prompt set (disjoint from critic train;
   reuse the test-split prompts so the base vs adapted comparison is paired by prompt).
3. Compute KL(adapted || base) on that prompt set (the P6 x-axis), same estimator we will
   use for every D-point later — define and record it now.
4. Score BOTH the base (D0) and adapted (D3) policy outputs with the FROZEN D0 critic; get
   teacher labels on the same adapted outputs as ground truth (label-only teacher, fixed
   prompt — the reliable oracle from L1).
5. Measure per-policy: ECE, violated-F1, and the FN vs FP split, on base vs adapted; report
   Delta_ECE, Delta_FN, Delta_FP per policy, with bootstrap CIs.

## What the diagnostic must answer (decides the G1/G2 framing)
- Is the DEGRADATION per-policy heterogeneous? i.e. does Delta_ECE (or Delta in violated-F1)
  vary across the ten policies far more than the base F1 did (base CV was 0.081)? Report the
  CV of per-policy Delta_ECE and compare to the base-level CV.
- Is there any sign of FN-asymmetry (Delta_FN > Delta_FP) even at this single point? (Not a
  G2 verdict — just a directional read.)
- Does a single adaptation point move the distribution enough (KL > 0) for the critic to
  degrade at all? If D3 barely moves KL, we may need a stronger point (D5/D6) for the real G2.

## Locked boundaries for this diagnostic
- The FROZEN D0 critic is used as-is (read-only checkpoint); it is NEVER retrained here.
- Teacher stays label-only, fixed prompt (the L1 oracle).
- This is ONE point (D3) for concept validation; it does NOT pre-empt the full D0-D6 grid,
  which will be pre-registered as G2/G3 AFTER this diagnostic informs the framing.
- No gate PASS/FAIL is declared. Results feed: (a) a locked G1 reframing (whether L3 becomes
  a base-homogeneity check supporting P1, with heterogeneity moved to the adapted delta), and
  (b) the full G2 pre-registration (metric = per-policy Delta_ECE/FN-asymmetry vs KL).

## Deliverable
reports/day5_adapt_diag.md: the D3 adaptation spec + KL(adapted||base); per-policy base-vs-
adapted ECE / violated-F1 / FN / FP with bootstrap CIs; CV of per-policy Delta_ECE vs the
base F1 CV; a directional FN-asymmetry read; and a recommendation on whether D3 is a strong
enough point or the grid should be anchored at a higher-KL point for G2. PaperGuru then
locks the G1 reframing + full G2 pre-registration.
