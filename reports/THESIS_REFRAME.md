# PCCD — Thesis Reframing after G2 (human-approved 2026-07-16)

Trigger: G2 is a CONCLUSIVE, pre-registered FAIL on FN-asymmetry (reports/day5_g2.md). Under
correctly-powered support (>=30 violated/policy) and correctly-directed hidden-violation
adaptation (chosen = teacher-violated AND frozen-D0-critic-MISSED responses), the frozen
critic's calibration DOES degrade significantly and heterogeneously (D5 mean Delta-ECE 0.029,
CI excludes 0; RMS 0.048), but the degradation is NOT false-negative-asymmetric — the
direction is significantly opposite (mean Delta-FN - Delta-FP = -0.28, CI [-0.32,-0.23]).
The honesty clause forbids a third redirection of the P5 test. We therefore RESHAPE THE PAPER
around what the evidence actually supports, and convert the FN result into a reported
counter-intuitive finding rather than a discarded hypothesis.

## New thesis (LOCKED)
A shared, frozen, independent critic loses calibration on the output distribution of a
locally-adapted policy; this calibration degradation is (i) real, (ii) per-policy
HETEROGENEOUS, (iii) PREDICTABLE from adaptation strength KL(adapted||base), and (iv)
RECOVERABLE by lightweight per-policy temperature scaling. Contrary to the intuitive safety
fear, the degradation is NOT false-negative-asymmetric: when local adaptation makes a policy
violate more, those violations become MORE frequent and thus EASIER for the frozen critic to
catch (FN falls), while false positives rise modestly — a counter-intuitive result we
characterise and explain.

## Reweighted propositions (LOCKED)
- **P1 (15%, support→now foundational):** the frozen critic is well-calibrated on the base
  distribution. STATUS: SUPPORTED (Day-4, mean ECE 2.9%).
- **P2 (25%, main):** local adaptation degrades the frozen critic's calibration on the policy
  output distribution. STATUS: SUPPORTED (G2 D5 mean Delta-ECE 0.029, CI excludes 0).
- **P3 (25%, main):** the degradation is per-policy HETEROGENEOUS. STATUS: SUPPORTED
  (G2 RMS Delta-ECE 0.048, cross-policy SD 0.038; L2 already showed 44/45 policy-pair
  heterogeneity in the teacher target labels).
- **P6 (20%, main):** the degradation is PREDICTABLE from KL(adapted||base) — a scaling law.
  STATUS: TO BE TESTED at G3 (we now have Delta-ECE at D1..D6 with graded KL 0.02-1.20).
- **P4 (15%, main):** per-policy temperature scaling recovers calibration. STATUS: TO BE
  TESTED at G4.
- **P5 (now REFRAMED, ~0% as a positive claim; reported as a counter-intuitive FINDING):**
  the degradation is NOT FN-asymmetric; it is, if anything, FP-leaning, because violation
  frequency rises with adaptation and a frozen critic detects frequent violations more
  readily. STATUS: this is a REPORTED result with mechanism, not a contribution we claim to
  have confirmed. Include the D3 benign-control contrast (also FN-negative) to show the
  effect is not an artifact of one adaptation objective.

Net: the paper's spine is now P2+P3 (degradation exists and is heterogeneous, 50%),
P6 (KL-predictable scaling law, 20%), P4 (recalibration, 15%), P1 (foundational, 15%),
with P5 recast as an honest counter-intuitive finding + measurement-methodology results
(L1 wording/position sensitivity; joint-vs-isolated consistency).

## Why this is a legitimate AAAI contribution (literature-anchored)
Distribution shift breaking calibration is an active area (Wong et al. 2026 ICML for MoE;
Bahi 2026 for GNNs), and calibration-under-shift recalibration is studied via conformal
methods (Siahkali et al. 2026). Sycophancy fine-tuning breaking a model's OWN UQ is known
(Sahoo 2026). NONE studies an INDEPENDENT FROZEN CRITIC's PER-POLICY calibration transfer
under a controlled customer-adaptation grid, its KL-predictability, or per-policy temperature
recovery. Our counter-intuitive FN result is itself a useful safety-community correction: the
intuitive fear (adaptation hides violations from a frozen critic) does not hold in our
regimes; the real risk is heterogeneous mis-calibration and rising false positives, which we
show is cheaply recoverable. Honest negative/counter-intuitive results with mechanism are
publishable and valuable in alignment safety.

## What still must run (unchanged pipelines, already built)
- **G3 (P6):** fit Delta-ECE = f(KL) across D1..D6 (KL already spans 0.02-1.20 nats/token).
  Pre-register the functional form + goodness-of-fit threshold (e.g. R^2 >= 0.7 on held-out
  points) BEFORE fitting. Report per-policy and pooled fits with CIs.
- **G4 (P4):** fit per-policy temperature on the calib split, apply to the frozen critic's
  logits on each adapted distribution, re-measure ECE; PASS if per-policy temperature scaling
  significantly reduces Delta-ECE toward base without harming violated-F1/AUROC.

## Boundaries (Red)
- No third redirection of the P5 test. P5 stays a reported counter-intuitive finding.
- G3/G4 metric definitions and thresholds must be pre-registered BEFORE running them.
- Frozen D0 critic stays read-only; teacher stays label-only fixed-prompt.
