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

---

## SECOND REFRAME after G3+G4 both FAIL (human-approved 2026-07-16)

G3 and G4 are both CONCLUSIVE pre-registered FAILs (reports/day6_g3.md, day7_g4.md), executed
cleanly (D0 hash unchanged, 0 optimizer failures, 0 bound hits, no threshold relaxed). We do
NOT attempt to rescue P4/P5/P6; all three stay as pre-registered negative results and their
verdicts are frozen. Final status of the original propositions:
- P1 SUPPORTED (base well-calibrated, natural test ECE 2.9%).
- P2 SUPPORTED (adaptation causes significant calibration degradation, D5 mean dECE 0.029).
- P3 SUPPORTED (per-policy heterogeneous; G3 per-policy slopes range +0.13 to -0.04 — strong
  heterogeneity is itself confirmed by the fact that a single KL law cannot fit all policies).
- P5 NOT SUPPORTED, direction opposite (dFN-dFP = -0.28) — reported counter-intuitive finding.
- P6 NOT SUPPORTED: KL has real aggregate association (positive slope, permutation p=0.024)
  but insufficient out-of-point prediction (LODO R^2 0.63 < 0.70); objective and policy
  identity materially condition the response.
- P4 NOT SUPPORTED: source-calib temperatures preserve discrimination but do NOT transfer to
  recover calibration on adapted distributions (D5 mean recovery -0.001; they even worsen the
  support-enriched D0 reference).

### FINAL THESIS (LOCKED)
"Local policy adaptation induces heterogeneous calibration-transfer failure in a frozen,
independent critic; neither the adaptation KL magnitude nor a source-fitted temperature map is
a reliable universal predictor or remedy." The paper is a systematic, pre-registered CAUTIONARY
measurement study that falsifies three convenient deployment assumptions (critic stability,
KL-predictability, source-only recalibration transfer), each with a locked negative result and
a mechanism, anchored by the positive P1/P2/P3 findings and the L1/L2 measurement-methodology
findings.

### Writing rules (LOCKED — avoid over-claiming failure)
Frame the negatives precisely, NOT as "insufficient experiments" or "temperature scaling
doesn't work":
- "KL magnitude is insufficient as a predictor ACROSS adaptation objectives" (not "KL is
  irrelevant"): report the positive slope + permutation significance alongside the LODO miss.
- "Source-only recalibration does not transfer" (not "temperature scaling fails"): it DOES
  preserve discrimination and improves source NLL; the failure is transfer.
- "Calibration failure and discrimination failure are distinct" (argmax/F1/AUROC preserved
  while ECE degrades).
- "Adaptation objective and target-distribution evidence are necessary variables."
- ALWAYS distinguish the natural base test (ECE 2.9%, P1) from the support-enriched G2 D0
  distribution (ECE 4.95%); do not let the enriched number undermine the P1 premise.

### Two bounded follow-up experiments (new, narrower propositions; do NOT touch frozen gates)
1. PRIORITY — Low-shot target-aware recalibration (new proposition P7, own pre-registration
   reports/PREREG_G5.md): G4 showed ZERO target labels cannot recalibrate. P7 asks the
   NECESSARY-CONDITION question: how many target labels does it take? On NEW, untouched target
   calib/test splits per adapted distribution, measure a learning curve at 50/100/200/500
   labels, comparing source-T (G4 baseline), target-global-T, target-per-policy-T, and
   hierarchical shrinkage (per-policy shrunk toward a global target-T). If a small budget
   recovers calibration, the paper regains a "problem -> boundary -> actionable remedy" arc.
   This is NOT a rerun of G4; it tests the narrower P7 and cannot change the G4 verdict.
2. Minimal external replication: reproduce ONLY P2/P3 (calibration degradation + heterogeneity)
   with a SECOND critic or policy backbone (not the full D1-D6 grid), so a single Qwen
   critic/policy pair is not a reviewer's "n=1" objection.

If resources allow only one, do #1 (cheapest, most likely to restore actionability). All
existing FAILs are retained as pre-registered negatives; new experiments test only the new
narrower proposition and cannot alter any prior verdict.

### Boundaries (Red, second reframe)
- P4/P5/P6 verdicts are FROZEN. No rescue, no threshold change, no re-run to pick results.
- P7 (low-shot) uses NEW untouched target splits; it must not reuse G2/G4 eval labels to fit.
- Frozen D0 critic read-only; teacher label-only fixed-prompt throughout.
