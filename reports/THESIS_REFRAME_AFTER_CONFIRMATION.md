# PCCD thesis reframe after independent confirmation

Date: 2026-07-16  
Authority: temporary delegated scientific decision under user authorization  
Trigger: `CORE_NOT_ESTABLISHED` on the single locked confirmation unseal

## Decision

The previous thesis — “policy adaptation causes mean calibration degradation in
a frozen safety critic” — is retired. It was strong on the discovery data but
did not replicate on the independent 3,500-family lockbox and new D5 training
seed.

The paper is repositioned as a preregistered stress test and measurement study:

> **Average calibration can remain acceptable, or even improve, while
> individual safety criteria undergo large and reproducible changes in opposite
> directions. Aggregate ECE is therefore not a criterion-wise deployment
> certificate for a frozen safety critic.**

This statement must be qualified in the manuscript:

- the mean-degradation hypothesis was not established;
- the criterion interaction was preregistered but confirmatory gatekeeping was
  not reached because P2-C failed;
- the pattern is strong secondary evidence, stable across two D5 training
  seeds on one lockbox;
- prevalence/support sensitivity prevents calling it a pure intrinsic
  calibration-map failure;
- all safety states are relative to a fixed Qwen2.5-32B reference annotation
  protocol until the blinded human audit is complete;
- evidence comes from one Qwen2.5 policy/critic/teacher ecosystem.

## Evidence hierarchy for the paper

1. **Primary confirmatory result — non-confirmation.** The independent mean
   ΔECE is `−0.00618`, CI `[−0.01105, 0.00009]`; the discovery mean-degradation
   effect does not generalize to this lockbox/seed.
2. **Replicated anchor.** Independent mean D0 ECE is `0.03943`, CI
   `[0.03787, 0.04554]`, within the locked 0.05 region.
3. **Registered secondary interaction.** Cross-criterion SD is `0.02483`, CI
   `[0.02094, 0.02897]`, with omnibus p=`1e-4`; H2/H4 move upward while
   H5/T1/T2 move downward relative to the mean.
4. **Seed stability.** Old/new D5 ΔECE ordering has Spearman `ρ=0.952`, CI
   `[0.806, 0.988]`.
5. **Boundary results.** FN asymmetry, scalar KL prediction, source-only
   recalibration, and the original low-shot tolerance claims remain frozen
   negatives or partial findings. P8-C was not reached, not failed.
6. **Construct-validity dependency.** The 800-cell human audit is the only
   remaining authorized empirical dependency.

## Recommended manuscript framing

Working title:

> **When Average Calibration Hides Safety-Criterion Shifts: A Preregistered
> Stress Test of Frozen Critics under Policy Adaptation**

Abstract-level claims may say:

- the preregistered independent test did not reproduce global calibration
  degradation;
- nevertheless, the same test revealed large oppositely signed criterion
  changes with high cross-seed stability;
- global ECE can mask criterion-level risk redistribution;
- scalar shift magnitude and source-only recalibration were unreliable in the
  earlier registered grid;
- the work provides a transparent evaluation protocol and negative-result map.

The abstract must not say:

- adaptation generally degrades mean calibration;
- P3-C was confirmed;
- hidden violations create FN-dominant failures;
- P8 or matrix scaling failed on the confirmation set;
- teacher labels are ground-truth safety;
- criterion shifts are proven to be intrinsic calibration-map deformation
  rather than prevalence/support changes.

## Publication outlook

Without the human audit, the current work remains approximately
**borderline/weak-reject for AAAI main track**: the experimental discipline is
strong, but the primary positive discovery did not confirm, the remaining
interaction is gate-qualified, and external validity is limited.

If the blinded audit finds no consequential domain×criterion differential
reference error, the paper becomes a credible negative/measurement contribution
with an unusually clean independent lockbox and a compelling “aggregate
cancellation” result. That can be competitive as an empirical AI-safety or
evaluation paper, though acceptance is not assured. If the audit finds
differential reference error, the paper should pivot further toward
LLM-reference measurement failure rather than frozen-critic calibration.

## Execution boundary

The one-unseal rule is final. Do not test another adapter point, objective,
metric, lockbox, calibrator, response-length correction, or threshold as a
replacement in this project phase. Complete the frozen human audit, then write
the manuscript around the full positive, null, and negative evidence together.
