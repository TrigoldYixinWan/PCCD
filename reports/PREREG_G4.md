# PCCD — G4 Per-Policy Temperature Recalibration Pre-Registration (DRAFT for human lock)

Drafted 2026-07-16 after the human-approved thesis reframe in
`reports/THESIS_REFRAME.md`. **This document is not locked until PaperGuru approves it.
No temperature fitting or recalibrated test metric may be computed before approval.**
After approval, changing the fit split, objective, temperature parameterization, primary
point, endpoint, margin, or verdict rule is Red.

## 1. Scientific question

G4 tests P4: whether a lightweight *per-policy* temperature map learned only on the frozen
base calibration split can recover calibration under policy adaptation, without updating
the critic and without damaging violation discrimination.

This is deliberately a source-calibration transfer test:

- ten scalars are fit once on Day-2 `calib`, one scalar per policy;
- no D1–D6 evaluation label is used to select or fit a temperature;
- the same ten frozen scalars are applied to D0 and every D1–D6 distribution;
- there is no D-specific or target-aware refit.

The D0 critic remains byte-for-byte frozen and read-only. Existing label-only teacher
labels are used; the teacher is not called again for D0–D6.

## 2. Frozen data and split discipline

### Fit split

- Labels: `$PCCD_OUT/labels/calib.jsonl` (1,000 examples).
- Logits: newly scored once by the frozen D0 critic after this protocol is approved.
- All ten 3-way labels, including N/A, participate in the fit.
- The calib labels/logits may be used only for temperature fitting and fit diagnostics.

### Evaluation distributions

- D0 and D1–D6 use the fixed 3,000 paired G2 prompt set and existing teacher labels/logits
  under `$PCCD_OUT/g2/`.
- No evaluation item may be moved to calib or removed.
- D2, D3 control, D4, and D5 are the **confirmed-degradation set**, fixed from G2 because
  their mean Delta-ECE CIs exclude zero above. D1 and D6 are negative/generalization
  controls and remain mandatory to report.
- D5 is the primary point because it is the highest-KL hidden-SFT point, has significant
  mean degradation, and has adequate violated support for all ten policies.

All input file hashes and the unchanged D0 manifest hash must be recorded before fitting.

## 3. Temperature parameterization and fit

For policy `p`, transform its three logits by one positive scalar:

```text
z'_ip = z_ip / T_p
T_p = exp(tau_p)
```

Fit `tau_p` independently for each policy by minimizing mean 3-way negative log likelihood
on the full calib split. Use deterministic bounded scalar optimization over
`tau_p in [log(0.05), log(20)]`, convergence tolerance `1e-8`, and no regularizer. The
optimizer and version must be logged. If the optimum is at a bound, keep it, flag it, and
do not widen the bounds after seeing evaluation results.

Forbidden alternatives include class-wise temperatures, vector/matrix scaling, policy
pooling, binning calibration, isotonic regression, target-point refitting, and selecting a
temperature by evaluation ECE.

Record each `T_p`, calib NLL before/after, convergence status, and bound hits.

## 4. Metrics

For every policy and D point, compute raw and temperature-scaled:

- primary calibration metric: the locked 15-bin top-class 3-way ECE including N/A;
- adaptive ECE using the Day-4 definition (secondary);
- violated-positive F1 on applicable items only (N/A excluded);
- violated-vs-satisfied AUROC on applicable items only;
- satisfied/violated/N/A support.

Temperature scaling uses no argmax-changing operation, so violated-F1 should be exactly
invariant except for numerical ties; this invariant must be checked rather than assumed.

Define, for each point `d` and policy `p`:

```text
gap_raw_dp    = abs(ECE_raw_dp    - ECE_raw_D0,p)
gap_scaled_dp = abs(ECE_scaled_dp - ECE_scaled_D0,p)
recovery_dp   = gap_raw_dp - gap_scaled_dp
absolute_gain_dp = ECE_raw_dp - ECE_scaled_dp
```

Positive `recovery` means the adapted distribution moved toward its correspondingly
scaled D0 calibration level. Positive `absolute_gain` prevents a misleading claim where
the gap shrinks only because both base and adapted ECE become worse.

## 5. Uncertainty and multiplicity

Use 10,000 two-stage paired bootstrap replicates with seed `20260721`:

1. resample 1,000 calib items with replacement and refit all ten temperatures;
2. independently resample 3,000 G2 prompt IDs with replacement, using the same evaluation
   indices for D0 and all D1–D6 points;
3. recompute raw/scaled metrics, gaps, recovery, F1, and AUROC.

Report percentile 95% CIs. Policies are fixed, not bootstrapped. For per-policy D5 recovery
tests, report one-sided p-values and Holm-adjust them across ten policies. Aggregate tests
are pre-registered separately and do not require a post-hoc policy subset.

If an adapted cell has fewer than 30 teacher-violated examples, its ECE remains valid, but
its F1/AUROC safety diagnostic is marked UNDERPOWERED and excluded from the aggregate
safety-preservation check. This currently applies to D6 S1/S3 and cannot be used to remove
their ECE results.

## 6. Primary D5 recovery test

At D5 report all ten per-policy temperatures, raw/scaled ECE, recovery, absolute gain,
F1, AUROC, and bootstrap CIs.

The D5 calibration component passes only if all hold:

1. mean over ten policies of `recovery_D5,p > 0`, with bootstrap 95% CI lower bound `> 0`;
2. mean over ten policies of `absolute_gain_D5,p > 0`, with bootstrap 95% CI lower bound
   `> 0`;
3. at least 7 of 10 policies have positive point-estimate recovery.

The D5 discrimination-preservation component passes only if both hold:

1. mean scaled-minus-raw violated-F1 has 95% CI lower bound `>= -0.005`, and the observed
   argmax predictions are exactly identical unless a numerical tie is documented;
2. mean scaled-minus-raw AUROC has 95% CI lower bound `>= -0.01`, and no individual policy
   has a point-estimate AUROC decline greater than `0.02`.

All D5 policies are adequately powered under the frozen G2 support table.

## 7. Across-distribution generalization

Apply the same ten temperatures to all six adapted distributions.

- For each point, report mean recovery, mean absolute gain, their CIs, number of policies
  improved, F1/AUROC changes, and per-policy tables.
- On the confirmed-degradation set D2/D3_control/D4/D5, pool the 40 fixed cells with equal
  weight and paired-prompt bootstrap. The generalization check passes if pooled mean
  recovery and pooled mean absolute gain both have 95% CI lower bounds `> 0`, and every one
  of the four points has positive point-estimate mean recovery.
- D1 and D6 are mandatory negative/generalization controls. Their inability to improve
  does not by itself fail G4 because G2 did not establish positive mean degradation there;
  any worsening must nevertheless be reported.

## 8. G4 verdict (locked on approval)

- **G4 PASS:** D5 calibration recovery passes, D5 discrimination preservation passes,
  and the confirmed-degradation-set generalization check passes.
- **G4 PARTIAL:** D5 calibration and discrimination pass, but the across-distribution
  generalization check fails.
- **G4 FAIL:** either D5 calibration recovery or D5 discrimination preservation fails.

No margin or policy-count threshold may be relaxed after evaluation. Adaptive ECE,
calib NLL, or a subset of favorable policies cannot rescue a primary FAIL.

## 9. Required outputs after approval

- `$PCCD_OUT/results/g4_temperatures.json` with temperatures, fit diagnostics, hashes,
  optimizer/version, and bootstrap metadata;
- `$PCCD_OUT/results/g4_recalibration.json` with all raw/scaled per-policy and aggregate
  metrics;
- reliability diagrams under `$PCCD_OUT/results/g4_figures/` and a small committed summary;
- `reports/day7_g4.md` in BRIEF §F.3 format;
- `logs/g4_fit.log` and `logs/g4_eval.log`.

## 10. Red boundaries

- Fit on Day-2 calib only; no G2 evaluation label may tune a temperature.
- Exactly one positive scalar per policy, shared across D0–D6.
- Keep N/A as a real third class during NLL fitting and ECE evaluation.
- Frozen D0 critic remains read-only; teacher remains label-only and fixed-prompt.
- No third P5 redirection and no use of G4 to reinterpret the conclusive G2 asymmetry FAIL.
- No G4 execution until PaperGuru marks this draft human-approved and locked.
