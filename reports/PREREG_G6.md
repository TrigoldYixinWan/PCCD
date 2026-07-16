# PCCD — G6 / P8 Structured Matrix-Scaling Recalibration Pre-Registration (DRAFT for human lock)

Drafted 2026-07-16 after P7 NEGATIVE (PREREG_G5.md / day8_g5.md) and a literature review that
identified structured matrix scaling as the evidence-backed next step beyond temperature.
**Not locked until PaperGuru approves.** After lock, changing the parameterization, splits,
budgets, verdict rule, or the overfitting guard is Red. This tests a NEW narrower proposition
P8 and CANNOT change any frozen verdict (P4/P5/P6/P7 stay frozen).

## 1. Proposition P8 (new)
P7 showed per-policy TEMPERATURE (a single scalar per policy) does not fully recover
calibration from 50-500 target labels, though it beats source-T and global-T. Temperature
scaling has a known ceiling: it can only rescale confidence, not correct per-class confusion
structure (motivation: Berta et al. 2025 structured matrix scaling; Dogah 2026 "temperature
scaling is not enough" — both held out of refs.bib until indexed, cited only in prose). P8
asks: does a slightly more expressive but still lightweight per-policy STRUCTURED MATRIX
scaling recover calibration to the base regime from the same small target budgets, without
harming discrimination?

## 2. Data discipline (LOCKED) — reuse the frozen P7 splits exactly
- Use the EXACT frozen TARGET-CALIB (1,000) / TARGET-TEST (2,000) ID partition from P7
  (g5_target_calib_ids.json / g5_target_test_ids.json, hashes in day8_g5.md), same nested
  budgets 50/100/200/500, same seed 20260722. This makes P8 directly comparable to P7.
- Frozen D0 critic logits and D5 teacher labels reused read-only; no teacher/critic/policy call.
- TARGET-TEST labels NEVER fit a scaler; used only to report ECE/F1/AUROC.

## 3. Parameterization (LOCKED)
For each policy p, its 3 logits z_p in R^3 are recalibrated by a structured affine map fit on
the target-calib labels:
```text
z'_p = W_p z_p + b_p
```
with W_p in R^{3x3}, b_p in R^3. To control the sparse-support overfitting risk (S2/S3 have
single-digit violated support at low budgets), the LOCKED primary parameterization is
DIAGONAL-plus-bias (per-class scale + per-class bias): W_p = diag(a_p), a_p in R^3, b_p in R^3
(6 params/policy) — strictly more expressive than a scalar temperature (which is a_p tied to a
single value, b_p=0) but far fewer params than full matrix scaling. Full 3x3 matrix scaling
(12 params/policy) is a SECONDARY method, reported but flagged as higher-overfit-risk.
Fit by 3-way NLL minimization (L-BFGS), including N/A, with L2 regularization toward the
IDENTITY map (lambda fixed BEFORE running, selected by leave-one-out NLL on the SOURCE base
calib split ONLY — never on target-test), logged. Clip to keep the map numerically stable.

## 4. Overfitting guard (LOCKED — this is the make-or-break design point)
Because low budgets + sparse per-policy violated support invite overfitting:
1. The regularization strength lambda is chosen ONLY on the source base-calib split (via
   LOO-NLL), frozen before touching any target data, and reported. It is NOT tuned on
   target-test.
2. Report the target-calib-fit NLL AND target-test ECE; a large train/test NLL gap is flagged
   as overfitting and reported honestly (does not get hidden).
3. Diagonal-plus-bias is PRIMARY (fewer params); full 3x3 is secondary. If full-matrix beats
   diagonal only on calib but not on test, that IS the overfitting story and is reported.
4. Same two-stage bootstrap as P7 (refit scaler on resampled calib; independently resample
   target-test; 10,000 replicates, seed 20260722).

## 5. Metrics & verdict (LOCKED — mirror P7 so results are comparable)
Per policy and mean, on TARGET-TEST, vs raw and vs the P7 per-policy-T baseline:
- 15-bin 3-way ECE (incl N/A); reduction vs raw; recovery vs P7 per-policy-T.
- violated-F1 (applicable) and AUROC non-inferiority. NOTE: unlike temperature, matrix scaling
  CAN change argmax; F1/AUROC are therefore genuine (not invariant) and MUST be checked.
- Smallest budget b* where diagonal-plus-bias achieves (a) reduction-vs-raw CI lower > 0,
  (b) mean ECE CI upper <= 0.05 (the P1 base regime), (c) discrimination non-inferiority
  (mean dF1 CI lower >= -0.005, mean dAUROC CI lower >= -0.01, no policy AUROC drop < -0.02).
- **P8 SUPPORTED (actionable remedy recovered):** such a b* exists at b <= 500, AND at b*
  diagonal-plus-bias beats P7 per-policy-T on mean ECE with paired 95% CI excluding 0.
- **P8 PARTIAL:** recovery (a)+(c) but ECE ceiling (b) still missed (improves over P7 but not
  to base regime).
- **P8 NEGATIVE:** no budget recovers to the base regime — then the paper's strong conclusion
  is that BOTH the temperature family AND structured matrix scaling fail to recover calibration
  from small target budgets, a method-class-level negative (stronger than P7 alone).

## 6. Boundaries (Red)
- Does NOT change frozen P4/P7 or any other verdict. P8 is a separate narrower claim.
- lambda chosen on source calib only; target-test never fits/selects anything.
- Frozen D0 critic read-only; teacher label-only fixed-prompt; reuse frozen P7 splits/labels.
- Diagonal-plus-bias is PRIMARY; full-matrix secondary; no post-hoc swap to whichever passes.

## 7. Outputs after approval
- `$PCCD_OUT/results/g6_matrix.json` (per method x budget x policy metrics, lambda, calib/test
  NLL, CIs, hashes); `reports/day9_g6.md` (BRIEF §F.3); `logs/g6_fit.log`; committed summary fig.
