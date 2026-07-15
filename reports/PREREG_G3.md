# PCCD — G3 Scaling-Law Pre-Registration (DRAFT for human lock)

Drafted 2026-07-16 after the human-approved thesis reframe in
`reports/THESIS_REFRAME.md`. **This document is not locked until PaperGuru approves it.
No G3 regression, model comparison, residual inspection, or goodness-of-fit computation
may run before that approval.** Once approved, any change to the primary form, held-out
scheme, threshold, or verdict rule is Red.

## 1. Transparency boundary

G2 has already produced and reported the six marginal `(KL, mean Delta-ECE)` point
estimates for D1–D6. Consequently, this is a prospective lock of the *analysis method*,
not a claim that the analyst is blind to the marginal points. The safeguards against
post-hoc curve selection are:

1. one theory-motivated primary functional form is fixed below;
2. every D point, including the low-KL D6 and benign D3 control, is retained;
3. prediction is evaluated by leave-one-D-point-out (LODO) cross-validation, not
   in-sample fit;
4. linear and logarithmic alternatives are sensitivity analyses and cannot replace the
   primary model for the verdict;
5. the exact raw G2 artifacts and hashes are frozen before execution.

The authoritative G2 analysis artifact currently has SHA-256
`b7520277cb83d4c44e8808f977578ae8a90660841cd6f06998582d51c012d200`.

## 2. Scientific question and frozen data

G3 tests P6: whether adaptation-induced calibration degradation is predictably related
to adaptation strength measured by `KL(adapted || base)`.

- Adaptation points: `D1`, `D2`, `D3_control`, `D4`, `D5`, `D6`.
- D0 is excluded from fitting because `(KL, Delta-ECE)=(0,0)` is true by construction;
  including it would add a tautological anchor and inflate apparent fit.
- Prompt set: the fixed 3,000-item G2 evaluation set. No item may be removed.
- Predictor `x_d`: the locked token-weighted Monte Carlo KL in nats/token, recomputed
  from `$PCCD_OUT/g2/<point>_kl_items.jsonl` when bootstrapping.
- Cell outcome `y_dp`: the same 15-bin, top-class, 3-way ECE (including N/A) used in G2,
  `ECE_dp - ECE_D0,p`, on exactly paired prompt IDs.
- Point-level outcome `y_d`: the unweighted arithmetic mean of `y_dp` over all ten
  policies. ECE does not require violated support, so all ten policies remain in every
  point-level mean, including D6 S1/S3.
- Each D point receives equal weight. No inverse-variance weighting or point exclusion is
  allowed in the primary fit.

Teacher labels, frozen-critic logits, and KL item records are read-only. The teacher is
not called again and the D0 critic is not loaded with an optimizer.

## 3. Primary functional form (locked on approval)

The primary point-level scaling law is an intercept-bearing square-root power law:

```text
y_d = alpha + beta * sqrt(max(KL_d, 0)) + epsilon_d
```

It is fit by ordinary least squares. The exponent `1/2` is fixed rather than estimated:
total-variation shift is bounded by a square-root function of KL (Pinsker-type
motivation), while estimating an exponent from only six unique adaptation points would
be unstable. The intercept is free; it is not forced through the origin.

The point estimate must use the six observed positive KL estimates without clipping.
`max(KL,0)` only defines behavior for a rare negative Monte Carlo bootstrap replicate;
the fraction of such replicates must be reported.

## 4. Held-out prediction and goodness of fit

For each held-out point `d`:

1. fit `alpha` and `beta` on the other five D points;
2. predict the held-out `y_d` without refitting or selecting a form;
3. collect the six held-out residuals.

Primary goodness of fit is:

```text
R2_LODO = 1 - sum_d (y_d - yhat_-d)^2 / sum_d (y_d - mean(y))^2
```

The denominator uses the mean of the six observed point-level outcomes. Negative R² is
valid and must not be truncated. Also report LODO MAE, RMSE, full-data slope/intercept,
ordinary in-sample R², and the six observed-versus-predicted pairs.

An exact one-sided permutation test uses all `6! = 720` assignments of the six KL values
to the six outcomes. Its statistic is `R2_LODO` when the full-data slope is positive and
`-infinity` otherwise. The p-value is the fraction of the 720 assignments whose statistic
is at least the observed statistic; ties count as exceedances.

## 5. Uncertainty

Use 10,000 paired-prompt bootstrap replicates with seed `20260720`.

For each replicate, draw 3,000 prompt indices with replacement and use the *same indices*
for D0 and all six D points. Recompute:

- each point's token-weighted KL from item log-ratio sums and token counts;
- each policy's D0 and adapted ECE from teacher labels and frozen-critic logits;
- the ten-policy mean Delta-ECE, slope, and LODO metrics.

Report percentile 95% CIs. Do not bootstrap policies as if they were exchangeable; the
ten policies are the fixed taxonomy of interest. These CIs are conditional on the six
fixed adaptation regimes and quantify prompt-level measurement uncertainty; they do not
claim sampling-based generalization to every possible adaptation algorithm.

## 6. Pooled and per-policy analyses

These analyses are required and use the same square-root predictor, but they cannot
replace the point-level primary test.

### Policy-aware pooled model

Fit the 60 cells with policy fixed effects and one common slope:

```text
y_dp = alpha_p + beta_pool * sqrt(KL_d) + epsilon_dp
```

LODO holds out all ten cells belonging to one D point at a time. Report:

- pooled LODO R² over the 60 held-out predictions;
- LODO RMSE;
- `beta_pool` with paired-prompt bootstrap CI;
- incremental predictive R² relative to a policy-intercept-only LODO model:
  `1 - SSE_full / SSE_policy_only`.

This point-grouped holdout prevents the 60 cells from being treated as 60 independent
adaptation strengths.

### Per-policy fits

Fit the primary form separately to each policy's six Delta-ECE values. Report per policy:
slope, slope CI, in-sample R², LODO R², LODO MAE, and observed/predicted values. Apply Holm
correction across the ten one-sided positive-slope tests. Per-policy results characterize
the P3 heterogeneity mechanism; they are not an alternate route to G3 PASS.

## 7. Sensitivity analyses (non-gating)

Fit these without changing the primary verdict:

1. linear: `y = alpha + beta * KL`;
2. logarithmic: `y = alpha + beta * log1p(KL)`;
3. primary square-root form forced through `(0,0)` as a clearly labeled sensitivity;
4. hidden-SFT-only descriptive fit on D2/D4/D5 (three points, no inferential claim).

Report their LODO metrics and AICc where defined. Do not select the best form and relabel
it primary.

## 8. G3 verdict (locked on approval)

The primary point-level model passes only if all three conditions hold:

1. full-data `beta > 0` and the paired-prompt bootstrap 95% CI lower bound is `> 0`;
2. point-level `R2_LODO >= 0.70`;
3. exact one-sided permutation `p <= 0.05`.

The pooled policy-aware result is a required generalization check:

- **G3 PASS:** all three primary conditions hold, pooled `beta_pool > 0`, and pooled
  incremental LODO R² is `>= 0.30`.
- **G3 PARTIAL:** all three primary conditions hold but either pooled check fails.
- **G3 FAIL:** any primary condition fails.

No threshold may be relaxed after fitting. A sensitivity form cannot rescue a primary
FAIL. Report P6 with the exact verdict rather than using in-sample R² alone.

## 9. Required outputs after approval

- `$PCCD_OUT/results/g3_scaling.json` with all point, pooled, per-policy, bootstrap, and
  permutation results;
- `$PCCD_OUT/results/g3_predictions.jsonl` with each held-out prediction;
- raw figures under `$PCCD_OUT/results/` and a small committed summary figure;
- `reports/day6_g3.md` in BRIEF §F.3 format, including hashes and exact command;
- `logs/g3_fit.log`.

## 10. Red boundaries

- No D point, policy, or low/negative outcome may be removed from the primary analysis.
- No functional-form selection after fitting.
- No third reinterpretation of P5; G3 concerns ECE/KL only.
- Frozen D0 critic stays read-only; teacher remains label-only and fixed-prompt.
- No G3 execution until PaperGuru marks this draft human-approved and locked.
