# Day 6 — G3 KL scaling law

## Verdict

**G3 FAIL** under the locked rules in `reports/PREREG_G3.md`.

The all-six square-root model has a positive slope whose paired-prompt bootstrap
CI excludes zero, and its exact permutation test is significant.  It nevertheless
misses the predictive threshold: LODO R² is `0.631786`, below the locked `0.70`.
The pooled policy-aware incremental LODO R² is also only `0.162254`, below its
required `0.30`.  Neither the hidden-SFT parallel analysis nor a non-gating
sensitivity is used to alter this verdict.

## Execution and frozen inputs

- Code commit used: `1c2d6d2` (`day6/g3-g4-run`).
- AutoDL environment: `/root/PCCD`, with `source scripts/setup/env.sh`.
- Command (stdout/stderr in `logs/g3_fit.log`):

```bash
nohup python src/fit_g3.py > logs/g3_fit.log 2>&1 &
```

- Bootstrap: 10,000 paired-prompt replicates, seed `20260720`; the same 3,000
  sampled prompt indices were used for D0 and all six adapted distributions.
- The authoritative G2 artifact passed its hard-coded SHA-256 check:
  `b7520277cb83d4c44e8808f977578ae8a90660841cd6f06998582d51c012d200`.
- Every KL and mean ΔECE point was recomputed from frozen item records and
  teacher/logit files and matched `g2_analysis.json` to absolute tolerance
  `1e-12`.  All 21 raw input hashes are stored in `g3_scaling.json`.
- No teacher or critic model was called.  No item, policy, or D point was removed.
- Negative bootstrap KL cells: `0 / 60,000`; the registered square-root clipping
  fallback was never activated.

## Primary all-six-point result

Locked model: `mean ΔECE = alpha + beta * sqrt(KL)` with a free intercept and
equal weight for each D point.

| Statistic | Result | Locked requirement | Outcome |
|---|---:|---:|---|
| Intercept alpha | -0.005156 | — | descriptive |
| Slope beta | 0.030855 | > 0 | met |
| Slope 95% CI | [0.025824, 0.035795] | lower > 0 | met |
| In-sample R² | 0.775518 | — | descriptive |
| LODO R² | 0.631786 | >= 0.70 | **not met** |
| LODO R² 95% CI | [0.370538, 0.813096] | — | descriptive |
| LODO MAE | 0.005706 | — | descriptive |
| LODO RMSE | 0.007354 | — | descriptive |
| Exact permutation p | 0.023611 (17/720) | <= 0.05 | met |

Observed and held-out predictions:

| Point | KL (nats/token) | mean ΔECE | LODO prediction | Residual |
|---|---:|---:|---:|---:|
| D1 | 0.212920 | 0.005286 | 0.010630 | -0.005344 |
| D2 | 1.038452 | 0.024770 | 0.026822 | -0.002052 |
| D3 control | 0.585680 | 0.030583 | 0.016031 | +0.014553 |
| D4 | 1.110715 | 0.021682 | 0.029659 | -0.007977 |
| D5 | 1.199633 | 0.028831 | 0.028546 | +0.000286 |
| D6 | 0.024871 | -0.001619 | 0.002406 | -0.004025 |

The dominant held-out error is the preregistered benign D3 control: at moderate
KL it degrades more than the higher-KL hidden-SFT points.  Thus KL carries a real
positive aggregate signal, but is not accurate enough as an objective-agnostic
predictor under the locked threshold.

## Mandatory hidden-SFT parallel result

The fixed-objective subset `{D2, D4, D5}` also does not form a reliable scaling
curve:

| Statistic | Hidden-SFT result |
|---|---:|
| Intercept | -0.035068 |
| Slope | 0.056968 |
| Slope 95% CI | [-0.028494, 0.105304] |
| In-sample R² | 0.367627 |
| LODO R² | -7.726439 |
| LODO MAE / RMSE | 0.008290 / 0.008649 |

The three observed ΔECE values are non-monotone (`D2 0.024770`, `D4 0.021682`,
`D5 0.028831`), and the slope CI crosses zero.  Therefore the result is not
“predictable within hidden SFT”; it agrees with the all-six FAIL rather than
creating a cross-objective disagreement that could be selectively reported.

## Policy-aware and per-policy results

The required 60-cell fixed-effects model has common slope `0.030855` with 95% CI
`[0.025824, 0.035795]`.  Its grouped LODO R² is `0.338893`; incremental LODO R²
over the policy-intercept-only model is `0.162254` (95% CI
`[0.115070, 0.208494]`), below the required `0.30`.

| Policy | Slope [95% CI] | In-sample R² | LODO R² | exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| H1 | 0.129874 [0.112839, 0.148078] | 0.913645 | 0.849491 | 0.001389 | 0.013889 |
| H2 | 0.002555 [-0.004930, 0.013531] | 0.009757 | -3.259147 | 0.373611 | 1.000000 |
| H3 | 0.052830 [0.032319, 0.063828] | 0.550129 | 0.277381 | 0.054167 | 0.379167 |
| H4 | 0.001146 [-0.002186, 0.014506] | 0.002288 | -0.739412 | 0.463889 | 1.000000 |
| H5 | 0.118504 [0.100671, 0.133555] | 0.936286 | 0.854543 | 0.008333 | 0.066667 |
| S1 | -0.012463 [-0.022069, -0.003079] | 0.249699 | -0.470298 | 0.858333 | 1.000000 |
| S2 | -0.019565 [-0.036146, -0.000653] | 0.044689 | -1.785805 | 0.694444 | 1.000000 |
| S3 | -0.042013 [-0.053301, -0.025761] | 0.933944 | 0.877801 | 0.995833 | 1.000000 |
| T1 | 0.049367 [0.030514, 0.064386] | 0.610624 | 0.289991 | 0.061111 | 0.379167 |
| T2 | 0.028316 [0.013047, 0.036352] | 0.546191 | -0.042253 | 0.002778 | 0.025000 |

Only H1 and T2 remain significant after Holm correction.  H5 has strong
descriptive fit but adjusted `p=0.0667`; S1-S3 have negative slopes, directly
showing why a single KL coefficient does not capture the full policy taxonomy.

## Non-gating sensitivities

| Form | Slope | In-sample R² | LODO R² | AICc |
|---|---:|---:|---:|---:|
| linear | 0.021973 | 0.678797 | 0.406368 | -51.770044 |
| log1p | 0.035737 | 0.739208 | 0.538310 | -53.020145 |
| sqrt through origin | 0.025228 | 0.744371 | 0.662137 | -58.140109 |

No sensitivity reaches the locked primary `0.70` predictive threshold, and none
is eligible to rescue the verdict.

## Artifacts

- Full raw result and all input hashes:
  `$PCCD_OUT/results/g3_scaling.json`, SHA-256
  `2ac9863a79e6ffd544407d25f48ec4203a791b88367e0519e62ef1234a8d64f7`.
- Held-out predictions: `$PCCD_OUT/results/g3_predictions.jsonl`, SHA-256
  `1264ce1e16335e35f00f43914d8bd8d0b3434668ca129dbb9cfa052346e76f73`.
- Raw figure: `$PCCD_OUT/results/g3_scaling.png`
- Committed summary figure: `reports/figures/day6_g3_scaling.png`
- Execution log: `logs/g3_fit.log`, SHA-256
  `5b1fc72b7c7d4def80e1e7055ff9393de695a291c2cfdbe1dac5459473d6dfff`.

## Scientific interpretation

P6, as the broad claim that KL alone predicts calibration degradation across
adaptation objectives, is **not supported**.  The positive slope and permutation
result establish association, but the locked out-of-point prediction criterion
correctly rejects a scaling-law claim.  The publishable statement is narrower:
adaptation strength correlates with mean drift, while objective and policy identity
materially condition the response; scalar KL is insufficient as a universal
deployment predictor.

> PaperGuru verdict (2026-07-16, human-approved): G3 FAIL ACCEPTED and FROZEN. Clean
> execution, no rescue. Write P6 as "KL magnitude is insufficient as a predictor ACROSS
> adaptation objectives" — NOT "KL is irrelevant": report the positive slope + permutation
> p=0.024 alongside the LODO 0.63<0.70 miss. The per-policy slopes ranging +0.13 (H1) to
> -0.04 (S3) are additional confirmation of P3 heterogeneity. Verdict is not to be revisited;
> see THESIS_REFRAME.md "SECOND REFRAME" for the final paper positioning.
