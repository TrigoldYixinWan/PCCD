# Day 8 — authorized token-level divergence analysis

## Status and conclusion

This was the pre-registered, **non-gating** mechanism analysis authorized in
`reports/MECHANISM_NOTES.md`. It does not change the frozen G3 verdict.

The proposed chi-square predictor is **not better than KL**. Its all-six-point
LODO R² is `-17.097648`, versus the frozen KL value `0.631786`; the paired
bootstrap difference is `-17.729434` with 95% CI
`[-344.268297, 0.086604]`. Reverse-KL is also worse. Total variation has a
slightly higher point estimate (`0.645992`), but its paired improvement over KL
is only `0.014206` with CI `[-0.048980, 0.078933]`, so it is not an identified
improvement. The paper must not claim that chi-square (or another tested scalar)
rescues P6.

## Execution and data boundary

- Token-recomputation code: commits `daf754c` and `eebf491` on
  `day8/divergence-g5`; analysis code used: `eebf491`.
- AutoDL environment: `/root/PCCD`, after `source scripts/setup/env.sh`.
- Exact frozen D1–D6 responses and saved adapters were used. No response was
  regenerated; the teacher and frozen critic were not called; existing
  `*_kl_items.jsonl` files were not overwritten.
- Deterministic teacher-forcing and hash freeze:

```bash
bash scripts/day8/recompute_token_ratios.sh \
  2>&1 | tee logs/day8_token_recompute_full.log
```

- Locked fit (10,000 paired-prompt bootstrap replicates, seed `20260720`,
  exact 720 assignments):

```bash
python src/analyze_divergence.py \
  2>&1 | tee logs/day8_divergence.log
```

The six new token files contain 3,000 records each. For every D point, the
newly summed and averaged token log-ratios reproduced the frozen per-item
`log_ratio_sum` and `log_ratio_mean` with maximum absolute error exactly `0.0`
(required tolerance `1e-6`; items over tolerance `0`). Files were set read-only
after their SHA-256 manifest passed `sha256sum -c`.

## Divergence estimates by D point

All estimators use the newly recovered per-token log-ratios under the frozen
adapted-response sample. Values are token-level plug-in estimates.

| Point | KL(adapted∥base) | chi-square(adapted∥base) | reverse-KL | TV | mean ΔECE |
|---|---:|---:|---:|---:|---:|
| D1 | 0.212920 | 435953.098069 | 0.180978 | 0.139840 | 0.005286 |
| D2 | 1.038452 | 136067888.286082 | 1.288654 | 0.370821 | 0.024770 |
| D3 control | 0.585680 | 31423783742.868225 | 0.421153 | 0.249302 | 0.030583 |
| D4 | 1.110715 | 641671297.530492 | 1.159637 | 0.371938 | 0.021682 |
| D5 | 1.199633 | 893327509.604916 | 1.071098 | 0.369925 | 0.028831 |
| D6 | 0.024871 | 13.374536 | 0.027219 | 0.052657 | -0.001619 |

The enormous chi-square values, especially at D3 control, are a direct
heavy-tail/importance-weight effect rather than evidence of better predictive
ordering.

## Locked-form predictive comparison

Every row uses the same registered model,
`mean ΔECE = alpha + beta * sqrt(predictor)`, with a free intercept, equal D-point
weights, leave-one-D-point-out prediction, 10,000 paired-prompt bootstrap
replicates, and the exact 720-assignment permutation test.

| Predictor | in-sample R² | LODO R² [95% CI] | ΔLODO R² vs KL [95% CI] | exact permutation p |
|---|---:|---:|---:|---:|
| KL(adapted∥base) | 0.775518 | 0.631786 [0.370538, 0.813096] | 0.000000 [0.000000, 0.000000] | 0.023611 (17/720) |
| chi-square(adapted∥base) | 0.346342 | -17.097648 [-343.479146, 0.617290] | -17.729434 [-344.268297, 0.086604] | 0.550000 (396/720) |
| reverse-KL | 0.644259 | 0.326409 [-0.074432, 0.646648] | -0.305377 [-0.553241, -0.086621] | 0.051389 (37/720) |
| total variation | 0.780854 | 0.645992 [0.392512, 0.818984] | 0.014206 [-0.048980, 0.078933] | 0.033333 (24/720) |

Thus chi-square is decisively not an empirical rescue here. TV remains a
descriptive sensitivity only: its improvement CI crosses zero, and a six-point
post-G3 sensitivity cannot support a replacement scaling-law claim. The result
is consistent with the broader mechanism interpretation that no single scalar
distance captures objective- and policy-dependent calibration degradation.

## Anomalies and implementation notes

- The recovered token arrays retain the model's singleton batch dimension
  (`[1, tokens]`). The reader was changed at `eebf491` to accept only that exact
  lossless shape (or a one-dimensional vector) before performing both frozen
  count and numerical checks. No artifact value was changed or regenerated.
- The chi-square plug-in is extremely tail-sensitive; the registered square-root
  transform does not prevent D3 from dominating held-out predictions. This is a
  substantive estimator behavior, not a numerical overflow (all estimates were
  finite).
- Existing KL values, G3 statistics, and the G3 FAIL verdict remain frozen.

## Artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| `$PCCD_OUT/g2/kl_tokens.sha256` | `2d6c61f7ae939b021f0620eebb38618988f8f1188b7e46081f3112921a6eaf96` |
| `$PCCD_OUT/g2/D1_kl_tokens.jsonl` | `fa1c7b436ebb86007a6154087657fb2e58023b07bf442676dc454ad3149fd21f` |
| `$PCCD_OUT/g2/D2_kl_tokens.jsonl` | `b926175cb2c3f7850dad7b42fdfa1019508ec64513dff1d08fd200260c9f86de` |
| `$PCCD_OUT/g2/D3_control_kl_tokens.jsonl` | `6478bf7ca7e977c210112cdae7efa804c7baec41645c049665921f4f292ffa88` |
| `$PCCD_OUT/g2/D4_kl_tokens.jsonl` | `1bff4cdf5b1194dc8b2a61d47ff35c9fab25e46c23a01fa605c9bf9847aaee05` |
| `$PCCD_OUT/g2/D5_kl_tokens.jsonl` | `5da6d098974fcd1f8cb57ba8dab5919b51e2e1f1716ff4c62892df5b94fc20c6` |
| `$PCCD_OUT/g2/D6_kl_tokens.jsonl` | `5a9a1753e775da1d6c79989370dcef6b65e4046096100c70410fce9b62d8e970` |
| `$PCCD_OUT/results/day8_divergence.json` | `f3a687f000605e3e9df4dbf41f13bf34db224fc42f5bf5566c2baf4220124cc5` |
| `$PCCD_OUT/results/day8_divergence.png` | `fc1f4ddbc8bf9bad8529f254555df5ca750445f9aac4a2726fa14ae4ee742f71` |
| `logs/day8_token_recompute_full.log` | `c8fd0ac78fc4422cfe10d5e8cd1b0f7993c2c8b9e59631d4cb579c6898ad47b8` |
| `logs/day8_divergence.log` | `9a89389849a85d58d3abe7c7c2b7b705db6fa3cb3cf7922a15d8011165ae3da4` |

Committed summary figure: `reports/figures/day8_divergence.png`.
