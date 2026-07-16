# Day 8 — G5 / P7 low-shot target-aware recalibration

## Verdict

**P7 NEGATIVE** under the locked rules in `reports/PREREG_G5.md`.

Per-policy target temperature scaling produces a large, statistically identified
improvement over both raw D5 logits and global target temperature scaling, while
preserving discrimination. It nevertheless does not satisfy the registered
definition of recovery at any budget through 500: the 95% CI upper bound for
mean ECE remains above the P1 anchor ceiling `0.05`. Therefore no `b*` exists,
and the result cannot be called an actionable full recovery.

The supported narrower statement is important: **target labels and per-policy
structure materially reduce post-adaptation miscalibration, but 50–500 labels do
not establish restoration to the base well-calibrated regime.**

## Execution and frozen inputs

- Locked preregistration: `reports/PREREG_G5.md` (2026-07-16).
- Implementation commit: `e1c66b3`; final code used: `beba6c5` (the latter only
  clips negative display error-bar lengths at zero; JSON endpoints are unchanged).
- AutoDL environment: `/root/PCCD`, after `source scripts/setup/env.sh`.
- D5 teacher labels and frozen D0 critic logits were reused read-only. No teacher,
  critic, policy, or adapter was called or updated.
- Split preparation, performed before any target temperature fit:

```bash
python src/fit_g5.py prepare \
  2>&1 | tee logs/day8_g5_prepare.log
sha256sum "$PCCD_OUT/results/g5_target_calib_ids.json" \
  "$PCCD_OUT/results/g5_target_test_ids.json" \
  > "$PCCD_OUT/results/g5_split_manifests.sha256"
chmod 0444 "$PCCD_OUT/results/g5_target_calib_ids.json" \
  "$PCCD_OUT/results/g5_target_test_ids.json" \
  "$PCCD_OUT/results/g5_split_manifests.sha256"
sha256sum -c "$PCCD_OUT/results/g5_split_manifests.sha256"
```

- Registered fit and two-stage bootstrap:

```bash
python src/fit_g5.py fit \
  2>&1 | tee logs/day8_g5.log
```

The split follows the exact locked algorithm: lexicographically sort all 3,000
D5 IDs, permute with `numpy.random.default_rng(20260722)`, assign the first 1,000
to TARGET-CALIB and the remaining 2,000 to TARGET-TEST, then use nested prefixes
of 50/100/200/500 from TARGET-CALIB. Observed overlap is `0`.

| Frozen manifest | n | SHA-256 |
|---|---:|---|
| TARGET-CALIB | 1,000 | `e83d23388d0f14baa22335d35b3d151571e4bee5e08091711c66fdfe0d0ce01d` |
| TARGET-TEST | 2,000 | `dd820979f9c9e30d5c6e7685608a29734743dc3dc1fa3f46655501ef38ba81e9` |

Two-stage bootstrap used 10,000 replicates and seed `20260722`, independently
resampling calibration rows (and refitting temperatures) and test prompts.
Temperature optimizer failures: `0`. Recorded temperature-bound hits across
bootstrap cells: `157`.

## Target-calibration support

Cells show `satisfied / violated / N/A`; every policy contributes exactly the
registered number of three-way fitting cells. Sparse violated support for S2/S3
at small budgets is retained and reported, with no fallback or resampling.

| Policy | b=50 | b=100 | b=200 | b=500 |
|---|---:|---:|---:|---:|
| H1 | 24/4/22 | 43/6/51 | 80/14/106 | 181/35/284 |
| H2 | 38/6/6 | 78/11/11 | 158/22/20 | 393/71/36 |
| H3 | 14/8/28 | 21/13/66 | 36/30/134 | 81/89/330 |
| H4 | 43/7/0 | 92/7/1 | 179/19/2 | 453/42/5 |
| H5 | 16/5/29 | 27/9/64 | 49/15/136 | 116/52/332 |
| S1 | 36/2/12 | 79/4/17 | 152/8/40 | 373/12/115 |
| S2 | 34/0/16 | 69/2/29 | 132/5/63 | 339/10/151 |
| S3 | 36/0/14 | 73/0/27 | 138/2/60 | 351/2/147 |
| T1 | 43/7/0 | 89/10/1 | 171/27/2 | 427/68/5 |
| T2 | 43/4/3 | 90/4/6 | 173/13/14 | 429/28/43 |

## Learning curve and method comparison

Raw TARGET-TEST mean ECE is `0.079495`. Source-T, learned on the original base
calib split, has mean ECE `0.087576` with 95% CI
`[0.084626, 0.096173]` and is worse than raw by `0.008081` (reduction-vs-raw CI
`[-0.012350, -0.005044]`).

The table reports point mean ECE followed by the registered two-stage bootstrap
95% CI. Source-T is budget-independent and is repeated to make each comparison
explicit.

| Budget | source-T | target-global-T | target-per-policy-T | hierarchical shrinkage |
|---:|---:|---:|---:|---:|
| 50 | 0.087576 [0.084626, 0.096173] | 0.088730 [0.078270, 0.115724] | 0.053046 [0.051827, 0.085756] | 0.059865 [0.051300, 0.086304] |
| 100 | 0.087576 [0.084626, 0.096173] | 0.082517 [0.076674, 0.097782] | 0.047208 [0.049432, 0.069964] | 0.048340 [0.048999, 0.070024] |
| 200 | 0.087576 [0.084626, 0.096173] | 0.079832 [0.076014, 0.091387] | 0.047371 [0.047451, 0.063017] | 0.048786 [0.047491, 0.064196] |
| 500 | 0.087576 [0.084626, 0.096173] | 0.077379 [0.075032, 0.087479] | 0.043698 [0.044935, 0.056267] | 0.043983 [0.045083, 0.056774] |

For target-per-policy-T, the locked aggregate checks are:

| Budget | reduction vs raw [95% CI] | recovery vs source-T [95% CI] | mean ΔF1 [95% CI] | mean ΔAUROC [95% CI] | ECE CI upper ≤0.05 |
|---:|---:|---:|---:|---:|---|
| 50 | 0.026450 [-0.004800, 0.030954] | 0.034531 [0.003766, 0.039362] | 0.000000 [0.000000, 0.000000] | -0.000348 [-0.001979, 0.001042] | no (0.085756) |
| 100 | 0.032288 [0.011150, 0.033444] | 0.040368 [0.020106, 0.041754] | 0.000000 [0.000000, 0.000000] | -0.000169 [-0.001260, 0.000820] | no (0.069964) |
| 200 | 0.032125 [0.018307, 0.035014] | 0.040206 [0.027324, 0.043570] | 0.000000 [0.000000, 0.000000] | -0.000286 [-0.001178, 0.000503] | no (0.063017) |
| 500 | 0.035797 [0.025269, 0.037051] | 0.043878 [0.033706, 0.046056] | 0.000000 [0.000000, 0.000000] | -0.000097 [-0.000635, 0.000400] | no (0.056267) |

At b=100, 200, and 500, reduction versus raw and discrimination
non-inferiority pass. At b=50 the reduction CI crosses zero. At every budget the
absolute-ECE CI upper bound exceeds `0.05`; hence no budget satisfies all three
recovery conditions.

## Per-policy result at b=500

This table reports target-per-policy-T, the primary registered method. Recovery
is `ECE(source-T) - ECE(method)`; reduction is `ECE(raw) - ECE(method)`.

| Policy | T [95% CI] | ECE [95% CI] | recovery [95% CI] | reduction vs raw [95% CI] | Holm recovery p | ΔF1 | ΔAUROC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | 2.084002 [1.866136, 2.309963] | 0.157034 [0.139913, 0.182392] | 0.099488 [0.078331, 0.112948] | 0.099025 [0.077866, 0.112479] | 0.001000 | 0.000000 | 0.000120 [-0.000100, 0.000362] |
| H2 | 1.512491 [1.319746, 1.693388] | 0.027970 [0.024557, 0.049830] | 0.023880 [0.004341, 0.027487] | 0.032259 [0.011378, 0.035919] | 0.024398 | 0.000000 | 0.000054 [-0.000008, 0.000145] |
| H3 | 1.437675 [1.253365, 1.618302] | 0.032456 [0.023605, 0.052894] | 0.024840 [0.000106, 0.040356] | 0.039019 [0.013731, 0.054621] | 0.074693 | 0.000000 | 0.000062 [-0.000062, 0.000200] |
| H4 | 0.939009 [0.814040, 1.056949] | 0.022743 [0.017506, 0.041579] | 0.033578 [0.018580, 0.043769] | 0.003563 [-0.008470, 0.011382] | 0.001000 | 0.000000 | 0.000727 [-0.000644, 0.001876] |
| H5 | 2.003066 [1.774424, 2.237606] | 0.056948 [0.045654, 0.089336] | 0.103173 [0.073406, 0.117524] | 0.098787 [0.068593, 0.112635] | 0.001000 | 0.000000 | -0.000032 [-0.000162, 0.000074] |
| S1 | 0.896118 [0.760637, 1.025994] | 0.019914 [0.015563, 0.040552] | 0.066005 [0.041515, 0.073830] | 0.009084 [-0.013242, 0.016918] | 0.001000 | 0.000000 | 0.000848 [-0.000500, 0.002597] |
| S2 | 1.092807 [0.955894, 1.245933] | 0.026271 [0.020589, 0.058362] | -0.003350 [-0.027360, 0.010658] | 0.011890 [-0.012735, 0.025138] | 1.000000 | 0.000000 | -0.000028 [-0.000674, 0.000849] |
| S3 | 0.994770 [0.876323, 1.120754] | 0.041531 [0.030969, 0.062179] | 0.029174 [0.008091, 0.045215] | 0.001978 [-0.012222, 0.013244] | 0.016498 | 0.000000 | 0.000193 [-0.001133, 0.001389] |
| T1 | 1.378435 [1.204924, 1.551100] | 0.021793 [0.020055, 0.052400] | 0.003955 [-0.019662, 0.012576] | 0.034663 [0.003014, 0.044174] | 1.000000 | 0.000000 | -0.005023 [-0.007979, -0.002455] |
| T2 | 0.794187 [0.681318, 0.904280] | 0.030318 [0.020253, 0.046277] | 0.058040 [0.041874, 0.070292] | 0.027706 [0.012654, 0.039886] | 0.001000 | 0.000000 | 0.002115 [0.000569, 0.004179] |

Seven of ten policies have Holm-significant recovery versus source-T at b=500.
H1 remains poorly calibrated (`ECE 0.157034`) despite a large improvement; H5
also remains above the anchor in point estimate. This policy heterogeneity is
consistent with the retained P2/P3 thesis.

## Per-policy structure contrast

Although no `b*` exists and the official registered structure-benefit component
therefore remains false, the pre-specified paired contrast is positive at every
budget. These are valid secondary results, not a verdict rescue.

| Budget | global minus per-policy ECE [95% CI] | global minus hierarchical ECE [95% CI] |
|---:|---:|---:|
| 50 | 0.035684 [0.009929, 0.040821] | 0.028865 [0.013741, 0.039648] |
| 100 | 0.035309 [0.014995, 0.038272] | 0.034177 [0.016894, 0.037630] |
| 200 | 0.032462 [0.019674, 0.036874] | 0.031047 [0.019615, 0.036402] |
| 500 | 0.033681 [0.023940, 0.037226] | 0.033396 [0.023740, 0.036964] |

Thus global target scaling is inadequate, while policy-specific scaling is
consistently useful. Hierarchical shrinkage does not improve over the direct
per-policy point estimates here, but tracks them closely at larger budgets.

## Locked verdict components

| Budget | reduction CI lower >0 | ECE CI upper ≤0.05 | F1 non-inferior | AUROC non-inferior | recovery achieved |
|---:|---|---|---|---|---|
| 50 | no | no | yes | yes | no |
| 100 | yes | no | yes | yes | no |
| 200 | yes | no | yes | yes | no |
| 500 | yes | no | yes | yes | no |

- `b*`: none.
- Registered structure-benefit pass: false because it is evaluated at `b*`,
  which does not exist.
- **P7: NEGATIVE.** Frozen G4 remains unchanged.

> PaperGuru verdict (2026-07-16, human-approved): P7 NEGATIVE ACCEPTED and FROZEN. Clean run.
> IMPORTANT positive secondary finding retained for the paper: per-policy target scaling
> significantly beats global target scaling at EVERY budget (b=500 advantage 0.034 [0.024,
> 0.037]) and preserves discrimination — "policy structure matters, but scalar per-policy
> temperature is insufficient to fully restore base calibration from 50-500 labels."
> A literature review (MECHANISM_NOTES / new refs) shows temperature scaling has a KNOWN
> ceiling (Dogah 2026, held out) and structured matrix scaling is the evidence-backed stronger
> step (Berta 2025, held out). Therefore ONE final actionability attempt is authorized:
> P8 structured (diagonal-plus-bias) matrix scaling, reports/PREREG_G6.md, reusing the exact
> frozen P7 splits, with a locked overfitting guard (source-calib-only lambda selection). If P8
> also fails, the paper reports a METHOD-CLASS-level negative (temperature AND matrix scaling
> both insufficient from small target budgets) — stronger than P7 alone. P7 verdict FROZEN.

## Anomalies and implementation notes

- A percentile two-stage bootstrap CI need not contain the unresampled point
  estimate because every replicate refits temperatures and ECE is biased under
  finite test resampling. This occurs for the per-policy point estimates at
  b=100/200/500. The report preserves both the original point estimates and the
  exact registered percentile endpoints; no CI was recentered.
- The first complete statistical run wrote the JSON and then stopped during
  plotting because Matplotlib rejects a negative *error-bar length* when a
  percentile interval does not contain the point estimate. Commit `beba6c5`
  clips only plotted lengths to zero. The full deterministic rerun produced the
  identical JSON SHA-256 (`902f...3b1`), proving that no statistic changed.
- S2/S3 have zero or very low violated support at small budgets. N/A is a real
  fitting class and the locked protocol prohibited fallback, so all cells remain.

## Artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| `$PCCD_OUT/results/g5_target_calib_ids.json` | `e83d23388d0f14baa22335d35b3d151571e4bee5e08091711c66fdfe0d0ce01d` |
| `$PCCD_OUT/results/g5_target_test_ids.json` | `dd820979f9c9e30d5c6e7685608a29734743dc3dc1fa3f46655501ef38ba81e9` |
| `$PCCD_OUT/results/g5_split_manifests.sha256` | `af64cf6df8988bc1db3aaa7865559afb4740cd286d31dc7227467606586aa93f` |
| `$PCCD_OUT/results/g5_lowshot.json` | `902f02cb02e89022876308cc424c2987ab0b640e5929aa3311cda773c73083b1` |
| `$PCCD_OUT/results/g5_learning_curve.png` | `e5c54e1f451a84d6ac7c523ca53c07220c009acf708a50ba3def16fed83ea8b9` |
| `logs/day8_g5_prepare.log` | `da0f7434423986b041bc0cdc0bd18b9412a4b9a06a1603ca38de6843cd296c46` |
| `logs/day8_g5.log` | `5ada8029daa7e685b999e1926216d79ffed2d8406cb6548e9fc07ea5a6732330` |

Other frozen input hashes, including D5 labels/logits, source temperatures, and
the D0 checkpoint manifest, are stored in `g5_lowshot.json`. Committed summary
figure: `reports/figures/day8_g5_learning_curve.png`.
