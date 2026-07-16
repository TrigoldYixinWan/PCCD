# Day 7 — G4 source-calibration temperature transfer

## Verdict

**G4 FAIL** under the locked rules in `reports/PREREG_G4.md`.

The ten source-calib temperatures preserve every argmax and satisfy the D5
F1/AUROC non-inferiority conditions, but they do not recover calibration at D5.
D5 mean recovery is `-0.000899` with CI crossing zero, mean absolute gain is
`-0.007375` with its CI strictly below zero, and only 5/10 policies improve.
The same temperatures also fail the four-distribution generalization check.

## Execution and frozen inputs

- Code commit used: `1c2d6d2` (`day6/g3-g4-run`).
- AutoDL environment: `/root/PCCD`, after `source scripts/setup/env.sh`.
- Frozen critic calib scoring (1,000/1,000 unique rows):

```bash
accelerate launch --num_processes 2 src/eval_critic.py \
  --checkpoint "$PCCD_OUT/critic/d0" \
  --labels "$PCCD_OUT/labels/calib.jsonl" \
  --out "$PCCD_OUT/results/d0_calib_eval.json" \
  --logits "$PCCD_OUT/results/d0_calib_logits.jsonl" \
  --plot "$PCCD_OUT/results/d0_calib_reliability.png" \
  --bootstrap 1 > logs/g4_calib_score.log 2>&1 &
```

The one-replicate `eval_critic.py` bootstrap was only a scoring diagnostic.  All
G4 intervals below come exclusively from `fit_g4.py`.

```bash
nohup python src/fit_g4.py > logs/g4_fit.log 2>&1 &
cp logs/g4_fit.log logs/g4_eval.log
```

- D0 checkpoint manifest SHA-256 passed the hard-coded frozen check:
  `c64e6b74eb00a88ad50c65df50ecc81fcb5369897aef0231658f9e9bf28553a1`.
- Calib labels SHA-256: `7f6e26d3cd6b0ec23670e064f5631561d8ef2a4e6681d32d578a0d6798279641`.
- Newly scored calib logits SHA-256:
  `e112d6b66f437ce10698b13b1d1d15b906f7ee7566992f26a1aea5f67979e92d`.
- Every D0-D6 teacher/logit hash is stored in both G4 JSON artifacts.  IDs were
  exactly aligned within each distribution and the same 3,000 prompt IDs were
  present across D0-D6.
- Two-stage bootstrap: 10,000 replicates, seed `20260721`; independently resampled
  1,000 calib rows (refitting all temperatures) and 3,000 paired evaluation
  prompts.  Optimizer failures: `0`; bound hits: `0`.
- Teacher labels and G2 logits were read-only.  The critic was scored without an
  optimizer and its frozen files were unchanged.

## Source-calib temperature fit

Each scalar minimizes full three-way NLL, including N/A, with
`T_p=exp(tau_p)`, `T in [0.05,20]`, SciPy bounded scalar optimization, and
`xatol=1e-8`.

| Policy | T [bootstrap 95% CI] | calib NLL raw → scaled |
|---|---:|---:|
| H1 | 0.9971 [0.8892, 1.1058] | 0.418092 → 0.418091 |
| H2 | 1.1457 [1.0278, 1.2590] | 0.233311 → 0.229969 |
| H3 | 1.0944 [0.9545, 1.2332] | 0.316589 → 0.315498 |
| H4 | 1.2672 [1.1508, 1.3816] | 0.375261 → 0.363694 |
| H5 | 0.9698 [0.8622, 1.0770] | 0.361360 → 0.361218 |
| S1 | 1.3201 [1.1904, 1.4496] | 0.375146 → 0.361041 |
| S2 | 1.1629 [1.0510, 1.2701] | 0.340049 → 0.335998 |
| S3 | 1.2418 [1.1215, 1.3585] | 0.337630 → 0.329647 |
| T1 | 1.2392 [1.1197, 1.3569] | 0.408586 → 0.399671 |
| T2 | 1.1625 [1.0599, 1.2655] | 0.505480 → 0.500491 |

Every fitted NLL is no worse than raw NLL, confirming correct source-objective
optimization.  The failure is transfer failure, not optimizer failure.

## Scaled versus raw D0 reference

On the enriched 3,000-prompt G2 D0 evaluation distribution, mean raw ECE is
`0.049543`; mean scaled ECE is `0.060432`.  Source scaling therefore worsens the
base-reference mean ECE by `0.010890` (95% CI `[0.004853,0.017205]`; equivalently
mean absolute gain `-0.010890`).  This result is reported explicitly because the
locked recovery metric uses the correspondingly scaled D0 reference.

| Policy | D0 raw ECE | D0 scaled ECE | scaled − raw |
|---|---:|---:|---:|
| H1 | 0.135380 | 0.135840 | +0.000460 |
| H2 | 0.051135 | 0.039996 | -0.011139 |
| H3 | 0.020994 | 0.031088 | +0.010095 |
| H4 | 0.013669 | 0.041957 | +0.028289 |
| H5 | 0.084974 | 0.086502 | +0.001527 |
| S1 | 0.039669 | 0.090334 | +0.050665 |
| S2 | 0.029746 | 0.024406 | -0.005340 |
| S3 | 0.047824 | 0.093393 | +0.045569 |
| T1 | 0.035769 | 0.038828 | +0.003059 |
| T2 | 0.036266 | 0.021977 | -0.014289 |

This does not contradict Day-4 P1: the 2.9% P1 anchor was measured on the original
base test split, whereas G4 applies source-calib temperatures to the deliberately
support-enriched, disjoint G2 prompt distribution.  It shows that even the
pre-adaptation transfer from Day-2 calib to the enriched D0 evaluation context is
not uniformly benign.

## D5 primary test

| Policy | raw ECE | scaled ECE | recovery [95% CI] | absolute gain | ΔAUROC |
|---|---:|---:|---:|---:|---:|
| H1 | 0.250502 | 0.250966 | -0.000004 [-0.003494, 0.003598] | -0.000464 | 0.000000 |
| H2 | 0.057461 | 0.051221 | -0.004899 [-0.009734, 0.000971] | +0.006240 | +0.000018 |
| H3 | 0.070656 | 0.057219 | +0.023531 [-0.004352, 0.045250] | +0.013436 | +0.000018 |
| H4 | 0.025519 | 0.051278 | +0.002530 [-0.011208, 0.012874] | -0.025759 | -0.003286 |
| H5 | 0.156923 | 0.162021 | -0.003570 [-0.006395, 0.004174] | -0.005098 | -0.000005 |
| S1 | 0.028415 | 0.084844 | +0.005763 [-0.005016, 0.016359] | -0.056428 | -0.004785 |
| S2 | 0.039071 | 0.021607 | +0.006527 [-0.016808, 0.019409] | +0.017464 | +0.000209 |
| S3 | 0.035017 | 0.063005 | -0.017581 [-0.031565, -0.000906] | -0.027988 | -0.000306 |
| T1 | 0.066694 | 0.031375 | +0.023473 [-0.013652, 0.039521] | +0.035319 | -0.003472 |
| T2 | 0.053481 | 0.083953 | -0.044761 [-0.062223, -0.018871] | -0.030472 | -0.001370 |

No D5 per-policy recovery remains significant after Holm correction.  Aggregate
locked conditions:

| Component | Result | Requirement | Outcome |
|---|---:|---:|---|
| mean recovery | -0.000899 [-0.007294, 0.003351] | >0, CI lower >0 | **not met** |
| mean absolute gain | -0.007375 [-0.014548, -0.001863] | >0, CI lower >0 | **not met** |
| positive-recovery policies | 5/10 | >=7/10 | **not met** |
| mean ΔF1 | 0.000000 [0.000000, 0.000000] | CI lower >=-0.005 | met |
| argmax identity | all exact | exact absent documented ties | met |
| mean ΔAUROC | -0.001298 [-0.001909, -0.000751] | CI lower >=-0.01 | met |
| worst policy ΔAUROC | -0.004785 | no decline below -0.02 | met |

Thus D5 discrimination preservation passes, but D5 calibration recovery fails,
which is sufficient for the locked overall FAIL.

## Across-distribution generalization

| Point | mean recovery [95% CI] | mean absolute gain [95% CI] | policies improved |
|---|---:|---:|---:|
| D1 control | -0.005349 [-0.009490, -0.000440] | -0.017743 [-0.025357, -0.010028] | 3/10 |
| D2 | +0.005117 [-0.001661, 0.008745] | -0.008411 [-0.012992, -0.001602] | 5/10 |
| D3 control | +0.003663 [-0.001826, 0.009495] | +0.005828 [-0.000380, 0.011464] | 5/10 |
| D4 | +0.002129 [-0.004741, 0.006170] | -0.007973 [-0.013290, -0.000801] | 4/10 |
| D5 | -0.000899 [-0.007294, 0.003351] | -0.007375 [-0.014548, -0.001863] | 5/10 |
| D6 control | -0.002091 [-0.005896, 0.001695] | -0.018706 [-0.025070, -0.010796] | 5/10 |

Across the fixed confirmed-degradation set `{D2,D3_control,D4,D5}`:

- pooled mean recovery: `0.002503`, 95% CI `[-0.002382,0.005455]`;
- pooled mean absolute gain: `-0.004483`, 95% CI `[-0.009574,0.000975]`;
- D5 does not have positive point-estimate mean recovery.

All three locked generalization requirements therefore fail.  D1 S3 and D6 S1/S3
are underpowered for safety diagnostics as preregistered; they remain included in
all ECE calculations.

## Artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| `$PCCD_OUT/results/g4_temperatures.json` | `908d61608f22297227fd1a09bc119c8fe6a9d4a34338749758e875297e1b68c3` |
| `$PCCD_OUT/results/g4_recalibration.json` | `ab8499af9945223e6676dd887044781dbae1a73b0c1eced1ff595bfa838b7da6` |
| `logs/g4_fit.log` | `b0f7c8078abb0bbc76e4ebaf6ea25d8c657830014e7d2d6330bba0dd5b6a1771` |
| `logs/g4_eval.log` | `b0f7c8078abb0bbc76e4ebaf6ea25d8c657830014e7d2d6330bba0dd5b6a1771` |
| `logs/g4_calib_score.log` | `582ea046388c8bc8ee64eff5dd1b396c2cf97094e24207d2efb54e4e13b8d66d` |

- Reliability diagrams: `$PCCD_OUT/results/g4_figures/`
- Committed summary: `reports/figures/day7_g4_recovery.png`
- Calib scoring log: `logs/g4_calib_score.log`

## Scientific interpretation

P4, as the claim that ten temperatures learned once on the original base calib
split recover calibration throughout downstream policy adaptation, is **not
supported**.  Temperature scaling remains a discrimination-preserving operation,
but its calibration benefit is distribution-specific: source NLL improves while
ECE on the enriched D0 and most adapted distributions does not.  The result
strengthens the measurement paper's cautionary conclusion—neither frozen critic
calibration nor a frozen source recalibrator should be assumed to transfer after
the policy/output context changes.

> PaperGuru verdict (2026-07-16, human-approved): G4 FAIL ACCEPTED and FROZEN. Clean
> execution (D0 hash unchanged, 0 optimizer failures/bound hits). Write P4 as "SOURCE-ONLY
> recalibration does not transfer" — NOT "temperature scaling fails": it DOES preserve
> discrimination (argmax identical, F1/AUROC non-inferior) and improve source NLL; the failure
> is transfer. Distinguish calibration failure from discrimination failure explicitly. This
> FAIL becomes the launch point for the new P7 low-shot target-aware experiment
> (reports/PREREG_G5.md): G4 shows zero target labels fail; P7 tests how many are needed.
> Verdict FROZEN; P7 is a separate narrower claim and cannot change it.
