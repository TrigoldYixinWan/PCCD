# Day 9 independent confirmation: P2-C, P3-C, and P8-C

Date: 2026-07-16

Locked protocol: `reports/PREREG_CONFIRMATION.md`  
Pre-unseal repository commit: `69843863365a0d9c43f612745a718d0a143a91f5`  
Pre-unseal manifest SHA-256:
`ad35c9e97e88228bb5ee03c8dc007ba3fc88be1ac2d5b5d924e5e2ed8fa7b184`

Final registered verdict:

> **CORE_NOT_ESTABLISHED; P8-C NOT_REACHED**

The independent lockbox replicated the D0 mean-calibration anchor but did not
replicate positive mean calibration degradation. The registered
criterion-interaction statistic is large, material, and highly stable across
the old and independent-seed D5 adapters, but the protocol makes P3-C
confirmatory only after P2-C passes. Therefore the project may report this as a
strong registered secondary finding, not as `P2_P3_CONFIRMED`.

## Integrity

- CONFIRM-TEST: 3,500 unique lexical families.
- Primary comparison: D0 versus independently trained new-seed D5.
- Reference integrity: evaluable; D0 100%, new D5 99.971% strict ten-key JSON
  on CONFIRM-TEST, with the one malformed new-D5 row retained as missing.
- Critic scoring: complete finite three-way logits for every row and criterion.
- Bootstrap: 10,000 paired query-family replicates.
- Primary package gate eligibility: PASS.
- Human audit packet: frozen before unseal; annotation pending.

## P2-C: mean calibration degradation was not established

| quantity | estimate | paired 95% CI | registered interpretation |
|---|---:|---:|---|
| mean D0 ECE | 0.039425 | [0.037866, 0.045538] | base anchor PASS; upper bound ≤0.05 |
| mean new-D5 ECE | 0.033243 | — | descriptive |
| mean ΔECE, new D5 − D0 | **−0.006183** | **[−0.011054, 0.000092]** | P2-C FAIL; CI contains zero |
| mean ΔECE, old D5 − D0 | −0.006957 | [−0.011603, −0.000448] | secondary old-seed comparison only |

The primary point estimate is opposite the discovery effect, and the CI does
not support any positive mean degradation. Because its upper endpoint is
slightly above zero, the locked label is `CORE_NOT_ESTABLISHED`, not
`P2_CONTRADICTED`. The effect is tagged `MATERIAL_LT_0.01`.

The independent D0 anchor is nevertheless supported: mean ECE remains below
the preregistered 0.05 operational region. This distinguishes “the frozen
critic is acceptably calibrated on average” from “every criterion transfers
unchanged.”

## Registered P3 statistic: strong interaction, but gate not reached

Although P2-C did not pass, the preregistered P3 computation produced:

- Helmert interaction Wald statistic: `175.270`;
- recentered paired-bootstrap p-value: `0.00009999`;
- cross-criterion SD of ΔECE: `0.024828`;
- SD 95% CI: `[0.020943, 0.028966]`;
- heterogeneity materiality tag: `MATERIAL_GE_0.01`.

The simultaneous max-|t| centered intervals localize five criteria whose
change differs from the ten-criterion mean:

| criterion | raw ΔECE | raw paired 95% CI | centered effect | simultaneous centered 95% CI |
|---|---:|---:|---:|---:|
| H2 | +0.013275 | [0.006571, 0.021814] | +0.019457 | [0.006486, 0.032429] |
| H4 | +0.038554 | [0.028546, 0.051662] | +0.044737 | [0.029759, 0.059714] |
| H5 | −0.027972 | [−0.039716, −0.009937] | −0.021790 | [−0.042699, −0.000880] |
| T1 | −0.027106 | [−0.041333, −0.008937] | −0.020923 | [−0.041154, −0.000692] |
| T2 | −0.055592 | [−0.069375, −0.038073] | −0.049409 | [−0.068279, −0.030540] |

H1, H3, S1, S2, and S3 have simultaneous centered intervals crossing zero.
Thus the near-zero/negative mean is a cancellation of criterion-specific
movements rather than evidence that every head is stable.

This is not an unqualified P3-C confirmation: Section 7 of the locked protocol
requires P2-C to pass first, and Section 10 permits the title/abstract
criterion-specific claim only under `P2_P3_CONFIRMED`.

## Independent-seed stability

The old-D5 secondary comparison, evaluated on the same new lockbox, shows the
same qualitative structure:

- interaction Wald statistic `186.329`, p=`0.00009999`;
- SD `0.028603`, 95% CI `[0.023258, 0.032279]`;
- old-versus-new D5 per-criterion ΔECE Spearman
  `ρ=0.9515`, family-bootstrap 95% CI `[0.8061, 0.9879]`.

This is strong seed-stability evidence for the criterion ordering. It is
secondary because the old adapter is not the independently registered primary
adapter and both adapters share the same confirmation prompts, teacher, and
critic.

## Sensitivity and mechanism boundaries

Bidirectional prevalence standardization did not rescue a positive mean drift:

- target standardized to D0 prevalence:
  mean ΔECE `−0.005234`, 95% CI `[−0.008851, 0.001698]`;
- D0 standardized to target prevalence:
  mean ΔECE `−0.067259`, 95% CI `[−0.078455, −0.056608]`;
- all point effective sample sizes were at least 100;
- registered `PREVALENCE_ROBUST` tag: false.

Therefore the independent result does not support a general
calibration-map-failure claim isolated from reference-state prevalence and
score-support change. The proper-score diagnostics are also heterogeneous:
H2/H4 worsen in ECE, NLL, and Brier, while several improving-ECE criteria also
improve proper scores; T2 is mixed. These diagnostics support a
criterion-dependent distributional change, not a universal scalar degradation
mechanism.

The frozen generation report also records a large response-length difference:
62.9% of D0 responses versus about 5% of either D5 seed reached 256 tokens.
This fixed behavior may contribute to support and reference-prevalence change.
It is disclosed as a possible mechanism but was not used for a post-unseal
rescue analysis.

## P8-C

P8-C is **NOT_REACHED** because P2-C did not pass. The P8 result record confirms
`target_inputs_read=false`: no confirmation target calibrator was fitted, and
no SMS/P7 comparison was attempted. This is not evidence that structured
matrix scaling succeeds or fails on the new lockbox.

## Result hashes

- `confirmation_p2_p3.json`:
  `8c1dfc21a2c56f576b05856cba9b9ff9cee99ebc839b087eabdf001af651bf67`.
- `g6_confirmation.json`:
  `6ec6a3d5d76ee9e096a4ed614dce733f15cb7c1fe71dc18bf8b9a2a050612fb4`.
- P2/P3 unseal log:
  `5466a4100327b60adc5f7dbe7df3fb55e9627a9265efea70b481d0b41ea872fc`.
- P8 gate log:
  `260daef243ee930832db35f8cb2d9f91394527be15799c91bf5d1739beb7004b`.

## Scientific conclusion

The discovery claim that hidden-violation adaptation causes positive *mean*
calibration degradation did not independently replicate and must be removed
from the paper's main claims. The strongest remaining empirical signal is that
an aggregate mean can conceal large, oppositely signed, seed-stable
criterion-level changes. Because confirmatory gatekeeping was not reached and
prevalence robustness failed, the paper must frame this as a registered
secondary risk-redistribution finding relative to the fixed reference
annotation protocol, pending human construct-validity audit.
