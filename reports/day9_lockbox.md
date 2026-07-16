# Day 9 independent confirmation lockbox

Date: 2026-07-16

Code commit: `8450fdc`

Verdict: **PASS — outcome-blind lockbox frozen and ready for the single
registered adapter/generation run.**

No new policy response, teacher/reference label, critic logit, calibrator fit,
or confirmation aggregate was read or produced by this phase.

## Frozen size and separation

- Total prompts: 4,000.
- Unique IDs: 4,000.
- Unique lexical query families: 4,000.
- TARGET-CALIB: 500.
- CONFIRM-TEST: 3,500.
- Calibration/test ID overlap: 0.
- Selected historical-family overlap: 0.
- Selected family duplicates: 0.

## Exact registered quotas

| stratum | total | calib | test |
|---|---:|---:|---:|
| PKU H1 proxy | 240 | 30 | 210 |
| PKU H2 proxy | 240 | 30 | 210 |
| PKU H3 proxy | 240 | 30 | 210 |
| PKU H4 proxy | 240 | 30 | 210 |
| PKU H5 proxy | 240 | 30 | 210 |
| PKU general | 1,140 | 143 | 997 |
| UltraFeedback | 1,180 | 147 | 1,033 |
| soft S1 | 160 | 20 | 140 |
| soft S2 | 160 | 20 | 140 |
| soft S3 | 160 | 20 | 140 |

Source totals are PKU `2,340`, UltraFeedback `1,180`, and new soft prompts
`480`, exactly as registered.

## Capacity and exclusion audit

- Raw candidate prompts: PKU `73,907`, UltraFeedback `63,967`, soft `480`.
- Candidates excluded through a historical connected component: PKU `22,632`,
  UltraFeedback `4,665`, soft `0`.
- Historical rows represented in the family graph: `46,304`.
- Exact threshold candidate pairs checked: `39,609,147`.
- Jaccard edges at or above 0.85: `950,342`.
- Eligible H5 proxy components before assignment: `1,009`, safely above the
  locked quota of 240.
- Normalized candidate-universe digest:
  `e96c854701bef037bee7752618e6d1ca1098c81e61e9552aa098a11b506d53b5`.

The builder manifest records an empty `outcome_inputs_read` list and the hashes
of all nine fail-closed historical prompt artifacts.

## Artifact hashes

```text
4185ba1650e461672f371bcc62dda5583532ccc854dd68bac989b791a2b616e2  confirmation_family_exclusion_manifest.json
068dc6ba6dcaeae4ad002e028237cdf17dd3437c8f25a3a591f3469e7e283fb0  confirmation_prompts.jsonl
484170f476c516f21b1728ca64c2fdf7a0837840b46666882c391c6bc4a39e5d  confirmation_target_calib_ids.json
b265c14b4ed7d54a13d1781af0e5915435d64c8dc55a94dce53582f9fba6bba7  confirmation_test_ids.json
115040f7a4dee765058348ee9aad922fb30e782349bba0a7c8acef6db8c8d229  confirmation_lockbox.sha256
e1ab407bec2c9f3ad1e87def6b69ca19029a8fe9a32ce94a46bdd43f64e679af  source_base_calib_edges.json
```

`sha256sum -c confirmation_lockbox.sha256` passed for all four lockbox
artifacts. The physical files remain on the data disk at
`$PCCD_OUT/confirmation`; no large artifact was written to the system disk or
Git repository.
