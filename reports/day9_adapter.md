# Day 9 independent D5 adapter

Date: 2026-07-16

Training code commit: `913e848`

Verdict: **PASS — the single registered new-seed D5 adapter was trained once,
validated, and frozen before lockbox response generation.**

No lockbox response, reference label, critic logit, calibration fit, or
confirmation aggregate had been generated when this report was written.

## Registered settings

| setting | frozen old D5 | new confirmation D5 |
|---|---:|---:|
| seed | 20260716 | **20260723** |
| objective / method | hidden-violation / SFT | hidden-violation / SFT |
| pairs | 512 | 512 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 | 32 / 64 / 0.05 |
| epochs / global steps | 4 / 64 | 4 / 64 |
| effective / per-device batch | 32 / 1 | 32 / 1 |
| world size / grad accumulation | 1 / 32 | 1 / 32 |
| learning rate | 2e-4 | 2e-4 |
| max length | 1024 | 1024 |

The validator compared every registered setting and found no difference other
than the required independent seed. The frozen pair corpus hash is
`27c0925636cc1ce2290fb40df49681e07c5c1eba9bcda313ca22f3c9e5c6ce41`
for both adapters.

## Training diagnostics

- Completed steps: 64/64; no restart or checkpoint selection.
- Runtime: 374.9 seconds.
- New training loss: `1.724493`.
- Frozen old-D5 training loss, reported only as a sanity comparison:
  `1.721826`.
- No OOM, NaN, infinite loss, or non-finite gradient was observed.
- Peak observed GPU allocation was approximately 20.5GB on one 96GB card.

The similarity of training losses is not an outcome criterion and did not
select or reject the checkpoint.

## Frozen hashes

- Adapter tree digest (SHA-256 of the sorted per-file hash manifest):
  `b11f448d350aea6d26e2cb91b61ad823c01ad5bd47be4c612f768a0804b75674`.
- Adapter files: 11; total bytes: 345,897,208.
- `adapter_model.safetensors`:
  `ee4f5a3871aa8bc64afea122a2c1f028199bf5c6a205408beafe5fcd1ac35038`.
- `adaptation_metadata.json`:
  `19242267ada02e6cd5835591be9da5fae1c5baa03871f311394a5f3422e7ba7d`.
- Validation record:
  `533b23ad1500712d5ca945e446f3338e650a698e3dca3840319bbc828a0d8ab2`.
- Training log:
  `3ace0b69446643e9e7875b8ec3eb61414044898f3492d2c085ff5e4c676a1381`.

The adapter remains at
`$PCCD_OUT/policy/confirm_D5_r32_seed20260723` and is read-only for the
remaining confirmation phases.
