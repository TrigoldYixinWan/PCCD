# Day 9 independent-confirmation pre-unseal freeze

Date: 2026-07-16

Protocol: `reports/PREREG_CONFIRMATION.md`  
Frozen repository commit: `69843863365a0d9c43f612745a718d0a143a91f5`

Verdict: **PASS — all registered inputs, model artifacts, outputs, dependency
wheel, and environment metadata were hashed before any confirmation aggregate
was computed.**

## Freeze record

- Hash manifest:
  `$PCCD_OUT/confirmation/confirmation_preunseal.sha256`.
- Manifest SHA-256:
  `ad35c9e97e88228bb5ee03c8dc007ba3fc88be1ac2d5b5d924e5e2ed8fa7b184`.
- Environment metadata:
  `$PCCD_OUT/confirmation/confirmation_preunseal_environment.json`.
- Environment-metadata SHA-256:
  `5d4fcc1ca4f143c869e564007e34feaf30fa314d108472417fbb2a5d4e96f957`.
- Files in manifest: 66.
- Full `sha256sum -c`: 66 OK, 0 failed.
- Repository branch: `day9/g6-matrix`; worktree status was clean.
- Python: 3.12.3.
- Published calibrator package: `probmetrics==1.3.0`; its downloaded wheel is
  included in the manifest.

## Coverage

The manifest includes:

- frozen 4,000-family prompt/split/family manifests and source-bin edges;
- all three response, reference-label, full-logit, and test-subset artifacts;
- blinded and private human-audit artifacts;
- the frozen D0 critic;
- old and independent-seed D5 adapters;
- the fixed hidden-pair adaptation corpus;
- Qwen2.5-7B and Qwen2.5-32B model configurations;
- runtime package/environment metadata and the exact `probmetrics` wheel.

The expected confirmation result files
`confirmation_p2_p3.json` and `g6_confirmation.json` did not exist when the
freeze was verified. Therefore the next execution is the single registered
unseal rather than a rerun or continuation of an already inspected aggregate.
