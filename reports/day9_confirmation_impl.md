# Day 9 independent-confirmation implementation

Date: 2026-07-16

Status: **implementation and AutoDL dependency/runtime preflight PASS; real
lockbox construction pending.**
No new adapter, lockbox response, reference label, critic logit, confirmatory
calibrator fit, or aggregate confirmation result was produced during this
implementation phase.

## Locked package implemented

- Outcome-blind 4,000-family lockbox with the exact registered H1-H5/general,
  UltraFeedback, and S1-S3 quotas; 500 TARGET-CALIB and 3,500 CONFIRM-TEST.
- Exact Unicode/word-five-shingle Jaccard family graph, transitive component
  exclusion, one prompt per family, fail-closed historical artifact registry,
  deterministic salted selection, and byte-reproducible manifests/hashes.
- One-GPU new-seed D5 training path plus a metadata validator that rejects any
  registered setting difference other than seed and stochastic training
  outcomes.
- Paired D0/old-D5/new-D5 response generation preserving family/stratum
  metadata, zero-retry reference labeling, frozen-critic logits-only scoring,
  deterministic manifest reordering, blinded 800-cell human-audit sampling,
  and a clean-commit pre-unseal artifact/environment freeze.
- P2-C/P3-C test-only analysis with fixed 15-bin ECE, paired family bootstrap,
  locked Helmert/Wald test, materiality/prevalence sensitivities, and secondary
  old-versus-new D5 delta-vector Spearman evidence.
- P8-C published `probmetrics==1.3.0` SMS primary, SVS secondary, paired
  per-criterion-temperature comparator, two-stage bootstrap, fit guards,
  max-|t| discrimination intervals, and exact registered verdict vocabulary.

## Missing-reference handling

The locked protocol allows up to 1% malformed reference output. The
implementation therefore preserves such cells as missing and never imputes
them. Critic scoring still uses the original prompt/response. Strict ten-key
success below 99% or any domain-by-criterion missing rate above 1% makes the
affected package `NON_EVALUABLE`.

## CPU verification

- `python scripts/day9/test_confirmation_lockbox_cpu.py`: 11/11 PASS.
- `python -m unittest discover -s tests -v`: 10/10 PASS.
- `python scripts/day9/test_g6_cpu.py`: PASS, including metrics, missingness,
  verdict partition, paired refitting, and dependency checks.
- `python scripts/day9/test_human_audit.py`: PASS, exactly 800 blinded cells.
- `python -m compileall -q src`: PASS.
- `git diff --check`: PASS.

The local Windows environment does not contain `probmetrics`; the local test
therefore reports integration as skipped. On AutoDL, `probmetrics==1.3.0`
passed constructor-default verification and real SMS/SVS fit/predict tests;
`pip check` reported no broken dependencies.

The first development-only timing probe also exposed that plain `fork` workers
cannot run PyTorch autograd after a parent calibration fit. The implementation
was changed before new outcomes to deterministic `forkserver` workers with a
per-replicate `SeedSequence([20260724, replicate])`. A 160-replicate,
four-budget, 16-worker benchmark on the consumed P7 split completed in 23.2
seconds with zero fit failures, making the locked 10,000-replicate run
operationally feasible.

## Remaining pre-outcome checks

1. Construct the real lockbox and confirm exact quotas, zero historical-family
   overlap, and frozen hashes.
2. Train and validate the one allowed new D5 seed, then execute the generation,
   reference, critic, audit-freeze, and one-unseal phases in
   `scripts/day9/run_confirmation.sh`.
