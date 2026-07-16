# Independent confirmation reframe and SMS dependency

Date: 2026-07-16

Authority: project owner delegated temporary PaperGuru decision authority to
Codex. The scientific choices and their evidence are recorded append-only in
`reports/DECISION_LOG.md`.

## Change

The unexecuted G6 draft is superseded by the locked independent package in
`reports/PREREG_CONFIRMATION.md`:

- all P1-P7 and L1-L3 verdicts remain frozen;
- the old G2/P7 target test becomes development evidence for any newly selected
  recalibration method;
- one outcome-blind lexical-family lockbox and one new D5 training seed provide
  the only confirmatory P2/P3/P8 run;
- the paper mainline becomes calibration transport failure plus
  criterion-specific drift, with repair auxiliary;
- a blinded human audit is required for construct-validity claims.

The old draft called a diagonal-plus-bias map "structured matrix scaling" even
though it is vector scaling and has a redundant common-logit bias. P8 now uses
the published Structured Matrix Scaling implementation in `probmetrics==1.3.0`
with its externally specified defaults; Structured Vector Scaling is secondary.
This exact dependency was added to `requirements.txt` and the setup script.

## Outcome-blind state at change

No new adapter, lockbox response, reference label, critic logit, scaler fit, or
aggregate confirmation statistic existed when the protocol commit was pushed.
The only remote operations were read-only capacity/artifact audits and a
dependency installation/dry-run. The capacity audit counted source metadata and
prompt overlap only; it did not inspect a new response outcome.

## Frozen boundaries

- No third P5 direction or altered P5 metric.
- No altered P4/P6/P7 verdict or threshold.
- Frozen D0 critic remains read-only.
- New CONFIRM-TEST is unsealed once; a failure cannot trigger another lockbox,
  adapter point, scaler, threshold, or direction in this project phase.
