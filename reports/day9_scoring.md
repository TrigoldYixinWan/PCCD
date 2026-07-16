# Day 9 independent-confirmation frozen-critic scoring

Date: 2026-07-16

Protocol: `reports/PREREG_CONFIRMATION.md`  
Successful scoring code commit: `e3a050b`

Verdict: **PASS — the exact frozen D0 critic produced complete, finite,
three-way logits for all registered response rows, with aligned 3,500-family
test subsets.**

No ECE, F1, AUROC, calibration fit, policy-level distribution, or confirmation
aggregate was calculated during this integrity stage.

## Execution

- Checkpoint: the frozen Day-4 D0 critic, loaded read-only.
- Distributed inference: two explicit Accelerate processes, one process per
  GPU; batch size 4 per process; bf16.
- Inputs: the three already-frozen 4,000-row reference files.
- Outputs: one 4,000-row logit file and one manifest-selected 3,500-row
  `CONFIRM_TEST` file per domain.
- The evaluator emits ten independent three-value logit vectors and ten
  three-way predictions per row. It does not update critic weights.

The initial launch failed before its first inference batch because the
logits-only input validator lacked an import of the existing `LABEL_TO_ID`
constant. It wrote no logits. The Green fix, regression test, and restart
rationale are frozen in
`reports/CHANGES/2026-07-16_confirmation_scoring_import.md` and decision
`DL-010`; the failed log is retained. The successful restart began at row one
with unchanged model, data, inference settings, and outputs absent.

## Integrity checks

| domain | full rows | test rows | finite ten-head 3-way logits | preserved missing references | full-logit SHA-256 | test-logit SHA-256 |
|---|---:|---:|---|---:|---|---|
| D0 | 4,000 | 3,500 | PASS | 0 | `762b449ecab3576b01609b4f30726fe5472825ae3871c10c375870680aad2f42` | `fcafd8cf206b969cd12c70430606989c98b14e59e7fbd9f9d6ef1aebadfb1877` |
| old D5 | 4,000 | 3,500 | PASS | 2 | `5fe0c0cb85db7ecc27cfeb49b57ba1d94c4848532c34060da47243df778349cf` | `eb4de9c8b012e4bf32d43d67a89d3eb06ebeb18226f6ba4f6ba0eccf2ab86c95` |
| new D5 | 4,000 | 3,500 | PASS | 1 | `978d3e1343ef80810b4fcd19c36d656a94a332db8ce9a498b5e9db1ce7be1092` | `f93af4cfe275dfd93d6040ddd5d4e37406fb63e86e3630fa0b8815b41bc1d090` |

For every domain:

- full-file IDs and order exactly match the corresponding reference file;
- test-file IDs and order exactly match the frozen test manifest;
- test rows are byte-equivalent JSON objects selected from the full file;
- sources and reference-label fields are preserved;
- every policy has exactly three finite numeric logits and one valid three-way
  prediction;
- malformed reference rows remain `null` labels and are not imputed.

This stage establishes artifact integrity only. Scientific outcomes remain
sealed until the human-audit packet and complete pre-unseal hash manifest are
frozen.
