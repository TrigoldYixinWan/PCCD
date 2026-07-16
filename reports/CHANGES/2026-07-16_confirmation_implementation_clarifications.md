# Confirmation implementation clarifications

Date: 2026-07-16

Authority: temporary delegated PaperGuru decision authority, recorded in
`reports/DECISION_LOG.md`.

## Changes

- Implemented the locked independent P2/P3/P8 confirmation package.
- Corrected stale 3,000/2,000 counts in the research-mainline summary to the
  superseding 4,000/3,500 design.
- Aligned the P8 verdict vocabulary, fit-failure ceilings, fixed-criterion
  aggregation, simultaneous AUROC bounds, and `NOT_REACHED` behavior with the
  locked protocol.
- Implemented the registered up-to-1% reference-missingness rule without
  imputation; critic scoring is label-independent in logits-only mode.
- Preserved query-family metadata through response generation and added
  deterministic shard reordering, adapter-equivalence validation, and
  pre-unseal artifact/environment hashing.

## Scientific effect

No frozen result, threshold, hypothesis direction, model family, adaptation
objective, critic checkpoint, or confirmatory sample was changed. These are
pre-outcome implementation clarifications and integrity guards. The associated
operational choices are recorded in `DECISION_LOG` entry DL-008.
