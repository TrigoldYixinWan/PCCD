# Day 9 human audit execution readiness

Date: 2026-07-16

Verdict: **READY FOR TWO INDEPENDENT HUMAN ANNOTATORS.**

The workflow and analysis were locked before any human label:

- protocol: `reports/PREREG_HUMAN_AUDIT.md`;
- annotator workflow: `src/human_audit_workflow.py`;
- weighted family-bootstrap analysis: `src/analyze_human_audit.py`;
- local and AutoDL test suites: 13/13 PASS.

## Frozen source and derived worksheets

- Blind JSONL SHA-256:
  `552df011ae43cd3587cf52ad2ec6e9e0c02fabfaf905068960873660d2f49a0d`.
- Annotator-A CSV SHA-256:
  `f34835fa6119bb91ab454faca7ca3f243cec92d0a125d66fbb911edfd40a15b7`.
- Annotator-B CSV SHA-256:
  `4b206558a9a6281ada092f788dee162f7ff2923d6e0a3a5e4bfacc5a19c73e06`.

Remote files:

- `$PCCD_OUT/confirmation/human_annotation/annotator_A.csv`
- `$PCCD_OUT/confirmation/human_annotation/annotator_B.csv`

The worksheets contain 800 blinded rows each in independently hashed order.
They contain no domain, source, reference state, critic output/confidence,
family ID, stratum, or sampling weight.

The next model-side action is intentionally blocked on external human work:
complete A and B independently, merge them, adjudicate disagreements, then run
the already-locked analysis. No further same-lockbox model experiment is
authorized.
