# Guard registry gated-access blocker

- Date: 2026-07-17
- Stage: `PREREG_LABELSOURCE_GUARD` §11.2
- Level: blocking pre-lock dependency; no scientific outcome accessed

The BeaverTails support gate passed, but all three preregistered guard
repositories require manual license acceptance.  AutoDL has no active
Hugging Face credentials, and the gated tokenizer/config/weights cannot be
read.  Consequently the exact chat templates/verbalizer IDs cannot be frozen
and the authorized distribution-only sanity cannot run.

Decision: stop at §11.2.  Preserve the pinned repository revisions and frozen
sanity cases in `reports/guard_registry.md`; do not use mirrors, substitutes, or
hard-label outputs.  Resume only after the owner accepts the licenses and logs
in on AutoDL.  No guard score, reliability metric, BeaverTails outcome, or Qwen
proxy annotation has been produced or inspected.
