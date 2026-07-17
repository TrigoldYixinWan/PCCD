# Llama Guard pre-lock sanity blocker

**Date:** 2026-07-18
**Status:** `RESOLVED_BY_PRELOCK_INTERFACE_FIX`
**Scope:** distribution-only interface sanity; no benchmark or reliability outcome was inspected.

## Verified assets

The pinned weight manifests for all three registered primary guards now verify
against their official SHA-256 digests:

- `meta-llama/Llama-Guard-3-8B` at `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425`;
- `google/shieldgemma-2b` at `d1dffc9c8c9237a90aab09c61383791e718ef9e8`;
- `google/shieldgemma-9b` at `b8b636016df4540721a098c7aab91c97ec6ee508`.

This resolves the prior interrupted-download condition only. It does not
establish that any guard interface is valid for scoring.

## Blocking result

With the frozen eight-case `configs/guard_sanity_cases.json`, the Llama Guard
sanity runner loaded the pinned model successfully but produced a degenerate
unsafe-probability range:

- minimum: `1.2378185976213985e-10`;
- maximum: `4.058651192906382e-10`;
- range: `2.8208325952849833e-10`;
- pre-registered minimum range: `0.05`;
- verdict: `pass=false`.

The result is finite and has six distinct values, but its near-zero range fails
the locked non-degeneracy condition. The post-download runner consequently
stopped after Llama Guard and did not execute either ShieldGemma sanity run.
No BeaverTails item, Qwen proxy annotation, guard score, ECE, F1, AUROC,
ranking, or threshold was computed.

## Required disposition

Keep `PREREG_LABELSOURCE_GUARD.md` as `DRAFT`; do not begin target scoring.
The next step requires a review of the Llama Guard official template and
first-token probability extraction against the pinned model card. Any repair
must be recorded and re-run only on the same frozen sanity cases before the
registry can be signed `LOCKED`.

## Resolution

The required review found a fixed `\n\n` formatting token before the semantic
label on all 8/8 frozen cases.  The corrected label-position implementation and
three passing guard sanity checks are documented in
`reports/CHANGES/2026-07-18_llama_guard_label_position_fix.md` and
`reports/day12_guard_sanity_intervention.md`.  Formal scoring remains blocked
on the other pre-lock gates and human signature.
