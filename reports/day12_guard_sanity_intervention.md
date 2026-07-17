# Day 12 guard-interface intervention

**Status:** `§11.2_SANITY_PASS_PENDING_REVIEW`
**Scope:** pre-lock metadata and distribution-only interface sanity

All three pinned primary guards now have complete weights matching the official
revision-specific LFS SHA-256 manifests.  No formal BeaverTails guard outcome,
Qwen proxy label, ECE, AUROC, F1, threshold, or ranking has been computed.

## Results

| Guard | Probability interface | Min | Max | Range | Verdict |
|---|---|---:|---:|---:|---|
| Llama Guard 3 8B | full-vocabulary `P(unsafe)` at semantic label after fixed `\n\n` token | 0.00002144 | 0.99987674 | 0.99985529 | PASS |
| ShieldGemma 2B | `softmax([Yes, No])[Yes]`, four native policies | 0.00003536 | 0.98901308 | 0.98897772 | PASS |
| ShieldGemma 9B | `softmax([Yes, No])[Yes]`, four native policies | 0.00003120 | 0.99242276 | 0.99239156 | PASS |

Every interface produced at least two exact probability values, every value was
finite and in `[0,1]`, and every pooled range exceeded the frozen `0.05`
threshold.  The original Llama failure is retained and superseded only for the
documented semantic-label position bug; cases and gate threshold are unchanged.

## Frozen artifact hashes

- corrected registry: `c3a0c9d0f1abbf4e137d84194708895d3d52a61947b94adc03cc02d5f3d5c425`
- corrected Llama sanity: `27bf7716ac242a7f3c89bea326998ecc96d3f3b670a4b7b9c828f296c601e71f`
- ShieldGemma-2B sanity: `843b831e785b124626f884261300dda57b731edf60b4be149520313f9ae5d43a`
- ShieldGemma-9B sanity: `647ed2dd3a919612b4c733c80201f6f3997e1f9e4fa0ab0309b1afa0f3fb44e3`
- token diagnostic: `894fca726d829edf4feb0162f61d1ac4fe140b415518ee16190c32be321ef22b`

## Gate boundary

This resolves the guard distribution-sanity portion of preregistration §11.2.
It does not set the preregistration header to `LOCKED`.  Taxonomy mapping,
Qwen-32B blind annotation schema/objective-subjective grouping, and the final
human signature remain required before any target scoring.
