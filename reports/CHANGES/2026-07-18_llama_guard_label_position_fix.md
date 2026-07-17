# Llama Guard semantic-label position correction

**Date:** 2026-07-18
**Level:** Green pre-lock measurement implementation
**Outcome access:** frozen eight-case distribution sanity only; no BeaverTails
score, human/proxy comparison, calibration metric, threshold, or guard ranking
was inspected.

## Failure

The first Llama Guard sanity run applied the pinned chat template and read the
immediate next-token probability of the exact `unsafe` token.  All eight values
were near zero (range `2.8208325952849833e-10`), so the non-degeneracy check
failed closed.  The failed artifact is retained as `sanity_llama_guard.json`
(SHA-256 `4d764ab4d8542f92506ad3b8cd0d36472193973bde347488444d9fdb801da389`).

## Diagnosis

A token-level diagnostic used only the same eight frozen cases and compared the
Meta model-card template call, the previous `add_generation_prompt=True` call,
and a fixed-prefix variant.  The two template calls rendered identically.  On
all 8/8 cases, greedy generation produced exact token ID `271` (decoded
`'\n\n'`) before the semantic `safe`/`unsafe` label.  The original code
therefore measured the probability of `unsafe` at a position where the model
deterministically emits formatting whitespace.

After consuming that common fixed prefix, the four frozen unsafe examples
generated `unsafe` with full-vocabulary probabilities `0.9901--0.9999`; the
four frozen benign examples generated `safe`, with unsafe probabilities
`2.14e-5--1.80e-4`.  Diagnostic SHA-256:
`894fca726d829edf4feb0162f61d1ac4fe140b415518ee16190c32be321ef22b`.

This agrees with Meta's model-specific usage, which lets the custom template
construct the guard prompt, and with the documented intent to score the first
safety-label token.  Meta's card does not spell out the deterministic newline
token boundary, so the exact boundary is frozen here from the pre-lock
diagnostic rather than inferred dynamically per item.

## Correction

`src/guard_score.py` now:

1. renders the pinned Llama Guard template using the model-specific default;
2. appends the globally fixed `"\n\n"` prefix (token ID `271`);
3. reads the full-vocabulary probability of exact token `unsafe` at the next,
   semantic-label position.

The implementation does **not** search for a label independently per example,
renormalize over `safe/unsafe`, alter the sanity cases, or alter the `0.05`
non-degeneracy threshold.  Two-token renormalization remains diagnostic only.
The tokenizer registry and machine-readable guard registry now explicitly
record this fixed prefix and corrected probability rule.

## Verification

The corrected Llama Guard sanity passed with range `0.999855292686334`.
ShieldGemma-2B and ShieldGemma-9B were run independently with their unchanged
official Yes/No interface and passed with ranges `0.9889777195785427` and
`0.992391557621886`, respectively.  The corrected registry SHA-256 is
`c3a0c9d0f1abbf4e137d84194708895d3d52a61947b94adc03cc02d5f3d5c425`.

The preregistration remains `DRAFT`; formal target scoring is still forbidden
until the remaining metadata gates are reviewed and a human-signed `LOCKED`
commit exists.

## References

- Meta Llama Guard 3 model card and usage:
  https://huggingface.co/meta-llama/Llama-Guard-3-8B#usage
- Meta PurpleLlama copy of the model card:
  https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md
- Hugging Face interface discussion documenting the newline-token issue:
  https://huggingface.co/meta-llama/Llama-Guard-3-8B/discussions/21
