# Guard registry pre-lock audit

**Status:** `BLOCKED_AT_§11.2` — source metadata is pinned, but the registry is
not frozen and the distribution-only probability sanity has not run.

**Audit date:** 2026-07-17

No guard output, calibration statistic, ECE, ranking, or BeaverTails label was
inspected in this gate.  The exact Hub heads below were resolved through the
official Hugging Face repository metadata API.  Public model cards were read at
those commits; gated tokenizer/config/weights were not accessible.

## Source metadata pinned so far

| Guard | Official repository | Revision | License | README SHA-256 |
|---|---|---|---|---|
| Llama Guard 3 8B | `meta-llama/Llama-Guard-3-8B` | `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425` | Llama 3.1 Community License (`llama3.1`) | `990e5f5f2dbe062d63ff0bfdab354aaa899c3486e4f807287d871ab9725ca0ac` |
| ShieldGemma 2B | `google/shieldgemma-2b` | `d1dffc9c8c9237a90aab09c61383791e718ef9e8` | Gemma Terms of Use (`gemma`) | `9184d3f33bb542841cb14436a938d9f53fae194f555c20e9c8e6ae60219b5893` |
| ShieldGemma 9B | `google/shieldgemma-9b` | `b8b636016df4540721a098c7aab91c97ec6ee508` | Gemma Terms of Use (`gemma`) | `2a5630d6014583c0adf891e777a76dde42bdc287e7a9d48801270519026d98ee` |

All three repositories have Hugging Face `gated=manual`.  The AutoDL
environment currently has neither a saved Hugging Face login nor an accessible
token for these repositories.

## Official interfaces to be verified after access

The model cards specify the following response-classification interfaces.  They
are recorded here but are not yet frozen because exact tokenizer templates and
token IDs must be read from the pinned gated files.

### Llama Guard 3 8B

- Input construction: pinned tokenizer's `apply_chat_template` on a two-message
  chat, `[{role: user, content: prompt}, {role: assistant, content: response}]`.
- Model-card probability: the first generated-token probability for the
  `unsafe` decision.  Before lock, the registry must record the exact decision
  token ID/decoded bytes and settle whether the card's full-vocabulary
  probability or a safe/unsafe-renormalized probability is used.  The
  preregistration currently says the card-documented first-token unsafe
  probability; it must not be silently changed after sanity output.
- Native response policies: S1–S14 as embedded by the pinned chat template.

### ShieldGemma 2B and 9B

- Input construction: pinned tokenizer's `apply_chat_template` on the same
  user/assistant messages, with exactly one official response guideline passed
  through the `guideline` argument.
- Model-card probability: `softmax([logit(Yes), logit(No)])[Yes]` at the next
  token.  Exact `Yes`/`No` token IDs and decoded bytes remain to be recorded.
- Native response guidelines: Dangerous Content, Harassment, Hate Speech, and
  Sexually Explicit Information, using the model card's response-specific
  wording ("The chatbot shall not generate ...").

The exact serialized chat template, response-guideline strings, tokenizer file
hashes, verbalizer IDs, dtype, padding, truncation, and malformed-score rule
remain mandatory registry fields.  They cannot be inferred safely from the
public README alone.

## Frozen sanity inputs

`configs/guard_sanity_cases.json` (SHA-256
`4368a5eb1c51be549a733b69444e01bcd2e59f717feffa6731fa72c1a433efce`)
freezes eight obvious benign/unsafe
prompt-response pairs before guard access.  The sanity is intentionally weaker
than an evaluation: it checks only finite in-range values, more than one exact
value per interface, and pooled range >0.05 per guard.  It forbids ECE, AUROC,
F1, accuracy, threshold selection, reliability plots, or guard ranking.  The
The runner must recheck this exact hash immediately before the first sanity
run and stop on any mismatch.

## Blocking condition and resolution

The §11.2 gate cannot pass without reading the pinned tokenizer/config files,
downloading the exact official weights, and running the one allowed
distribution-only sanity.  The project owner must:

1. accept the access terms on all three official Hugging Face model pages;
2. authenticate on AutoDL after `source scripts/setup/env.sh` using a read-only
   Hugging Face token (`hf auth login`), without pasting the token into chat or
   a tracked file; and
3. confirm `hf auth whoami` succeeds in that same environment.

Until then, `PREREG_LABELSOURCE_GUARD.md` remains `DRAFT`; taxonomy mapping,
Qwen schema freeze, guard scoring, annotation, ECE, and ranking are not
authorized.  No mirror or substitute checkpoint will be used to manufacture a
passing sanity result.
