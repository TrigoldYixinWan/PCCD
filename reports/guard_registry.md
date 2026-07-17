# Guard registry pre-lock audit

**Status:** `BLOCKED_AT_§11.2` — source metadata and the ShieldGemma tokenizer
interface are pinned, but Llama Guard access is awaiting Meta approval. The
registry is not frozen and the distribution-only probability sanity has not
run.

**Audit date:** 2026-07-17

No guard outcome, calibration statistic, ECE, ranking, or BeaverTails label was
inspected. The exact Hub heads below were resolved through the official Hugging
Face metadata API. After owner authentication, both ShieldGemma repositories
became accessible; Llama Guard 3 continued to return `Access denied. This
repository requires approval.`

## Source metadata

| Guard | Official repository | Revision | License | README SHA-256 |
|---|---|---|---|---|
| Llama Guard 3 8B | `meta-llama/Llama-Guard-3-8B` | `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425` | Llama 3.1 Community License (`llama3.1`) | `990e5f5f2dbe062d63ff0bfdab354aaa899c3486e4f807287d871ab9725ca0ac` |
| ShieldGemma 2B | `google/shieldgemma-2b` | `d1dffc9c8c9237a90aab09c61383791e718ef9e8` | Gemma Terms of Use (`gemma`) | `9184d3f33bb542841cb14436a938d9f53fae194f555c20e9c8e6ae60219b5893` |
| ShieldGemma 9B | `google/shieldgemma-9b` | `b8b636016df4540721a098c7aab91c97ec6ee508` | Gemma Terms of Use (`gemma`) | `2a5630d6014583c0adf891e777a76dde42bdc287e7a9d48801270519026d98ee` |

All three repositories have Hugging Face `gated=manual`. AutoDL authentication
now succeeds. Access is confirmed for both ShieldGemma repositories but not for
Llama Guard 3.

## ShieldGemma tokenizer/interface freeze

The 2B tokenizer files are downloaded from the pinned revision. The 9B model
references the same tokenizer blobs and will be independently hash-checked
after its pinned download completes.

- `tokenizer_config.json` SHA-256:
  `20dc327cc0ddb4bad2ebc5042f561f89fc980855a87db593ed80ac07ca5efcba`
- exact chat-template SHA-256:
  `19acd33317c8f50293a85a53c141f282e4175da3da68473c30967c4ef9e366d0`
- `tokenizer.json` SHA-256:
  `3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922`
- `Yes`: exact single token, ID `3553`
- `No`: exact single token, ID `1294`
- fixed probability rule:
  `softmax([logit(token 3553), logit(token 1294)])[0]`
- deterministic metadata record:
  `$PCCD_OUT/labelsource/prelock/shieldgemma_2b_tokenizer_registry.json`,
  SHA-256
  `a66328c6c74671baab506c9bc4a201dd29908c209980675740057540d3091f94`.

Input construction is the pinned tokenizer's `apply_chat_template` on
`[{role: user, content: prompt}, {role: assistant, content: response}]`, with
exactly one official response guideline passed through `guideline`. The four
response guidelines are Dangerous Content, Harassment, Hate Speech, and
Sexually Explicit Information, using the model card's response-specific
wording (“The chatbot shall not generate ...”).

## Llama Guard interface still to freeze

The intended input is the pinned tokenizer's `apply_chat_template` on the same
two-message chat. The model card specifies the probability of the first
generated `unsafe` token. Before lock, the exact template, decision token
ID/bytes, and whether the documented full-vocabulary probability or a
safe/unsafe-renormalized probability is used must be explicit. No inference or
silent choice is permitted before repository access is granted.

The full registry must also freeze dtype, padding, truncation, malformed-score
handling, all tokenizer/config hashes, and exact response prompt serialization.

## Frozen sanity inputs

`configs/guard_sanity_cases.json` (SHA-256
`4368a5eb1c51be549a733b69444e01bcd2e59f717feffa6731fa72c1a433efce`)
freezes eight obvious benign/unsafe prompt-response pairs before guard access.
The sanity checks only finite in-range values, at least two exact values per
interface, and pooled probability range >0.05 per guard. It forbids ECE, AUROC,
F1, accuracy, threshold selection, reliability plots, or guard ranking. The
runner must recheck this hash immediately before the first sanity run.

## Blocking condition and resolution

Authentication is resolved, and both ShieldGemma downloads are proceeding on
the data disk. The remaining owner action is to open
`meta-llama/Llama-Guard-3-8B` using the same Hugging Face account and submit the
access request, or wait if its status is `Pending`. The exact pinned download
will then be retried without changing the registered guard.

Until all three pinned models pass the distribution-only sanity,
`PREREG_LABELSOURCE_GUARD.md` remains `DRAFT`; taxonomy adjudication, Qwen schema
freeze, formal guard scoring, annotation, ECE, and ranking remain unauthorized.
No mirror or substitute checkpoint will be used.
