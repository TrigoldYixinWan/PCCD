# Day 9 independent-confirmation response generation

Date: 2026-07-16

Protocol: `reports/PREREG_CONFIRMATION.md`  
Execution code commit: `913e848`

Verdict: **PASS — all three registered response sets were generated exactly
once and passed outcome-blind integrity checks.**

At the time of this report, reference labels, frozen-critic logits, calibration
fits, and confirmation aggregates had not been inspected. The checks below use
only manifests, generation metadata, token counts, finish reasons, and hashes.

## Frozen execution

- Ordered prompt manifest: 4,000 unique IDs and 4,000 unique lexical families.
- Split: 500 `TARGET_CALIB`, 3,500 `CONFIRM_TEST`.
- Source composition: 2,340 PKU, 1,180 UF, 480 soft-policy prompts.
- Variants: D0 base policy, frozen old D5 adapter, and the independently trained
  new-seed D5 adapter.
- Shared decoding: temperature `1.0`, top-p `1.0`, maximum 256 generated
  tokens, seed `20260723`.
- Every response file preserves the exact manifest ID order, prompt text,
  source, family ID, split, and stratum metadata.
- No response-generation retry, checkpoint selection, prompt replacement, or
  length-based exclusion was performed.

## Integrity and operational diagnostics

| variant | rows | empty | stop | length | generated tokens, min / mean / max | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| D0 | 4,000 | 0 | 1,484 | 2,516 | 2 / 207.505 / 256 | `e9840d1c0f4b0d5cbb25bd18e97e7ffc4ba8cc110ff526297bbcd22231d1718d` |
| old D5 | 4,000 | 0 | 3,813 | 187 | 2 / 104.730 / 256 | `123dad10f3c4b88b7b0a6018bf3bf4577414a61435bdb174cd57d75ac5c36370` |
| new D5 | 4,000 | 0 | 3,792 | 208 | 2 / 107.746 / 256 | `1b3e6e858cd048e8633648b756ac92cb4b9ea20107c278334d62d4dacf8e672e` |

Prompt-manifest SHA-256:
`068dc6ba6dcaeae4ad002e028237cdf17dd3437c8f25a3a591f3469e7e283fb0`.

All three files contain exactly 4,000 valid JSON records, have no duplicate or
missing IDs, and match the frozen prompt manifest row-for-row.

## Length-limit disclosure

D0 reached the registered 256-token ceiling on 2,516/4,000 responses (62.9%),
compared with 187/4,000 (4.7%) for old D5 and 208/4,000 (5.2%) for new D5.
This is a large fixed behavioral difference and is retained without
intervention. It must be reported as a possible response-length/support
mechanism and examined with the preregistered descriptive content and support
diagnostics; it cannot justify regeneration, a larger token limit, exclusion,
or any change to the locked P2-C/P3-C/P8-C verdict rules.

The two D5 variants show similar token-length and finish-reason distributions,
which is an outcome-blind operational consistency check for the independent
seed. It is not evidence for or against the scientific hypotheses.
