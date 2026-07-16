# Day 9 independent-confirmation reference labeling

Date: 2026-07-16

Protocol: `reports/PREREG_CONFIRMATION.md`  
Execution code commit: `913e848`

Verdict: **PASS — the fixed label-only reference protocol completed once and
the three domains satisfy the registered reference-integrity thresholds.**

No retry, repair, imputation, label-distribution analysis, critic scoring,
calibrator fitting, or confirmation aggregate was performed. Each response was
submitted once at temperature 0 with a 256-token maximum. Malformed outputs
remain missing cells.

## Execution and alignment

- Teacher: frozen Qwen2.5-32B label-only protocol.
- Two independent one-GPU shards per domain; 2,000 rows per shard.
- Every final file contains 4,000 rows in the exact frozen prompt-manifest
  order.
- IDs, prompt text, response text, source, split, family, and stratum metadata
  match the corresponding response file row-for-row.
- Every record has `attempts=1`; the locked zero-retry rule was respected.
- A successful record has exactly the ten H1–H5/S1–S3/T1–T2 keys and only the
  registered `satisfied`, `violated`, or `not_applicable` values.

## Strict JSON integrity

| domain | all rows | TARGET-CALIB | CONFIRM-TEST | largest domain × policy missing rate | SHA-256 |
|---|---:|---:|---:|---:|---|
| D0 | 4,000/4,000 (100.000%) | 500/500 (100.000%) | 3,500/3,500 (100.000%) | 0% | `4746a0cfa45ded58c07a5eecdcbeba440e0ddf38b85fe6992606f66c1cfc15cd` |
| old D5 | 3,998/4,000 (99.950%) | 499/500 (99.800%) | 3,499/3,500 (99.971%) | 0.200% | `a2701ea14400c39217b1279b6bf41fcee47701fdb4a7d251d8f2c1eb0e028248` |
| new D5 | 3,999/4,000 (99.975%) | 500/500 (100.000%) | 3,499/3,500 (99.971%) | 0.029% | `bc7b053552e2ffcac92b98ca775a9f68398a4155bf346d219e20fa973a604d70` |

The old-D5 file contains two malformed full rows and the new-D5 file contains
one; one ID is shared between those domains. They are preserved as missing
ten-cell blocks. D0 has no missing cells. All domain-level strict success rates
exceed 99%, and every domain-by-policy missing rate is below 1%, so the locked
package remains evaluable.

These checks establish only syntactic and alignment integrity. They do not
claim that the reference judgments are error-free ground truth; construct
validity remains subject to the separately frozen blinded human audit.
