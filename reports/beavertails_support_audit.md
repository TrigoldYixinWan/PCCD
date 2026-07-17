# BeaverTails 330k_test support audit

**Status:** `PASS` for PREREG_LABELSOURCE_GUARD §11.1; pre-lock metadata only.

**Run date:** 2026-07-17

No model was loaded and no guard probability, reliability statistic, ECE, or
ranking was computed.  The audit reads only the official BeaverTails human
annotation file and freezes the exact item/reference manifests used by later
stages.

## Source and unit of analysis

- Dataset: `PKU-Alignment/BeaverTails`
- Revision: `8401fe609d288129cc684a9b3be6a93e41cfe678`
- Split/file: `330k_test` / `round0_330k_test.jsonl.xz`
- Raw-file SHA-256:
  `1c726c662b85bf4761913041fe9e098f80e5206344864597dbde86bc9a1bfc96`
- Raw annotation rows: **33,396**
- Independent analysis units: **11,088 unique exact prompt-response pairs**
- Majority-safe / majority-unsafe pairs: **4,936 / 6,152**
- `is_safe` majority ties: **0**

The 330k file repeats prompt-response pairs across crowdworker annotations.
Therefore 33,396 is an annotation-row count, not an independent QA-pair count.
The preregistration draft's references to “~33.4k QA pairs” and “~18.5k unsafe”
must be corrected before `LOCKED`; the supported analysis population is 11,088
unique pairs and 6,152 majority-unsafe pairs.  This metadata correction does
not alter a threshold or inspect an outcome.

## Frozen aggregation rule

Annotations were grouped by
`SHA256(prompt + NUL + response)`.  `is_safe` and every native category flag
were aggregated independently by strict majority.  A category tie is missing
for that category only; an `is_safe` tie cannot be a benign negative.  A
category's preregistered negative support is conservative: majority
`is_safe=true` and category majority=false.  Two ties occurred, both for
`non_violent_unethical_behavior`; no other category tied.

Most pairs have three annotation rows (11,051/11,088).  The complete annotation
count histogram is: 3→11,051, 4→6, 6→24, 9→4, 12→2, 15→1.

## Per-category support

| Native category | Human positive | Benign negative | Ties | Freeze |
|---|---:|---:|---:|---|
| animal_abuse | 143 | 4,936 | 0 | PRIMARY |
| child_abuse | 49 | 4,936 | 0 | RARE |
| controversial_topics,politics | 218 | 4,936 | 0 | PRIMARY |
| discrimination,stereotype,injustice | 861 | 4,936 | 0 | PRIMARY |
| drug_abuse,weapons,banned_substance | 587 | 4,936 | 0 | PRIMARY |
| financial_crime,property_crime,theft | 1,110 | 4,936 | 0 | PRIMARY |
| hate_speech,offensive_language | 822 | 4,936 | 0 | PRIMARY |
| misinformation_regarding_ethics,laws_and_safety | 38 | 4,936 | 0 | RARE |
| non_violent_unethical_behavior | 1,949 | 4,936 | 2 | PRIMARY |
| privacy_violation | 546 | 4,936 | 0 | PRIMARY |
| self_harm | 60 | 4,936 | 0 | RARE |
| sexually_explicit,adult_content | 216 | 4,936 | 0 | PRIMARY |
| terrorism,organized_crime | 54 | 4,936 | 0 | RARE |
| violence,aiding_and_abetting,incitement | 2,987 | 4,936 | 0 | PRIMARY |

The frozen threshold is at least 100 human positives and 100 benign negatives.
**10 PRIMARY categories** pass, exceeding the required minimum of six.  The
four RARE categories remain mandatory low-N descriptive cells but are excluded
from the primary aggregate exactly as preregistered.

The draft's expected-list prose also needs a metadata-only correction before
lock: `animal_abuse` is PRIMARY, while `misinformation...` is RARE.  The
threshold and resulting classification were applied without amendment.

## Frozen manifests

The deterministic builder is
`scripts/day12/audit_beavertails_support.py`.  It produces two separate,
sorted 11,088-row artifacts:

- blind items (`id`, full pair hash, prompt, response):
  `beavertails_330k_test_blind_items.jsonl`, SHA-256
  `602de7160b536ae8567bda8a14ae6394a7413710c15b22157384c4d503a0c142`;
- private human reference (same IDs, vote counts, majority labels):
  `beavertails_330k_test_human_reference.jsonl`, SHA-256
  `6c64b844c97ef123fe9395e3a5f150e1c7bb0b7eba04d342e6b3669d9b2a75e9`;
- support summary:
  `beavertails_330k_test_support.json`, SHA-256
  `10a9022ef7e6dde3fcc917ab63d3da9ef9caedb107268e30e3bc5b3174983249`.

The blind and private files are physically separate so later Qwen annotation
can consume the former without access to the latter.  Their destination is
`$PCCD_OUT/labelsource/prelock/`; the raw dataset and generated artifacts are
kept off the system disk.

## Gate verdict and remaining pre-lock issue

§11.1 is **PASS (10 ≥ 6)**.  This does not authorize guard scoring.  Before the
overall preregistration can be signed `LOCKED`, PaperGuru must accept the
unique-pair unit correction and the two expected-list corrections above, and
§11.2–§11.4 must independently pass.
