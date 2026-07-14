# Change: Complete the fixed five-split sample pool (severity: Yellow)

Date / commit: 2026-07-14 / `256d85b`

Trigger: `src/sample_data.py` requested 6,200 PKU + 3,200 UltraFeedback + 1,300
self-built items, for at most 10,700 items before deduplication. Its fixed split sizes
are train 8,000 + calib 1,000 + test 1,000 + audit 400 + conflict 400 = 10,800.
Therefore the conflict split could contain at most 300 items even with zero duplicate
IDs. This contradicts the 400-item audit and 400-item conflict deliverables in
`configs/plan_9day.md`.

What I changed: In `src/sample_data.py`, the PKU candidate request changes from 6,200
to 6,300 and the pool description changes from approximately 10.7k to 10.8k. The
UltraFeedback request (3,200), self-built request (1,300), seed, deduplication, shuffle,
and all five split sizes remain unchanged.

Why this and not an alternative: Adding 100 PKU items is the smallest change that makes
the existing fixed splits attainable. The nearby code comment already identifies the
PKU surplus as coverage for the conflict split, and hard safety is the central source for
FN-asymmetry. Increasing UltraFeedback or soft-style items would instead shift the added
coverage toward supporting task/style policies. Reducing a fixed split would violate the
stated deliverables. The generated pool will still be checked for post-dedup size and
source composition before labeling.

Impact on propositions/gates: P1–P6 definitions and G1–G4 criteria do not change. The
additional candidate can change the deterministic shuffled membership of all splits, so
all Day-2 labels must be generated from the new pool rather than mixed with any earlier
pool. It restores the intended 400-item conflict input for the later reliability work.

Reversibility: Revert the one-line count change and regenerate the entire pool and all
dependent labels. No results have been produced from either pool in this execution, so
no result is provisional at the time of the change.

Open question for PaperGuru: Confirm that allocating the missing 100 candidates to PKU
is preferred over a proportional or UltraFeedback/soft allocation.
