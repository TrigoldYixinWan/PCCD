# Change: Complete the fixed five-split sample pool (severity: Yellow)

Date / commits: 2026-07-14 / `256d85b`, `3b13967`, `e20de11`

Trigger: `src/sample_data.py` requested 6,200 PKU + 3,200 UltraFeedback + 1,300
self-built items, for at most 10,700 items before deduplication. Its fixed split sizes
are train 8,000 + calib 1,000 + test 1,000 + audit 400 + conflict 400 = 10,800.
In addition, the 1,300 soft items were random draws from 10 tasks and a small finite set
of style combinations. Because IDs hash prompt+response, deduplication could collapse
them to roughly 120 unique records. `split_pool` silently returned short trailing splits.
This contradicts the 400-item audit and 400-item conflict deliverables in
`configs/plan_9day.md` and could also shorten test.

What I changed: In `src/sample_data.py`, the PKU candidate request changes from 6,200
to 6,400 and the pool description changes from approximately 10.7k to 10.8k. An initial
dry-run at 6,300 produced 6,276 unique PKU records and only 10,776 total unique records;
the final 100 candidates provide explicit deduplication headroom. Each of the
1,300 self-built items now receives a deterministic natural scenario composed from 10
teams, 13 initiatives, and 10 periods; the scenario appears in both prompt and response
body, making all 1,300 controlled examples textually unique. `split_pool` now raises if
fewer than 10,800 unique items are available or if any split has the wrong size. The
UltraFeedback request (3,200), self-built count (1,300), seed, deduplication, shuffle,
style matching probability, and all five split sizes remain unchanged.

Why this and not an alternative: Adding PKU headroom closes the explicit arithmetic gap
and the observed 24-record PKU dedup loss;
the nearby code comment already identifies the PKU surplus as coverage for the conflict
split, and hard safety is central to FN-asymmetry. Making the existing 1,300 soft records
unique preserves their count and policy purpose without retaining cross-split duplicates
or increasing their share. The scenario vocabulary uses ordinary deployment context and
does not change which style pole is requested or whether the response matches it. Reducing
a fixed split would violate the deliverables, while allowing duplicate IDs across splits
would create leakage. The generated pool will still be checked before labeling.

Impact on propositions/gates: P1–P6 definitions and G1–G4 criteria do not change. The
additional candidate can change the deterministic shuffled membership of all splits, so
all Day-2 labels must be generated from the new pool rather than mixed with any earlier
pool. It restores the intended 400-item conflict input for the later reliability work.

Reversibility: Revert the one-line count change and regenerate the entire pool and all
dependent labels. No results have been produced from either pool in this execution, so
no result is provisional at the time of the change.

Review resolution: PaperGuru approved the 6,400-PKU allocation on 2026-07-14 because
hard-safety coverage is central to FN asymmetry and supports the conflict split. Commit
`4f26ee9` subsequently clarified the audit-split docstring to match perturbation at label
time; it made no logic or sample-allocation change. No Yellow question remains open.
