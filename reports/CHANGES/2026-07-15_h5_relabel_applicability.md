# Change: Stop on off-category H5 applicability after relabel (severity: Red)

Date / commit: 2026-07-15 / execution revision
`1039a2912e3beb31166bb3ac215286274bb4252e`

Trigger: The approved H5 relabel produced 1,065/1,376, 120/181, and 126/175 global
`satisfied`/`violated` counts for train/calib/test. A read-only join to the original PKU
harm-category fields showed that the selected response actually belongs to
`Endangering National Security` only 196/33/23 times. Of those target responses, the
teacher emitted 0/192/4, 0/33/0, and 0/22/1 `satisfied`/`violated`/`not_applicable`.
Therefore 92.13%, 89.04%, and 92.69% of all teacher H5-applicable labels lie outside the
approved selected-response category. Calib/test hit the expected approximately 30 target
items, but their 301 global applicable labels do not measure that category.

What I changed: No policy, prompt, sampling, label, metric, or gate logic was changed.
The minimal Red-level action was to preserve the regenerated artifacts, update
`reports/day2_labeling.md` and the three static-audit logs with the observed values, mark
the H5 scientific result PROVISIONAL, and stop before Day-3/training.

Why this and not an alternative: Category-directed oversampling was explicitly disallowed
for this rerun and would not fix the semantic mismatch. Overriding teacher labels from PKU
metadata, injecting the full taxonomy definition into the teacher prompt, category-gating
applicability, or accepting the global counts as a pass would each change the meaning of
the labels or whether the gate passed. Those decisions require PaperGuru sign-off under
BRIEF Section G.3. No data or metric was modified to match an expectation.

Impact on propositions/gates: H5 is intended to supply a distinct, data-backed hard-safety
dimension for policy heterogeneity and FN-asymmetry. The current global H5 two-class counts
are predominantly off-category, while true selected-response target examples contain no
`satisfied` labels. Treating these counts as valid could compromise P3/P5 and downstream
G1/G2 interpretation. Non-H5 distributions also moved after the joint ten-label prompt
changed (up to 13.90% distributional total variation), so the rerun cannot yet be assumed
to differ only in H5.

Reversibility: All regenerated pool/label artifacts and raw logs are preserved under
`$PCCD_OUT` and `/root/PCCD/logs`. The superseded prior hashes remain in Git history. No
workaround was applied, so review can choose a new labeling rule and rerun cleanly. The
current Day-2 scientific verdict is PROVISIONAL; artifact integrity itself passed.

Open question for PaperGuru: Should H5 applicability be anchored to PKU category metadata,
should the full H5 definition be injected and all ten policies relabeled again, or should
the project explicitly redefine H5 as the teacher's broader semantic national-security
judgment? A decision is also needed on whether the observed non-H5 distribution drift is
acceptable for downstream comparisons.
