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

## Resolution (PaperGuru, 2026-07-15, human-approved)

DECISION: **Accept the teacher's broad-semantic judgment as the correct ground truth**
(answer to open-question option 3, chosen deliberately over category-gating).

Why this is methodologically correct, not a concession: the object of study is a deployed
FROZEN CRITIC that scores a policy's outputs. At deployment the critic must decide, for ANY
response the policy emits, whether policy H5 is violated — it has no access to a PKU
category tag. The teacher's broad-semantic label therefore IS the deployment-faithful oracle
we want to measure the critic against. Category-gating H5 (restricting applicability to PKU
"Endangering National Security" rows) would make H5 a probe that only exists on national-
security data — divorced from the deployment reality and inconsistent with how H1-H4 are
also (correctly) judged broadly. Hence H5's global 1,065/1,376 (train) and 120/126
(calib/test) satisfied/violated counts are VALID two-class coverage.

Answers to the three open questions:
1. Anchor H5 to PKU category metadata? NO — that would break deployment fidelity.
2. Inject full definition + relabel all ten again? The full H5 definition is ALREADY passed
   to the teacher (verified in build_messages); no separate injection is needed. One more
   relabel IS required, but only to bake the new pku_harm_categories metadata into the pool
   (see follow-up 1), not to change judgment semantics.
3. Redefine H5 as the teacher's broad national-security judgment? YES — this is the adopted
   framing. H5 = "does this response endanger national security", judged broadly.

Non-H5 drift (TV up to 13.90%): ACCEPTABLE. It is the expected effect of joint ten-policy
prompting (changing one policy's wording shifts the teacher's joint judgment). It is NOT a
confound because the ENTIRE pipeline (train/calib/test, all adaptation levels, all gates)
will use this single relabel; there is no cross-pool mixing. Commit 4e2cf9e is hereby the
SOLE authoritative Day-2 baseline; every earlier Day-2 label is void.

Follow-ups (both actioned):
1. GREEN (done by PaperGuru): src/sample_data.py now records `meta.pku_harm_categories` for
   each PKU item, so downstream G2 analysis can STRATIFY the FN-asymmetry result — reporting
   both the global asymmetry AND the asymmetry restricted to true-category subsets (e.g. the
   ~196 genuine national-security items). This strengthens P5 evidence; it does not change
   labels. Because the pool must now carry this metadata, ONE more relabel pass is needed.
2. The stratified national-security subset (teacher gave 0 satisfied / 192 violated there)
   is itself a REPORTABLE finding: on genuinely unsafe national-security responses the
   critic's job is almost entirely to catch violations — a natural stress case for FN-
   asymmetry. Keep this subset explicitly in the G2 report.

Status: H5 scientific result is no longer PROVISIONAL once the metadata relabel + audit
confirm unchanged H5 two-class coverage. Integrity already passed.
