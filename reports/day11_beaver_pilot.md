# BeaverTails Qwen–human compatibility pilot

**Status:** `GO` under the frozen exploratory rule; **not** an external-guard
confirmation gate.

**Run date:** 2026-07-17

**Scope:** one deterministic Qwen2.5-32B label-only call over a frozen,
multi-annotator BeaverTails subset. No guard inference, ECE, ranking, or
reanalysis of any PCCD lockbox was performed. `PREREG_EXTERNAL_GUARD.md`
remains `DRAFT`, and AEGIS remains `NOT_LOCKABLE`.

## Result

The pilot's narrow question was whether the frozen Qwen teacher is sufficiently
compatible with human BeaverTails QA-pair labels to justify investing in a
separate human-labelled external benchmark. It passed:

| Measure | Result |
|---|---:|
| Frozen QA pairs | 1,032 |
| Majority-safe negatives | 400 |
| Positive support (violence / hate / sexual / self-harm / illegal / privacy) | 231 / 126 / 123 / 120 / 126 / 120 |
| Strict six-key JSON parse | 1,030 / 1,032 = **99.806%** |
| Macro balanced accuracy | **0.8935** |
| Macro balanced-accuracy 95% CI | **[0.8811, 0.9055]** |
| Criteria with point BA ≥ 0.70 | 6 / 6 |
| Criteria with BA CI lower bound ≥ 0.70 | 6 / 6 |
| Frozen decision | **GO** |

The two strict-parse failures were retained and penalized as worst-case errors;
they were lexical misspellings (`violued`, `violited`), not silently repaired.

## Per-criterion results

| Criterion | Positive / safe support | Balanced accuracy | 95% CI | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| violence/physical | 231 / 400 | 0.7725 | [0.7375, 0.8061] | 0.6450 | 0.9000 |
| hate/identity | 126 / 400 | 0.8299 | [0.7883, 0.8679] | 0.7698 | 0.8900 |
| sexual | 123 / 400 | 0.9400 | [0.9116, 0.9658] | 0.9024 | 0.9775 |
| self-harm | 120 / 400 | 0.9783 | [0.9598, 0.9934] | 0.9667 | 0.9900 |
| illegal/criminal | 126 / 400 | 0.8755 | [0.8411, 0.9073] | 0.8810 | 0.8700 |
| privacy/PII | 120 / 400 | 0.9646 | [0.9437, 0.9816] | 0.9667 | 0.9625 |

The point-estimate range across criteria is 0.2058 and the criterion BA
standard deviation is 0.0746 (bootstrap 95% CIs [0.1713, 0.2437] and
[0.0641, 0.0878], respectively). This is a teacher–human compatibility
heterogeneity diagnostic, not evidence of calibration drift.

## Data and integrity

The data revision is BeaverTails commit
`8401fe609d288129cc684a9b3be6a93e41cfe678`. The pilot uses unique exact
prompt–response pairs from `330k_train`, excludes every exact pair appearing in
the 30k train/test and 330k test files, and aggregates repeated crowdworker
rows by strict majority before Qwen output access. The frozen manifest recorded
these hashes before inference:

- blind `items.jsonl`: `f8eeb4a55f1129cd1e7d31042511a073b694124418f696cbffd9e48a53da8da9`;
- private `reference.jsonl`: `1cf50110ba2fcd043f7ab0bb990437c8e9d12519ed7fce46957e9014c05ff0b6`;
- manifest: `73f43bf1cadd868297492ad592b9faabcc43b144304dce98987b37143f4fb292`;
- Qwen predictions: `07d9e2be9fdcfd50add1db820369b7a636b97f7289e025428e68e512ed7e2c7c`;
- analysis: `1117ac91b8ba36d9e6991b1d416e051230fd6ea396bfecc826efa9ce991f8a8d`.

The complete pilot outputs remain on the AutoDL data disk under
`$PCCD_OUT/beaver_pilot/`. The prediction file contains exactly 1,032 records,
and no process remains running after the single call.

## Interpretation and boundary

This is a useful positive feasibility result:

1. It reduces (but does not eliminate) the concern that the original Qwen
   teacher is globally incompatible with human safety labels.
2. It justifies designing a separate human-labelled external benchmark and a
   guard-level study, subject to a revised and signed preregistration.
3. It does **not** show that Qwen and humans agree equally on all criteria; the
   violence criterion is materially weaker than privacy or self-harm.
4. It does **not** validate the old PCCD adaptation claim, criterion-wise ECE
   direction, cancellation, guard ranking, or any P2–P8/G1–G6 verdict.
5. The pilot subset is stratified and therefore cannot be reported as a
   natural-prevalence accuracy estimate.

The appropriate next authorization is a preregistered human-labelled external
benchmark and guard scoring after PaperGuru resolves the AEGIS blocker. No
second Beaver pilot, threshold change, or post-hoc sample change is permitted.
