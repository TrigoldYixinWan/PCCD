# Collision audit: ICLR 2025 guard-model calibration and adjacent work

Date: 2026-07-16
Status: **COMPLETE — GAP SURVIVES, BUT ONLY AS A NARROW SAFETY-SPECIFIC
EMPIRICAL/PROTOCOL CONTRIBUTION**

This audit implements the mandatory literature check in
`reports/PAPERGURU_PIVOT_VERDICT.md`. It changes no frozen PCCD verdict and
does not authorize an external experiment. In particular, P2-P8, G1-G6,
`CORE_NOT_ESTABLISHED`, the retirement of the mean-degradation thesis, and the
fixed-reference limitations remain unchanged.

## 1. Bottom-line decision

The NO-GO condition is **not triggered**. Liu et al. (ICLR 2025) study
binary safe/unsafe calibration by benchmark and response-model source, but do
not report:

- calibration by safety criterion/category;
- cancellation among signed criterion-level calibration shifts;
- mean-versus-worst-criterion guard-ranking reversals;
- a criterion-level deployment certificate with simultaneous uncertainty; or
- stability of a criterion-calibration vector across models or shifts.

The gap nevertheless became narrower after the adjacent-work scan. The project
cannot claim as novel that:

- aggregate calibration may hide subgroup miscalibration;
- worst-group calibration is useful;
- safety categories have different classification performance; or
- guard calibration changes across benchmarks or response-model sources.

The defensible increment is instead:

> a safety-guard-specific, cross-family and cross-taxonomy stress-test that
> estimates criterion-level calibration drift under distribution or
> response-source shift, controls family-wise uncertainty, and tests whether
> mean versus worst-criterion selection changes the guard chosen for
> deployment.

This is a conditional path to a new external study, not evidence that the new
claims are true.

## 2. Material audited

Primary collision:

- Liu et al., *On Calibration of LLM-based Guard Models for Reliable Content
  Moderation*, [arXiv:2410.10414](https://arxiv.org/html/2410.10414), ICLR
  2025.
- Full paper, all HTML/PDF tables, Appendix A, Appendix B, and the authors'
  [code repository](https://github.com/Waffle-Liu/calibration_guard_model).
- Code was inspected at commit
  [`b46b1a3`](https://github.com/Waffle-Liu/calibration_guard_model/tree/b46b1a3f2d4d7a042cd21bc93573ff09eb396116).

Adjacent guard and moderation work:

- Harsh et al., *Benchmarking Open-Source Safety Guard Models: A Comprehensive
  Evaluation*, [arXiv:2605.28830](https://arxiv.org/html/2605.28830v1).
- Machlovi et al., *A Multi-Perspective Benchmark and Moderation Model for
  Evaluating Safety and Adversarial Robustness* (GuardEval/GGuard),
  [arXiv:2601.03273](https://arxiv.org/html/2601.03273v2). Its first version
  was submitted in 2025; the audited version is the 2026 revision.
- Inan et al., *Llama Guard*,
  [arXiv:2312.06674](https://arxiv.org/html/2312.06674v1).
- Surana, *Fair and Calibrated Toxicity Detection with Robust Training and
  Abstention*, [arXiv:2605.14074](https://arxiv.org/html/2605.14074v1).

Adjacent multicalibration and multi-group uncertainty work:

- Detommaso et al., *Multicalibration for Confidence Scoring in LLMs*,
  [arXiv:2404.04689](https://arxiv.org/abs/2404.04689).
- Hansen et al., *When is Multicalibration Post-Processing Necessary?*,
  [arXiv:2406.06487](https://arxiv.org/html/2406.06487v1).
- Wu et al., *Bridging Multicalibration and Out-of-distribution Generalization
  Beyond Covariate Shift*,
  [NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/859b6564b04959833fdf52ae6f726f84-Abstract-Conference.html).
- Liu and Wu, *Multi-group Uncertainty Quantification for Long-form Text
  Generation*, [arXiv:2407.21057](https://arxiv.org/abs/2407.21057).

## 3. Required Q1-Q5 answers

| Question | Answer | Consequence |
|---|---|---|
| Q1. Calibration by safety category, rather than benchmark/task? | **No.** Categories are described, but the reported calibration task is binary safe/unsafe and results are indexed by benchmark or response-model source. | Criterion-level calibration remains open. |
| Q2. Explicit criterion-shift cancellation test? | **No.** There is no signed criterion-drift vector or cancellation statistic. | The exact cancellation audit is not preempted, though the arithmetic fact is not novel. |
| Q3. Average versus worst-criterion guard ranking? | **No.** Table 1 selects the best average ECE across benchmarks, not the worst safety criterion, and does not test rank reversal. | A safety-criterion deployment-selection result remains open. |
| Q4. Criterion-level certificate with simultaneous uncertainty? | **No.** The paper reports point estimates and reliability diagrams; no criterion family, simultaneous interval, or deployment threshold is defined. | An auditable certificate is a possible protocol contribution. |
| Q5. Stability of a criterion-calibration vector across models/shifts? | **No.** Response-source variability is reported only for scalar binary ECE/F1 by source model. | Vector stability across safety criteria remains open. |

### Q1 evidence: the analysis deliberately collapses categories to binary

The paper first notes that guards can emit a binary decision followed by an
unsafe category (Sections 1 and 3). Section 4 then states that different safety
taxonomies make multiclass comparison difficult and therefore emphasizes
binary safe/unsafe classification. The main outcomes are:

- Table 1: ECE for each guard by public benchmark, plus an `Avg.` column;
- Table 2: ECE/F1 for each guard by response-model source on HarmBench-adv;
- Appendix Table 6: dataset support reported only as safe/unsafe counts;
- Appendix Table 14: FPR/FNR by benchmark, not by safety criterion.

The appendix explicitly says OpenAI Moderation contains eight category flags
and Aegis covers thirteen unsafe categories, but no category-wise ECE table or
reliability diagram follows. The repository confirms the collapse:

- OpenAI category flags are converted to one binary label through
  `int(1 in list(d.values()))` in
  [`eval.py:233`](https://github.com/Waffle-Liu/calibration_guard_model/blob/b46b1a3f2d4d7a042cd21bc93573ff09eb396116/eval.py#L233).
- Aegis annotator labels are reduced to one binary prompt label in
  [`eval.py:253-258`](https://github.com/Waffle-Liu/calibration_guard_model/blob/b46b1a3f2d4d7a042cd21bc93573ff09eb396116/eval.py#L253-L258).
- BeaverTails uses only `is_safe`, and WildGuard uses only binary prompt and
  response harm labels in the same evaluation script.

Thus the paper contains category metadata and category-aware guard prompts,
but its calibration estimand is not per category.

### Q2 evidence: no criterion vector exists to cancel

Liu et al. report large variation across benchmarks and across ten response
models. Those are scalar ECE values for independently defined dataset/source
cells. The paper does not define

```text
Delta_p = ECE_target,p - ECE_source,p
```

or any statistic over signed criterion-specific changes. Searches of the paper,
appendix, and repository found no criterion cancellation, sign-mixture, or
mean-absolute-versus-absolute-mean analysis.

The proposed cancellation statistic is therefore not duplicated. It must still
be presented as an audit statistic following from the triangle inequality, not
as a theorem.

### Q3 evidence: average benchmark ranking is not worst-criterion ranking

Table 1 bolds the guard with the best `Avg.` ECE for prompt and response
classification. The paper observes that individual guards vary across
benchmarks, but does not construct a safety-criterion vector, calculate a
worst-criterion value, or compare the guard selected by the two objectives.

The paper therefore establishes that guard rankings depend on evaluation
setting, but not the proposed deployment consequence:

```text
argmin_guard mean_p(ECE_guard,p)
    versus
argmin_guard max_p(ECE_guard,p).
```

### Q4 evidence: no simultaneous criterion certificate

The ethical statement correctly warns that confidence calibration should not
be the sole deployment criterion and recommends a holistic evaluation. That is
deployment guidance, not a criterion-level statistical certificate.

The primary evaluation scripts calculate one point ECE:

- [`eval.py:398`](https://github.com/Waffle-Liu/calibration_guard_model/blob/b46b1a3f2d4d7a042cd21bc93573ff09eb396116/eval.py#L398);
- [`eval_model_dep.py:236`](https://github.com/Waffle-Liu/calibration_guard_model/blob/b46b1a3f2d4d7a042cd21bc93573ff09eb396116/eval_model_dep.py#L236).

The bundled calibration library contains a generic 100-resample bootstrap
utility at
[`calibration/utils.py:63-90`](https://github.com/Waffle-Liu/calibration_guard_model/blob/b46b1a3f2d4d7a042cd21bc93573ff09eb396116/calibration/utils.py#L63-L90),
but the paper's main evaluation scripts do not call it. More importantly, the
repository contains no simultaneous max-statistic interval across safety
criteria, no family-wise error control, no minimum-support rule, and no
pass/fail deployment certificate.

### Q5 evidence: response-source robustness is scalar, not vector stability

Section 4.2.3 is the closest collision. It partitions HarmBench-adv responses
by ten generating models and reports that guards have inconsistent binary ECE
and F1 across those sources. This directly owns the broad claim that
response-source shift affects guard calibration.

It does not:

- compute category-wise ECE within a response source;
- correlate criterion vectors across response models;
- test whether criterion ordering is stable;
- estimate vector drift with joint uncertainty; or
- replicate a criterion profile across guard families and taxonomies.

Any external PCCD follow-up must therefore cite Liu et al. as the source-shift
baseline and claim only the additional criterion geometry and deployment
selection consequence.

## 4. Adjacent-work collision matrix

### Harsh et al. 2026

This ICLR workshop paper evaluates fourteen guards on 79,331 examples in eight
NIST-derived safety categories. It reports category-level detection behavior
and argues that recall is critical for deployment. The full text contains no
calibration analysis.

Collision: large-scale category-aware guard benchmarking and practical model
selection are already present. Remaining gap: criterion-conditioned
calibration, cancellation, simultaneous uncertainty, and mean/worst
calibration ranking.

### Machlovi et al. 2025/2026 (GuardEval/GGuard)

GuardEval harmonizes 106 fine-grained categories and compares moderation
performance across heterogeneous benchmarks. It reports a single binary safety
calibration result for GGuard: temperature scaling reduces overall 10-bin ECE
from 0.1203 to 0.0554. Its fine-grained analysis concerns classification,
taxonomy overlap, FPR/FNR, and fairness, not per-category ECE or
criterion-drift cancellation.

Collision: taxonomy harmonization, cross-benchmark moderation, and an overall
calibration result are already present. Remaining gap: native-taxonomy
criterion calibration and the deployment certificate. Because GuardEval
already maps taxonomies, the new work must publish its mapping before scoring
and report native-taxonomy results, rather than claiming mapping itself as a
contribution.

### Llama Guard

Llama Guard supports one-versus-all policy prompting and reports per-category
AUPRC. This establishes category-specific guard performance and configurable
taxonomies. It does not report per-category calibration.

Collision: per-category safety performance is not new. Remaining gap:
criterion-conditioned probability reliability under shift.

### Multicalibration for LLM confidence and multi-group UQ

Detommaso et al. explicitly target simultaneous calibration over intersecting
groups for LLM confidence scoring; Liu and Wu show that aggregate uncertainty
guarantees can fail within subgroups of long-form generations. These papers
are not safety-guard evaluations, but they own the general subgroup-calibration
framing.

Collision: “global calibration does not imply subgroup calibration” is not a
new thesis. Remaining gap: a safety-specific empirical object, human
multi-category labels, guard families, shift, and deployment selection.

### Hansen et al. 2024

Hansen et al. directly report maximum group-wise ECE/smECE, analyze
worst-group calibration, and show that the metric used for worst-group
selection can change the chosen method. This is the strongest collision with
the proposed certificate logic.

Collision: maximum subgroup calibration and its use in method selection are
not new. Remaining gap: overlapping safety criteria, guard models, cross-family
and cross-taxonomy replication, signed shift cancellation, and an explicit
mean-versus-worst **guard** selection comparison.

### Wu et al. 2024

Wu et al. connect multicalibration to OOD generalization under and beyond
covariate shift. The new paper cannot claim a general theoretical connection
between subgroup calibration and shift.

Remaining gap: an empirical safety-guard stress-test and deployment protocol,
not a new multicalibration theory.

### Fair and calibrated toxicity detection, 2026

This preprint reports an aggregate ECE of 0.013 alongside materially worse
identity-subgroup ECE, with bootstrap intervals. It is direct evidence that
aggregate calibration can hide subgroup failures in a safety-adjacent
moderation problem.

Collision: the aggregate-hides-subgroup phenomenon is already explicit even
in toxicity moderation. Remaining gap: safety **criteria** rather than identity
groups, guard-model distribution shift, cross-taxonomy replication,
criterion-shift cancellation, and guard-ranking consequences. This work must
be cited if the external study proceeds.

## 5. Data and construct feasibility discovered during the audit

No guard outcome was scored or inspected. Dataset schemas were checked only to
avoid writing an impossible preregistration.

- BeaverTails exposes human safe/unsafe judgments and fourteen category flags,
  making it a strong primary candidate.
- AEGIS 2.0 exposes prompt/response labels, label-source fields, and
  `violated_categories`. A primary human-data analysis must use only rows whose
  relevant label source is human and must first verify that the category field
  belongs to that same annotated unit.
- WildGuardTest has human binary prompt/response labels, but the commonly
  exposed test schema does not provide a reliable per-item risk category.
  It cannot count as a primary criterion benchmark unless native category
  metadata is recovered and frozen before any guard scoring.
- Public benchmarks do not provide paired base/adapted policy responses.
  External conclusions are limited to benchmark, collection, task, or
  response-source shift.

## 6. Surviving novelty and prohibited claims

Allowed, conditional on external confirmation:

- criterion-conditioned calibration profiles of safety guards under
  distribution/response-source shift;
- cancellation and worst-criterion diagnostics as registered audit statistics;
- a simultaneous criterion-level stress-test/certificate;
- evidence that mean and worst-criterion objectives select different guards;
- cross-family and cross-taxonomy replication.

Prohibited:

- policy-adaptation generalization from public benchmarks;
- calling the Qwen reference annotations human ground truth;
- claiming global-versus-subgroup calibration as novel;
- claiming maximum/worst-group calibration as novel;
- claiming per-category safety evaluation as novel;
- reviving P5, mean degradation, or any frozen PCCD gate;
- choosing taxonomies, shifts, guards, criteria, or mappings after outcome
  inspection.

## 7. Final collision verdict

**GAP_SURVIVES.** ICLR 2025 does not cover both per-category cancellation and
mean/worst-criterion guard rank reversal, so the specified NO-GO trigger does
not fire.

The result is not an unconditional GO. The external preregistration must solve
four feasibility risks before it can be locked:

1. two genuinely human-labelled, category-resolved benchmark populations;
2. exact natural source-to-target shift definitions;
3. defensible and reproducible probability extraction for three guard
   families; and
4. sufficient positive and benign support for simultaneous criterion
   calibration.

If any of those fails in a metadata-only audit, the pivot stops before model
scoring. If they pass and PaperGuru locks the protocol, the project may proceed
as a narrow external validation study.
