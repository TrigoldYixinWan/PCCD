# PaperGuru Review Request: “Aggregate Calibration Is Not a Safety Certificate”

**Status:** proposal for literature and feasibility review; not a change to any frozen verdict

**Requested decision:** `GO`, `REVISE`, or `NO-GO` before any new experiment is registered

**Current confirmatory verdict remains:** `CORE_NOT_ESTABLISHED; P8-C NOT_REACHED`

## 1. Executive recommendation

I recommend considering a controlled thesis pivot from:

> policy adaptation causes an average deterioration in safety-critic calibration, which can then be repaired,

to:

> **aggregate calibration can conceal large, opposite-signed, safety-criterion-specific shifts; therefore an aggregate calibration score is not a sufficient deployment certificate for multi-criterion safety critics.**

The proposed paper would study calibration as a **vector over safety criteria**, rather than treating its mean as the complete object of interest. Its practical contribution would be a criterion-level stress-test and reporting protocol that detects:

1. criteria whose calibration worsens despite a stable or improved average;
2. cancellation between improving and worsening criteria;
3. model-selection rank reversals between mean and worst-criterion calibration;
4. whether these profiles reproduce across guard families, model sizes, taxonomies, and distribution shifts.

This is the best route I currently see that preserves scientific honesty and gives the surprising result a coherent purpose. It does **not** rescue the rejected mean-degradation claim, does **not** revive P5, and does **not** treat the Qwen teacher as human ground truth.

The recommendation is conditional. A closely related ICLR 2025 paper already studies calibration across nine LLM guard models and twelve benchmarks. The pivot is viable only if a literature audit confirms that **criterion-wise cancellation, worst-criterion certification, and aggregate-versus-criterion rank reversal** are not already substantially covered.

## 2. Why this pivot follows from the evidence

The independent confirmation did not establish the original core:

- D0 mean ECE: **0.039425**, 95% CI **[0.037866, 0.045538]**.
- New D5 minus D0 mean criterion ECE: **-0.006183**, 95% CI **[-0.011054, 0.000092]**.
- Therefore, the preregistered claim of positive average calibration degradation was not supported.

At the same time, the criterion-level responses were strongly nonuniform:

| Criterion | D5 minus D0 ECE | 95% CI |
|---|---:|---:|
| H2 | +0.013275 | [0.006571, 0.021814] |
| H4 | +0.038554 | [0.028546, 0.051662] |
| H5 | -0.027972 | [-0.039716, -0.009937] |
| T1 | -0.027106 | [-0.041333, -0.008937] |
| T2 | -0.055592 | [-0.069375, -0.038073] |

The registered interaction diagnostics were large:

- criterion interaction Wald statistic: **175.270**;
- permutation p-value: **0.00009999**;
- cross-criterion SD of ECE changes: **0.024828**, 95% CI **[0.020943, 0.028966]**.

The old and independently regenerated D5 criterion-delta vectors were also directionally stable:

- Spearman correlation: **0.9515**, 95% CI **[0.8061, 0.9879]**.

These statistics do not prove that the observed criterion pattern is a human-grounded property of safety. The current Qwen reference, prevalence sensitivity, and incomplete human audit prevent that claim. They do, however, motivate a cleaner question:

> Can a safety critic look acceptable under an aggregate calibration summary while becoming materially less reliable on particular safety criteria?

This question is not dependent on the failed average-degradation hypothesis. It can be tested independently on existing human-annotated safety data.

## 3. Candidate paper story

### Candidate thesis

Multi-criterion safety critics should not be certified by a single aggregate calibration statistic. Distribution shift can rotate their criterion-level calibration profile: some criteria improve, others deteriorate, and the average can remain stable through cancellation. A deployment-relevant evaluation therefore needs criterion-wise uncertainty, worst-criterion risk, and rank-robust model selection.

### Proposed contribution package

1. **Empirical phenomenon:** demonstrate aggregate/criterion disagreement under multiple naturally occurring or constructed distribution shifts.
2. **Cross-system generality:** test multiple guard families and sizes, not only the current Qwen critic.
3. **Human-grounded validation:** make existing human-annotated benchmarks the primary evidence.
4. **Evaluation protocol:** release a “Criterion Calibration Stress Test” with simultaneous confidence intervals, cancellation diagnostics, and mean-versus-worst selection comparisons.
5. **PCCD case study:** retain the current preregistered experiment as a transparent case that generated the hypothesis, with its negative mean result and reference-label limitations intact.

### Candidate title

> **Aggregate Calibration Is Not a Safety Certificate: Criterion-Specific Reliability Shifts in LLM Guard Models**

Alternative:

> **When Average Calibration Hides Safety-Criterion Failures in LLM Guards**

## 4. Claims that would be allowed

Subject to successful external validation, the paper could claim:

- aggregate calibration does not reliably summarize criterion-specific reliability under shift;
- criterion-specific ECE changes can have opposite signs and cancel in the mean;
- guard rankings can differ when evaluated by average versus worst-criterion calibration;
- criterion-wise reporting exposes deployment risks missed by aggregate summaries;
- the effect appears across specified guard families, sizes, datasets, and shifts.

The paper must not claim:

- that policy adaptation generally worsens mean calibration;
- that the current Qwen-labelled PCCD data constitute human ground truth;
- that all observed criterion differences are intrinsic rather than prevalence- or support-mediated;
- that the elementary fact “global calibration need not imply subgroup calibration” is itself novel;
- that a criterion profile generalizes across models before cross-model evidence exists;
- that the study provides a repair unless a separately preregistered intervention succeeds.

## 5. Formal reporting objects

For criterion \(p\), shift/domain \(d\), and a fixed calibration estimator:

\[
\Delta_{p,d} = \operatorname{ECE}_{p,d} - \operatorname{ECE}_{p,\mathrm{source}}.
\]

Report at least:

\[
\overline{\Delta}_d = \frac{1}{P}\sum_p \Delta_{p,d},
\qquad
W_d = \max_p \Delta_{p,d},
\]

\[
H_d = \operatorname{SD}_p(\Delta_{p,d}),
\qquad
C_d = \frac{1}{P}\sum_p |\Delta_{p,d}| -
\left|\frac{1}{P}\sum_p \Delta_{p,d}\right|.
\]

Here:

- \(\overline{\Delta}\) is mean criterion drift;
- \(W\) is worst-criterion deterioration;
- \(H\) is cross-criterion heterogeneity;
- \(C\) is a cancellation diagnostic.

The cancellation quantity follows directly from the triangle inequality and should be presented as an audit statistic, not a novel theorem. Likewise, the fact that a small mean does not upper-bound the largest criterion shift is elementary. Any theoretical contribution must go beyond these facts—for example, finite-sample simultaneous certification under sparse, overlapping safety criteria or a principled connection between shift structure and criterion calibration.

The proposed deployment certificate should require:

- simultaneous or family-wise uncertainty over all criteria;
- a prespecified maximum allowed positive criterion drift;
- minimum positive/negative class support per criterion;
- prevalence and support sensitivity;
- disclosure when mean and worst-criterion model rankings disagree.

## 6. Primary external validation without new human annotation

If we do not conduct the pending human audit, the current Qwen-labelled experiment must become secondary evidence. The main results should use existing human-annotated labels under each benchmark’s native taxonomy.

Candidate resources for PaperGuru to verify:

| Resource | Potential role | Key caution |
|---|---|---|
| Aegis Safety Dataset 1.0 | manually annotated interactions; multi-category evaluation | verify accessible model scores and exact label provenance |
| AEGIS2.0 | 34,248 hybrid human/jury-labelled interactions with a detailed taxonomy | distinguish directly human-labelled from jury-expanded portions |
| WildGuardTest | 5,299 human-annotated moderation items across risk categories | inspect per-category support and response-harm labels |
| BeaverTails | large safety-labelled QA corpus | verify whether labels and splits support calibration rather than only classification |

Primary references:

- AEGIS2.0, NAACL 2025: <https://aclanthology.org/2025.naacl-long.306/>
- WildGuard, NeurIPS 2024 Datasets and Benchmarks: <https://papers.nips.cc/paper_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract-Datasets_and_Benchmarks_Track.html>
- BeaverTails: <https://arxiv.org/abs/2307.04657>
- Aegis Safety Dataset 1.0: <https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0>

Candidate guard families include Llama Guard, ShieldGemma, WildGuard, and Aegis models, subject to:

- access to logits or defensible probability scores;
- adequate category support;
- license compatibility;
- a clear relationship between model outputs and each dataset’s taxonomy.

Taxonomies should be evaluated natively. Do not merge unrelated categories merely to manufacture consistency. Cross-taxonomy synthesis may compare higher-level statistics such as mean/worst disagreement, but should not assert that differently named categories are identical without a registered mapping and sensitivity analysis.

### Required scope correction

Public safety benchmarks may not contain paired responses from a base policy and its adapted descendants. If so, external validation can establish a result about **distribution shift, response-source shift, or benchmark shift**, not specifically about policy adaptation. The adaptation-specific claim should remain limited to evidence that directly observes paired base/adapted outputs.

## 7. Minimum empirical design

Before running, register:

1. at least two human-annotated benchmarks with sufficiently different taxonomies;
2. at least three guard systems spanning more than one model family and preferably more than one scale;
3. the exact source-to-target shifts;
4. native criteria and all inclusion/support rules;
5. calibration estimators, binning, class definition, and score extraction;
6. simultaneous confidence intervals and multiplicity control;
7. the primary mean-versus-worst model-selection comparison;
8. all prevalence, support, and taxonomy-mapping sensitivity analyses.

Recommended primary hypotheses:

- **H-Aggregate:** mean calibration drift alone fails to identify all criteria with material positive drift.
- **H-Cancellation:** at least one preregistered target domain has a positive cancellation gap with a confidence interval excluding a small practical threshold.
- **H-Rank:** guard selection by mean ECE disagrees with selection by worst-criterion ECE in a preregistered fraction of domains.
- **H-Generalization:** the above pattern appears in at least two datasets and two model families.

Recommended null-safe reporting:

- publish every criterion and domain, including null or reversed cases;
- use paired/bootstrap uncertainty at the item and criterion levels as appropriate;
- report class support and N/A prevalence beside every ECE;
- add Brier score, log loss, and adaptive ECE sensitivity;
- separate discrimination failures from calibration failures;
- avoid choosing domains or criteria after seeing which ones produce the strongest cancellation.

## 8. Novelty collision that must be resolved first

The most important comparison is:

> **On Calibration of LLM-based Guard Models for Reliable Content Moderation**, ICLR 2025
>
> <https://proceedings.iclr.cc/paper_files/paper/2025/hash/a99f732df9b668284b449da0214a3286-Abstract-Conference.html>

It evaluates nine guard models over twelve benchmarks, studies overconfidence, jailbreak-induced miscalibration, response-model shift, and post-hoc calibration. PaperGuru should inspect the full paper, supplement, code, and tables to answer:

1. Does it report calibration by safety category, or mainly by benchmark/task?
2. Does it explicitly test cancellation between criterion-specific calibration changes?
3. Does it compare average and worst-criterion guard rankings?
4. Does it propose a criterion-wise deployment certificate with simultaneous uncertainty?
5. Does it study stability or transfer of a criterion-calibration vector across models or shifts?

If the answer is substantially “yes” to items 2–4, this pivot is likely too incremental unless we contribute a materially stronger theoretical or benchmark component.

The second major collision is multicalibration:

- Hébert-Johnson et al., ICML 2018: <https://proceedings.mlr.press/v80/hebert-johnson18a.html>
- Wu et al., NeurIPS 2024, *Bridging Multicalibration and Out-of-Distribution Generalization*: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/859b6564b04959833fdf52ae6f726f84-Abstract-Conference.html>

The general lesson that global calibration does not guarantee calibration on identifiable subgroups is established. The paper needs a safety-specific contribution beyond relabelling categories as groups. Candidate novelty could come from:

- structured, overlapping safety criteria rather than demographic groups;
- paired response/policy shifts and their criterion-calibration geometry;
- evidence that aggregate reporting changes guard selection;
- an auditable certificate and benchmark protocol tailored to guard deployment;
- cross-taxonomy replication and support-aware uncertainty.

PaperGuru should also check recent guard benchmarks and category-adaptive safety work, including any 2026 preprints, for contemporaneous overlap before approving the pivot.

## 9. AAAI publication assessment

AAAI evaluates significance, novelty, technical soundness, and clarity. Its AI Alignment track explicitly welcomes empirical robustness evaluation, scalable oversight, practical evaluation tools, and reproducible resources: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>

My current conditional assessment:

- **Current evidence alone:** weak for AAAI, because the mean confirmatory claim failed and the criterion interaction is not yet independently human-grounded.
- **With only a renamed narrative:** still weak; reviewers would likely view it as post-hoc reframing.
- **With preregistered human-labelled, cross-guard validation plus a released audit protocol:** potentially credible for the AAAI AI Alignment track or the main empirical track.
- **With robust mean/worst rank reversals across datasets and model families:** substantially more attractive, because the result changes evaluation and deployment decisions rather than merely describing variance.
- **If the ICLR 2025 paper already contains the same criterion-level result:** likely not competitive without an additional theoretical, dataset, or adaptation-specific contribution.

The strongest paper is not “we finally found a positive result.” It is:

> a failed preregistered average-effect claim exposed a structurally important weakness in how multi-criterion safety calibration is summarized; the revised claim is then independently tested on human-labelled benchmarks and multiple guard families.

That sequence is scientifically defensible only if the new hypotheses are preregistered before inspecting the external results.

## 10. Requested PaperGuru literature and feasibility review

Please return:

1. **Verdict:** `GO`, `REVISE`, or `NO-GO`.
2. **Novelty map:** 10–15 closest papers, with a claim-by-claim comparison.
3. **Collision assessment:** detailed comparison with the ICLR 2025 guard-calibration paper and multicalibration/OOD literature.
4. **Data/model matrix:** exact datasets, splits, guard checkpoints, score extraction, taxonomy, licenses, and estimated compute.
5. **Claim boundary:** whether the defensible general claim is policy adaptation, response-source shift, benchmark shift, or a combination.
6. **Minimum acceptance package:** the smallest experiment set that would be persuasive for AAAI rather than merely exploratory.
7. **Proposed title and abstract:** one conservative and one ambitious version.
8. **Registered analysis plan:** if `GO`, draft the next preregistration with hypotheses, thresholds, exclusions, and multiplicity control.
9. **Stop conditions:** explicit results or literature collisions that should terminate this pivot.

## 11. Decision rubric

### GO

Proceed only if all are true:

- no prior work already establishes the exact criterion-cancellation and mean/worst rank-reversal contribution for guard calibration;
- at least two human-labelled multi-category benchmarks are usable;
- at least three guards from multiple families expose defensible confidence scores;
- category support permits simultaneous criterion-level inference;
- the paper can contribute an actionable audit/certificate, not only a descriptive plot;
- all new primary hypotheses can be frozen before external result inspection.

### REVISE

Revise the proposal if:

- the general cancellation phenomenon is known, but an adaptation-specific, cross-taxonomy, or criterion-vector-stability contribution remains open;
- only some datasets provide sufficient probability and category support;
- the empirical contribution is strong but the formal contribution is elementary;
- the best venue is an AAAI workshop or a benchmark/evaluation track rather than the main track.

### NO-GO

Stop this pivot if:

- prior guard-calibration work already reports the same criterion-level cancellation and model-selection consequence;
- usable human-labelled datasets do not expose adequate per-category support;
- guard outputs cannot be converted into defensible confidence scores;
- the effect disappears under human labels, proper support control, or alternative calibration metrics;
- the story would require post-hoc taxonomy grouping or selective domain reporting.

## 12. Frozen boundaries

- Do not change any existing P2–P8, G1–G6, or confirmatory verdict.
- Do not run another rescue analysis on the same lockbox.
- Do not call the current Qwen reference human ground truth.
- Do not revive P5 or claim mean degradation.
- Do not start P8-C from this proposal.
- Do not inspect external outcome results before the new hypotheses and exclusions are registered.
- Do not present elementary subgroup-calibration facts as theoretical novelty.
- Treat any final thesis change as Red-level and human/PaperGuru-approved.

## 13. Proposed decision

My recommendation to PaperGuru is:

> **Conduct the literature/data feasibility audit first. If and only if the exact criterion-cancellation and mean/worst rank-reversal gap survives that audit, approve a preregistered cross-guard study using existing human-labelled datasets, while retaining PCCD as a transparent hypothesis-generating negative case.**

This route does not guarantee a positive result, but it offers the best remaining chance of a rigorous, useful, and AAAI-relevant contribution without manufacturing support for a failed claim or requiring new manual annotation.
