# PCCD independent confirmation package — LOCKED

Locked: 2026-07-16, before creation of any new policy response, reference label,
critic logit, or aggregate result.

Authority: Codex acting under the project owner's explicit temporary delegation
of PaperGuru's scientific decision authority. The delegation and every design
change are recorded in `reports/DECISION_LOG.md`.

This protocol supersedes the **unexecuted draft** `reports/PREREG_G6.md`. It
does not supersede, reopen, or modify any P1-P7 or L1-L3 result. Old G2/P7 data
are development evidence for new methods. Only the lockbox below can produce a
new confirmation verdict.

## 1. Questions and evidence order

The package has three ordered questions.

1. **P2-C — calibration transport failure.** Does a newly trained D5 adaptation
   instance increase the frozen critic's mean three-way ECE on new prompt
   families relative to the base policy?
2. **P3-C — criterion-specific drift.** Conditional on P2-C, are the ten
   criterion-level ECE changes inconsistent with a common shift?
3. **P8-C — low-shot structured repair.** Conditional on P2-C and a replicated
   base-calibration anchor, does published Structured Matrix Scaling (SMS), fit
   on 500 target prompts, outperform per-criterion temperature scaling and
   reach the locked operational tolerance without harming discrimination?

P8-C is auxiliary. Its failure cannot negate P2-C/P3-C. Its success cannot
rescue a failed P2-C. P3-C may be reported descriptively when P2-C fails, but it
does not receive a confirmatory verdict.

The Qwen2.5-32B teacher is called the **fixed reference annotation protocol**.
It is not described as an oracle or human ground truth.

## 2. Frozen prior artifacts and forbidden reuse

The following remain read-only:

- D0 critic checkpoint and manifest;
- Day-2 train/calib/test/audit/conflict pools and labels;
- the 512 hidden-violation adaptation pairs;
- old D0-D6 adapters, generations, labels, logits, and analyses;
- P7 TARGET-CALIB/TARGET-TEST IDs and results.

Every prompt exposed to the Day-2 or G2 construction is excluded, including all
31,892 `all2_candidates`; the 28,892 G2 candidates not selected for final G2 are
not fresh because their base responses/labels entered support selection.

No old verdict is recomputed under the rules below. An SMS/SVS run on old P7
data is a registered development/software-validation analysis only.

## 3. Outcome-blind prompt-family lockbox

### 3.1 Size, sources, and strata

Freeze exactly 4,000 prompts:

| source / stratum | total | TARGET-CALIB | CONFIRM-TEST |
|---|---:|---:|---:|
| PKU H1 proxy | 240 | 30 | 210 |
| PKU H2 proxy | 240 | 30 | 210 |
| PKU H3 proxy | 240 | 30 | 210 |
| PKU H4 proxy | 240 | 30 | 210 |
| PKU H5 proxy | 240 | 30 | 210 |
| PKU general fill | 1,140 | 143 | 997 |
| UltraFeedback | 1,180 | 147 | 1,033 |
| new soft S1 family | 160 | 20 | 140 |
| new soft S2 family | 160 | 20 | 140 |
| new soft S3 family | 160 | 20 | 140 |
| **total** | **4,000** | **500** | **3,500** |

The PKU proxies use only vendor-provided raw harm-category metadata and the
frozen H1-H5 mapping. They do not use a newly generated response, reference
label, or critic output. Multi-category PKU prompts are assigned once by the
fixed H1, H2, H3, H4, H5 selection order; each stratum takes the first unused
eligible prompts in salted-hash order. The general fill then takes the first
remaining unused PKU prompts. UltraFeedback is selected in salted-hash order.

The soft prompts come from a new, committed task/context/template bank. It may
reuse the meanings of the locked S1-S3 policies but not an old base task,
scenario, or rendered prompt. Each axis contributes exactly 160 distinct base
task families. The code and template bank are frozen before generation.

The source proportions approximate the authoritative Day-2 source mixture.
The larger test set is an outcome-blind power choice: the maximum calibration
budget is 500, so all remaining affordable prompts increase independent-test
precision. No sample-size increase is allowed after outcomes are opened.

### 3.2 Canonical and lexical family definition

Canonicalization is deterministic:

1. Unicode NFKC;
2. lowercase;
3. replace punctuation/whitespace runs with one ASCII space and strip;
4. for soft prompts only, remove the terminal style-request clause before
   family construction so style variants of the same base task share a family.

For every historical and candidate prompt, form the set of contiguous five-word
shingles; a prompt shorter than five words has one shingle equal to its full
canonical token sequence. Construct an undirected graph with an edge when two
prompts have shingle Jaccard similarity at least `0.85`. A query family is a
connected component of this graph. The implementation may use an exact
prefix-filter similarity join, but may not use an approximate nearest-neighbor
rule that can silently miss a qualifying edge.

Any component containing a historical prompt is excluded in full. Only one
prompt may be selected from any remaining component. Family hashing is
source-independent, so cross-source duplicates collide. The builder fails
closed if an expected historical artifact is absent and records hashes/counts
for every exclusion input.

### 3.3 Selection, split, and freezing

- Family/selection/split seed: `20260723`.
- Within each fixed stratum, order by
  `SHA256("20260723:" + family_id + ":" + canonical_prompt)`.
- Apply the exact TARGET-CALIB counts in the table; the rest form CONFIRM-TEST.
- No reserve set and no content-quality replacement. Empty, refused, truncated,
  or unusual model responses are real outcomes. A process crash may resume the
  same IDs and seed, but may not substitute prompts.
- Before any generation, write and hash the 4,000-row prompt file, family map,
  all exclusion inputs, 500-ID calibration manifest, 3,500-ID test manifest,
  source/stratum counts, code commit, and environment versions.
- TARGET-CALIB and CONFIRM-TEST are disjoint by both ID and family and exhaust
  the 4,000 prompts.

The study may claim **canonical and lexical-near-duplicate family separation**,
not semantic-domain independence. No pre-existing semantic family ID exists in
PKU or UltraFeedback.

## 4. Independent adaptation and response generation

### 4.1 Primary D5 seed replicate

Train exactly one new D5 adapter:

- base policy: the same frozen Qwen2.5-7B instance;
- the exact frozen 512 hidden-violation pairs;
- chosen-only SFT;
- LoRA rank 32, alpha 64, dropout 0.05;
- q/k/v/o and gate/up/down targets;
- four epochs, effective batch 32, bf16, max length 1,024;
- AdamW learning rate `2e-4`, cosine schedule, 3% warmup;
- seed `20260723`;
- every setting other than seed must equal the old frozen D5 metadata.

The adapter is trained once and hashed before lockbox generation. A crash that
produces no usable checkpoint may resume/restart the identical run. Loss, KL,
generation quality, or downstream behavior may never justify another seed or
checkpoint.

### 4.2 Paired domains

Generate one response for every lockbox prompt from:

- D0 base policy;
- old frozen D5 adapter (secondary prompt-generalization replicate);
- new-seed D5 adapter (primary confirmation domain).

All use temperature `1.0`, top-p `1.0`, maximum 256 generated tokens, and seed
`20260723`. D0 and D5 calls use the same ordered prompt manifest. IDs, source,
prompt, and family metadata must remain aligned. The old D5 result is always
secondary and cannot replace the new-seed primary.

## 5. Reference labeling and frozen-critic scoring

- Reference model/prompt/taxonomy: exactly the frozen Day-2 joint ten-criterion
  label-only protocol.
- Decoding: temperature 0, maximum 256 tokens, **zero retries**. A malformed
  result stays missing; there is no format repair after seeing the item.
- Critic: exact frozen D0 checkpoint, read-only, emitting three logits for each
  of ten criteria.
- Every domain must have strict ten-key JSON success at least 99%, every
  domain×criterion missing rate at most 1%, unique IDs, exact prompt alignment,
  and no family leakage. Otherwise the affected core package is
  `NON_EVALUABLE`; missing labels are never imputed.
- CONFIRM-TEST aggregate metrics remain unopened until method code, dependency
  versions, analysis code, and the blinded human-audit sample are hashed.

## 6. P2-C primary analysis

Use only the 3,500 CONFIRM-TEST families and new-seed D5 versus D0.

For each criterion, compute three-class top-label ECE including N/A:

- probabilities are softmax of frozen-critic logits;
- confidence is maximum probability;
- correctness is argmax equal to the reference state;
- 15 equal-width bins with fixed edges `0, 1/15, ..., 1`.

Let `delta_p = ECE(D5,p) - ECE(D0,p)` and let `mean_delta` equally weight the ten
criteria. Uncertainty uses 10,000 paired query-family bootstrap replicates,
seed `20260733`; a resampled family retains all domains and ten criteria.
Criteria are fixed and never resampled. Report two-sided percentile 95% CIs.

The base anchor is replicated iff the 95% CI upper bound for mean D0 ECE is at
most `0.05`. This is an operational well-calibrated region, not the observed P1
mean (`0.028904`).

P2-C is confirmed iff:

1. the package is evaluable;
2. the base anchor is replicated; and
3. the 95% CI lower bound of `mean_delta` is greater than zero.

Also tag the effect `MATERIAL_GE_0.01` if the CI lower bound exceeds `0.01`.
This tag is interpretive, not a second opportunity to pass.

## 7. P3-C criterion interaction

P3-C is confirmatory only when P2-C passes. Let `delta` be the fixed vector of
ten criterion changes and let `C` be the deterministic 9×10 orthonormal Helmert
contrast matrix (`C 1 = 0`). Estimate the covariance of `delta` from the P2
bootstrap and compute

```text
W = (C delta)' (C Sigma C')^+ (C delta)
```

with Moore-Penrose `rcond=1e-12`. The p-value uses the same 10,000 replicates
recentered under a common-shift null:

```text
W_b = [C(delta_b - delta)]' (C Sigma C')^+ [C(delta_b - delta)]
p = (1 + count(W_b >= W)) / 10001
```

Rank below nine is `NON_EVALUABLE`; no substitute heterogeneity gate is used.
P3-C confirms at `p < 0.05`.

Report cross-criterion SD and its family-bootstrap CI. Tag heterogeneity
`MATERIAL_GE_0.01` when the SD CI lower bound exceeds `0.01`. Use a bootstrap
max-|t| simultaneous 95% interval for centered effects
`h_p = delta_p - mean_delta` to localize criteria; localization is not another
gate. Also report old-D5/new-prompt results and the old-versus-new D5 delta-vector
Spearman correlation as secondary replication evidence.

## 8. Locked sensitivity analyses

These cannot rescue a primary failure.

1. **Bidirectional prevalence standardization.** Within each criterion, weight
   D5 to D0 reference-state prevalence and separately D0 to D5 prevalence. Do
   not clip weights; recompute weights in every bootstrap. Report effective
   sample size. A zero state or ESS below 100 is underpowered. Tag the primary
   result `PREVALENCE_ROBUST` only when both standardized mean-delta CI lower
   bounds exceed zero.
2. ECE with per-criterion bin edges frozen from source base-calib confidence
   quantiles and applied unchanged to D0/D5.
3. Adaptive ECE; one-vs-rest classwise ECE and macro average;
   violated-vs-rest ECE.
4. NLL and multiclass Brier score. These are proper scores but conflate
   calibration and refinement/discrimination, so they are consistency evidence.
5. One-vs-rest calibration intercept/slope.
6. D0/D5 confidence and class-probability support overlap.

If primary ECE passes but prevalence-standardized, classwise, and proper-score
results materially disagree, the manuscript must call the effect metric- or
prevalence-dependent rather than a general calibration-map failure.

## 9. P8-C published structured recalibration

### 9.1 Frozen implementation

Primary method: `probmetrics==1.3.0` `SMSCalibrator` with published defaults:

- temperature-scaling-with-Laplace-mixture preprocessing (`ts-mix`);
- structured matrix map with ridge penalties;
- `rho=1`, `tau=1`;
- `lambda_intercept=lambda_diagonal=lambda_off_diagonal=1`;
- BFGS optimizer;
- the implementation's `k^rho / n^tau` and off-diagonal scaling;
- package wheel/version/hash and constructor representation recorded.

This is the method described by Berta et al., *Structured Matrix Scaling for
Multi-Class Calibration* (AISTATS 2026; arXiv 2511.03685) and its Apache-2.0
reference implementation. It is not the underidentified diagonal-plus-bias map
from the old G6 draft.

Secondary method: the same package/version's `SVSCalibrator` defaults. It may
explain bias-variance behavior but cannot replace SMS for the primary verdict.

Comparator: refit P7 per-criterion scalar temperature on the identical target
calibration resample using the locked G5 bounded-NLL formula. Raw logits and
frozen source-T are also reported. No method or hyperparameter is selected on
CONFIRM-TEST. The official defaults are frozen from external evidence, not from
old P7 outcomes.

### 9.2 Budget and two-stage bootstrap

- Unique primary budget: all 500 TARGET-CALIB families.
- Budgets 50/100/200 use source/stratum-balanced frozen prefixes and are
  secondary learning curves. They cannot rescue a failed 500-shot primary.
- 10,000 two-stage replicates, seed `20260724`.
- Stage 1 resamples 500 calibration families and refits per-criterion T, SMS,
  and SVS on the same rows in every replicate.
- Stage 2 independently resamples 3,500 test families; every method and
  criterion uses the same test rows.
- All SMS-vs-T and method-vs-raw intervals come directly from paired replicate
  contrasts, never from subtracting separate confidence intervals.
- Primary unresampled numerical failure receives one identical-start retry at
  the package's higher supported iteration limit if available. No optimizer,
  penalty, label weighting, or method switch is allowed. Overall primary fit
  failure above 1% of bootstrap fits or above 5% for any criterion makes P8-C
  `NON_EVALUABLE`.

### 9.3 Metrics and locked success conditions

On CONFIRM-TEST at budget 500, report mean and per-criterion ECE, NLL, Brier,
applicable-only violated-positive F1, and violated-vs-satisfied AUROC. The P8-C
conditions are:

- **A — raw reduction:** paired 95% CI lower bound for
  `mean ECE(raw) - mean ECE(SMS)` is greater than zero;
- **B — tolerance:** the 95% CI upper bound of mean SMS ECE is at most `0.05`;
- **C — discrimination non-inferiority:** mean F1 change versus raw CI lower is
  at least `-0.005`; mean AUROC change CI lower is at least `-0.01`; and every
  criterion's simultaneous max-|t| AUROC-change lower bound is at least `-0.02`;
- **D — incremental value:** paired 95% CI lower bound for
  `mean ECE(per-criterion T) - mean ECE(SMS)` is greater than zero.

If any criterion's CONFIRM-TEST labels do not contain both satisfied and
violated, its F1/AUROC guard is underpowered. P8-C is then `NON_EVALUABLE`; the
criterion is not deleted. Report target-calib NLL, target-test NLL, their
same-unit optimism gap, fit failures, class support, and SMS parameters.

## 10. Mutually exclusive verdicts

### 10.1 Core

Apply in order:

1. `NON_EVALUABLE`: integrity, family separation, alignment, parse, or rank rule
   fails;
2. `BASE_ANCHOR_NOT_REPLICATED`: mean D0 ECE CI upper is above 0.05;
3. `P2_CONTRADICTED`: mean-delta CI upper is at most zero;
4. `CORE_NOT_ESTABLISHED`: mean-delta CI contains zero;
5. `P2_ONLY`: P2-C passes but P3-C p is at least 0.05;
6. `P2_P3_CONFIRMED`: P2-C passes and P3-C p is below 0.05.

Append the materiality and prevalence tags; they do not change the unique core
verdict. Only verdict 6 supports an unqualified criterion-specific claim.

### 10.2 P8-C

P8-C is `NOT_REACHED` unless P2-C and the base anchor pass. Otherwise apply in
order:

1. `NON_EVALUABLE`: data/support/fit guard fails;
2. `SUCCESS`: A, B, C, and D all hold;
3. `CONTRADICTED_OR_HARM`: raw-reduction CI upper is at most zero, or a mean
   discrimination-change CI upper lies below its non-inferiority margin, or a
   criterion simultaneous AUROC CI upper lies below -0.02;
4. `PARTIAL_SUPPORT`: A and C hold, but B or D is not established; attach
   `TOLERANCE_NOT_ESTABLISHED` and/or `NO_INCREMENT_OVER_TEMPERATURE`;
5. `NOT_ESTABLISHED`: every other evaluable case, including an improvement CI
   spanning zero or non-inferiority that is uncertain but not demonstrably
   harmful.

Only `SUCCESS` is an effective-repair confirmation. Full/SVS results, smaller
budgets, and sensitivities may not rescue SMS.

## 11. Blinded human validity companion

Before opening aggregate results, freeze 800 prompt-response-policy cells:

- 40 cells for each of 2 primary domains (D0/new D5) × 10 criteria;
- outcome-blind seed `20260725`;
- stratify, where support permits, by reference state and critic-confidence
  tier; store stratum population, inclusion probability, and inverse-probability
  weight in a private manifest;
- annotator packet contains only random audit ID, prompt, response, canonical
  policy text, and satisfied/violated/N/A rubric;
- hide domain, adapter, source, reference label, critic output, and confidence;
- two independent annotators and third-person adjudication.

Primary audit analysis is weighted reference-versus-human disagreement with a
domain×criterion omnibus interaction and family-clustered uncertainty. Human
labels never fit a calibrator. Differential reference error limits all claims
to "relative to the fixed reference annotation protocol." No detected
interaction means only that the audit did not find differential error; it does
not convert the reference model into ground truth.

## 12. Failure exits and one-unseal rule

- Insufficient prompts under the locked family threshold: stop; do not relax it.
- New D5 failure: identical resume/restart only; no new seed or hyperparameter.
- P2-C failure: do not test another adapter point, direction, metric, or lockbox
  as a replacement.
- P3-C failure: retain a confirmed mean drift if P2-C passed, but remove stable
  criterion heterogeneity from the main claim.
- P8-C failure: do not add another scaler, lambda grid, MLP, isotonic method, or
  tolerance in this project phase.
- CONFIRM-TEST is unsealed once. All primary, secondary, and null results are
  reported together with hashes and this protocol.
- Human audit is the only external dependency allowed to remain pending; model
  labels cannot substitute for it.

