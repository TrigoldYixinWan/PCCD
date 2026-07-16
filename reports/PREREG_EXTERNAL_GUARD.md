# External guard criterion-calibration study — preregistration draft

Date drafted: 2026-07-16
Status: **DRAFT — REVISE-then-LOCK. NOT LOCKED. NO EXTERNAL GUARD SCORING AUTHORIZED.**
PaperGuru adjudication of §12 is in `reports/PAPERGURU_EXTERNAL_PREREG_ADJUDICATION.md`
(GAP_SURVIVES accepted; eight items ruled). This document becomes LOCKED only in a signed
PaperGuru commit AFTER the metadata-only pre-lock gates (AEGIS provenance + support, native->
common taxonomy map with two-reviewer sign-off, guard repo/revision/verbalizer registry +
source-only comparability diagnostic, BT-COHORT relabel, substantive-domain declaration) are
completed and frozen. Until then, Codex may fill registries but may NOT score any guard.

This draft follows the surviving gap documented in
`reports/ICLR2025_COLLISION_AUDIT.md`. It has been written before inspecting
any external guard score, calibration metric, criterion drift, or model
ranking. Dataset schemas and provenance fields were checked only for
feasibility.

PaperGuru must resolve the open items in Section 12 and explicitly mark this
document `LOCKED` before any external model inference or outcome aggregation.

## 1. Claim boundary

The external study concerns only:

> criterion-conditioned calibration of LLM safety guards under benchmark,
> collection-cohort, task, or response-source distribution shift.

It does **not** test policy adaptation. Public benchmarks have no paired
base/adapted policy responses, so no external result may be described as an
adaptation effect.

PCCD remains a transparent hypothesis-generating case study with
`CORE_NOT_ESTABLISHED`. All P2-P8 and G1-G6 verdicts remain frozen. Qwen
reference annotations are not human ground truth, the old lockbox is not
reanalyzed, and P5/mean degradation are not revived.

## 2. Primary study object

Each guard produces a probability of an overall unsafe decision. For a native
safety criterion `p`, its criterion-conditioned evaluation population is:

- positive: a human-labelled unsafe item whose native annotation includes
  criterion `p`;
- negative: a human-labelled safe item from the same domain;
- excluded: unsafe items labelled only with other criteria.

Multi-label unsafe items may contribute to more than one criterion, but the
bootstrap keeps all copies of the same underlying item together. This is a
one-versus-benign calibration question:

> when the guard assigns unsafe probability `q`, is that probability reliable
> for detecting criterion `p` versus genuinely safe content in this domain?

This estimand permits the same binary probability interface across guard
families. It must be described as **criterion-conditioned binary
calibration**, not as calibration of a guard's internally generated category
label.

Primary criterion results use a common 50:50 positive/benign target prevalence
through deterministic weighting, so differences are not mechanically driven
by category prevalence. Natural-prevalence results and alternative benign-pool
definitions are mandatory sensitivities.

## 3. Human-labelled benchmark registry

The proposed primary benchmarks are:

1. **BeaverTails**: human safe/unsafe response judgments plus fourteen native
   category flags.
2. **AEGIS 2.0 human-source subset**: include a prompt or response label only
   when its corresponding `*_label_source` is human. The metadata-only audit
   must verify that `violated_categories` refers to the same annotated unit.

WildGuardTest is a secondary binary-shift sensitivity only. Its common public
schema does not expose reliable per-item risk categories and therefore it
cannot satisfy the two-benchmark criterion requirement.

No benchmark may be silently replaced after guard outcomes are seen. If
AEGIS 2.0 fails the category-attribution or support audit, the study is
`NOT_LOCKABLE`; PaperGuru must approve a revised preregistration before any
guard is scored.

## 4. Source and target domain registry

The final manifests must be frozen by item ID/text hash before model inference.
The recommended comparisons are:

### BeaverTails source

- `BT-SOURCE`: deduplicated `30k_train`.
- `BT-CONTROL`: deduplicated `30k_test`; held-out control, not advertised as a
  substantive distribution shift.
- `BT-COHORT`: items in `330k_test` absent by exact and registered near-duplicate
  hash from all 30k splits. RULED (adjudication Item 1): 30k and 330k are SCALE VARIANTS of
  one collection, NOT independent cohorts. This is named a **within-collection
  sampling/annotation-round shift** and treated as a WEAK shift; substantive-shift weight rests
  on the cross-benchmark (BT<->AEGIS) and AG-RESPONSE task-shift comparisons, not on this cell.
- `BT-TO-AEGIS`: AEGIS human prompt-test items under the common taxonomy map in
  Section 5; benchmark shift.

### AEGIS 2.0 source

- `AG-SOURCE`: human-labelled prompt items in the training split.
- `AG-CONTROL`: human-labelled prompt items in the test split; held-out control.
- `AG-RESPONSE`: human-labelled response items in the test split;
  prompt-to-response/task-source shift.
- `AG-TO-BT`: BeaverTails `BT-COHORT` under the common taxonomy map; benchmark
  shift.

The two controls are always reported but do not count as evidence of
distribution-shift generalization. Each benchmark therefore has two proposed
substantive comparisons: one within-taxonomy natural shift and one
cross-benchmark shift.

If AEGIS does not have adequate human response support, `AG-RESPONSE` is not
replaced post hoc. The preregistration returns to PaperGuru for revision before
scoring.

## 5. Taxonomy discipline

Native-taxonomy results are primary for the within-benchmark comparisons.

Cross-benchmark comparisons use one mapping, frozen before guard scoring, over
the intersection of categories that can be defended from the published
definitions. The provisional common categories are:

1. violence or physical harm;
2. hate or identity-directed harm;
3. sexual content, including minors as a prespecified nested sensitivity;
4. suicide or self-harm;
5. illegal/criminal activity, with weapons and controlled substances reported
   both pooled and separate where native labels permit; and
6. privacy or personally identifying information.

Two reviewers must independently map native labels from definitions alone and
resolve disagreements before outcome access. The signed mapping table, policy
text, inclusion/exclusion notes, and SHA-256 hash become part of the lock.

Primary cross-benchmark analysis requires at least four common criteria to
meet support. No result may depend on adding, merging, or removing a category
after scoring. Native, fine-grained results remain visible beside the common
mapping.

## 6. Guard registry and probability extraction

Proposed primary guards, spanning three backbone families and two scales:

1. Meta `Llama-Guard-3-8B` (Llama family);
2. AllenAI `WildGuard` (Mistral family);
3. Google `ShieldGemma-2B` (Gemma family).

Before lock, the exact Hugging Face repository, revision hash, license, chat
template, decision verbalizers, and supported task must be entered for each.
No model is fine-tuned.

Probability extraction follows each released guard's documented classification
interface:

- normalize the logits of the complete safe/unsafe decision verbalizers at the
  first decision position;
- for a multi-token verbalizer, use the registered sequence log-probability,
  not only its first token;
- use one fixed official prompt/template per guard and task;
- use deterministic inference; malformed decisions are retained and reported,
  never retried or repaired;
- record both raw logits and normalized probability.

ShieldGemma receives one frozen omnibus safety policy that lists the benchmark
taxonomy; it is not separately tuned on each observed outcome. If any guard
lacks a defensible probability extraction or cannot support the registered
prompt/response task, the study is `NOT_LOCKABLE` rather than substituting a
model after outcome access.

RULED (adjudication Item 6): before target analysis, run a SOURCE-ONLY comparability
diagnostic — each guard's source reliability diagram + support — so that ShieldGemma's
omnibus-policy score is confirmed to be a defensible "overall unsafe probability" estimand
comparable to Llama Guard / WildGuard. If ShieldGemma's source probability is degenerate
(e.g. near-constant), it becomes a sensitivity guard and a pre-named 4th family replaces it as
primary. This protects H-Rank from an incomparable score. The diagnostic reads SOURCE only and
no target/outcome result.

A fourth guard may be declared before lock as a sensitivity model, but the
three primary guards and their order cannot change after scoring.

## 7. Support and evaluability

For a criterion to enter a primary source-to-target vector, both source and
target must contain at least:

- 100 human-labelled criterion-positive items; and
- 100 human-labelled safe negatives.

Support from 50 to 99 per class is reported as `UNDERPOWERED_DIAGNOSTIC` and is
excluded from primary vector aggregation. Support below 50 is descriptive only.
At least four criteria must be primary-eligible for a guard/domain vector.

The metadata-only support audit may inspect labels and counts but not guard
scores. It freezes:

- exact included item IDs and hashes;
- all overlap removals;
- native criterion support;
- safe-negative support;
- multi-label prevalence; and
- source/target manifests.

If fewer than two benchmarks, three guard families, or four criteria per
primary vector remain eligible, the study is `NON_EVALUABLE` and stops before
inference.

## 8. Metrics and certificate

For guard `g`, criterion `p`, source `s`, and target `d`:

```text
Delta[g,p,d] = ECE[g,p,d] - ECE[g,p,s]
M[g,d]       = mean_p Delta[g,p,d]
W[g,d]       = max_p Delta[g,p,d]
H[g,d]       = SD_p Delta[g,p,d]
C[g,d]       = mean_p |Delta[g,p,d]| - |mean_p Delta[g,p,d]|
```

`C` is an audit statistic from the triangle inequality, not a theorem.

Also define:

```text
A[g,d] = aggregate binary ECE[g,d] - aggregate binary ECE[g,s]
Miss[g,d] = W[g,d] - max(A[g,d], 0)
```

Primary ECE is 15-bin equal-mass ECE under the standardized 50:50
one-versus-benign population. Aggregate binary ECE in `A` uses the analogous
50:50 unsafe/safe standardization and the same estimator, so `A` and `W` are
on a comparable prevalence scale. Mandatory sensitivities:

- 15-bin equal-width ECE;
- adaptive ECE;
- smooth calibration error where computationally feasible;
- Brier score and log loss;
- AUROC, violated/unsafe F1, FNR, and FPR;
- natural prevalence;
- shared versus domain-specific benign pools; and
- native versus common mapped taxonomy.

Every table reports positive/negative support, multi-label rate, prevalence,
malformed-score rate, and discrimination beside calibration.

The auditable criterion certificate for a guard/domain contains:

- simultaneous 95% intervals for every `Delta[g,p,d]`;
- simultaneous upper bound on `W`;
- `M`, `H`, and `C` with uncertainty;
- aggregate drift `A` and missed-worst gap `Miss`;
- the criteria responsible for the worst positive drift;
- whether mean and worst-criterion guard selection disagree; and
- a fail-closed `INSUFFICIENT_SUPPORT` marker for excluded criteria.

## 9. Four preregistered hypotheses

Practical constants proposed for PaperGuru review:

```text
material criterion deterioration tau = 0.020 absolute ECE
near-zero aggregate tolerance epsilon = 0.010 absolute ECE
material cancellation c0 = 0.010 absolute ECE
material rank-selection margin r0 = 0.010 absolute ECE
```

### H-Aggregate

Aggregate calibration drift fails to identify a material criterion
deterioration.

Primary support requires at least one corrected guard/domain cell with either:

1. the simultaneous upper interval for `A` at or below `epsilon` and the
   simultaneous lower interval for at least one `Delta_p` above `tau`; or
2. the lower interval for `Miss` above `tau`.

All missed and non-missed cells are reported.

### H-Cancellation

At least one preregistered target domain has material signed cancellation
between criterion shifts.

Primary support requires:

- lower 95% interval for `C` above `c0`; and
- a simultaneous positive criterion shift above `tau` plus a simultaneous
  negative criterion shift below `-tau`.

The second requirement prevents small noisy sign changes from being called
cancellation.

### H-Rank

Selecting a guard by mean criterion ECE disagrees with selecting it by
worst-criterion ECE often enough to change a deployment choice.

For each target domain:

```text
g_mean  = argmin_g mean_p ECE[g,p,d]
g_worst = argmin_g max_p  ECE[g,p,d].
```

A robust rank reversal requires:

- `g_mean != g_worst`;
- the bootstrap probability of distinct winners is at least 0.95; and
- choosing `g_worst` instead of `g_mean` improves worst-criterion ECE by more
  than `r0` with a paired 95% interval excluding zero.

H-Rank is supported if robust reversal occurs in **at least two substantive target domains that
span both benchmarks AND include both a within-benchmark and a cross-benchmark shift type**
(adjudication Item 8; the uninformative "one third" is dropped because the pre-declared
substantive-domain count is small). The EXACT substantive-domain list and count are frozen in
the lock (controls excluded; BT-COHORT flagged weak). Outcome-blind power note (registered): with
three primary guards this reversal test is low-powered; a null H-Rank is reported as
INCONCLUSIVE-FOR-RANK, not as evidence that no reversal exists.

### H-Generalization

The aggregate/criterion disagreement is not specific to one benchmark,
taxonomy, or backbone family.

This conjunctive hypothesis is supported only if:

- H-Aggregate or H-Cancellation has a multiplicity-corrected positive cell in
  each of the two primary benchmarks;
- the qualifying cells involve at least two guard backbone families;
- at least one qualifying result uses a native within-benchmark taxonomy, so
  the result is not created solely by cross-taxonomy mapping; and
- the direction survives the registered prevalence and adaptive-ECE
  sensitivities.

H-Generalization has no separate uncorrected search over favorable subsets.

## 10. Uncertainty and multiplicity

- Resampling unit: original prompt/conversation/item ID. All duplicated
  criterion memberships and source/target records for that item remain in the
  same cluster.
- Replicates: 10,000, seed `20260801`.
- Within each guard × source-target comparison, use a studentized bootstrap
  max-|t| simultaneous 95% interval for the full eligible criterion vector.
- Use paired bootstrap contrasts for guards evaluated on the same item set.
- Apply Holm correction at family-wise alpha 0.05 across the top-level
  H-Aggregate, H-Cancellation, and H-Rank tests and their preregistered
  guard/domain cells.
- H-Generalization is a conjunction of corrected component findings and spends
  no additional alpha.
- Criteria are fixed scientific objects and are not resampled as if drawn from
  a population.

If fewer than 90% of bootstrap replicates are valid, a simultaneous covariance
matrix is rank-deficient, or a primary verbalizer has non-finite probability,
the affected test is `NON_EVALUABLE`.

## 11. Study-level verdict

Apply in order:

1. `NON_EVALUABLE`: human provenance, probability extraction, support,
   manifests, or inference/uncertainty checks fail.
2. `EXTERNAL_CORE_SUPPORTED`: H-Aggregate and H-Generalization are supported,
   and at least one of H-Cancellation or H-Rank is supported.
3. `NARROW_PARTIAL_SUPPORT`: H-Aggregate generalizes, but neither registered
   cancellation nor deployment rank reversal passes.
4. `EXTERNAL_CORE_NOT_ESTABLISHED`: H-Aggregate or H-Generalization fails.

Null and reversed findings remain publishable in the artifact and cannot be
removed. No second external confirmatory run, replacement benchmark, new guard,
new criterion merge, or threshold change is allowed after outcome access.

## 12. Open items PaperGuru must resolve before lock

This draft deliberately does not conceal the remaining feasibility risks.
PaperGuru must approve or revise:

1. whether BeaverTails `30k` versus exclusive `330k` is a defensible
   collection-cohort shift rather than only a dataset-version comparison;
2. whether AEGIS 2.0 `violated_categories` can be attributed to the same
   human-labelled prompt/response unit;
3. whether AEGIS human response support is adequate for `AG-RESPONSE`;
4. the exact common-taxonomy mapping and two-reviewer procedure;
5. the exact model revision and probability verbalizers for all three guards;
6. whether ShieldGemma's frozen omnibus policy is sufficiently comparable to
   the other guards' overall unsafe score;
7. the proposed support minima and practical thresholds; and
8. whether H-Rank's one-third/two-domain replication rule is appropriately
   powered.

Only metadata, documentation, and model-card interfaces may be inspected while
resolving these items. No guard output, ECE, drift, cancellation, or ranking may
be viewed before the final lock commit.

## 13. Required frozen artifacts before inference

- benchmark revisions and licenses;
- raw and deduplicated ID manifests with SHA-256;
- source/target overlap report;
- human-label provenance audit;
- criterion support table;
- native and common taxonomy maps with reviewer sign-off and hashes;
- exact guard repository revisions;
- prompt/template and verbalizer registry;
- environment lock;
- analysis code and synthetic unit tests;
- one command that scores manifests without reading aggregate outcomes; and
- a signed PaperGuru lock commit.
