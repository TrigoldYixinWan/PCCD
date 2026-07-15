# PCCD — G1 Pre-Registration Protocol (three-layer, locked before any rerun)

Status: PRE-REGISTERED 2026-07-15 by PaperGuru (human-approved). This document LOCKS the
criteria, metric definitions, and analysis choices for G1 BEFORE the diagnostic and the
single confirmatory rerun are executed. Nothing here may be changed after seeing new
results; a change after data is observed is a protocol violation and must be recorded as a
Red CHANGES entry with explicit justification.

Motivation: the first Day-3 run (reports/day3_g1.md, commit 3a76df8) revealed that the
original single G1 criterion conflated three distinct things and that its teacher-label
perturbation gate used an over-strict whole-record threshold. We therefore (a) FREEZE the
first run as a genuine measurement finding, (b) split G1 into three independently
pre-registered layers, (c) run a ~100-item diagnostic to locate the source of order/
paraphrase instability and N/A collapse before any rerun, and (d) lock all metric
definitions below. We keep the original random splits AND add a pre-registered balanced
diagnostic set; we NEVER replace data or oversample to manufacture a pass.

--------------------------------------------------------------------------------
## Layer L1 — Teacher-label RELIABILITY (annotation quality of the ground truth)
Question: are the teacher's labels stable under semantics-preserving perturbations?
This is a property of the LABELING PROCESS, not of the policies or the critic.

Primary metric (LOCKED): **policy-cell micro-agreement** = fraction of (item × policy)
cells whose label is unchanged between canonical and perturbed run, over cells where BOTH
parsed. Whole-record exact-match is REPORTED as secondary/descriptive only — it is not a
gate, because joint 10-policy exact match is mathematically over-strict (temperature-0
repeat itself was 93.5% whole-record vs 98.5% cell-micro).

Pre-registered L1 thresholds (per perturbation, on the confirmatory 400-item audit):
- repeat_sampling (temp 0):        cell-micro >= 97%   (determinism floor)
- policy_order_swap:               cell-micro >= 90%
- policy_paraphrase:               cell-micro >= 90%
All with 95% item-cluster bootstrap CIs (10,000 replicates, seed fixed). A perturbation
PASSES L1 iff its point estimate meets the threshold AND the CI lower bound >= threshold-2pp.

L1 verdict rule: PASS if all three meet threshold; PARTIAL if repeat passes but a
perturbation fails; FAIL if repeat itself < 97%. A PARTIAL/FAIL is a publishable
reliability finding, NOT a reason to relax the threshold.

### L1 confirmatory-run execution rules (LOCKED 2026-07-15 after the diagnostic)
The Day-3 diagnostic (reports/day3_diag.md) established that the first run's perturbation
failure was driven by TWO fixable MEASUREMENT bugs, not teacher instability: (a) six
paraphrase strings were non-equivalent (added scope-narrowing qualifiers) — repaired in
src/policy_defs.py `_PARAPHRASE`; (b) `policy_order_swap` used random orders, and the
teacher has a position-dependent output-schema failure (e.g. S3 in position 1 dropped its
key 10/10 times). The confirmatory 400-item audit fixes ONLY these measurement bugs and the
position assignment; it does NOT change any threshold and is run exactly ONCE.

Locked execution details:
1. Paraphrase: use the D-1 repaired equivalent strings; FREEZE the exact paraphrase text and
   its SHA-256 before the run and record both in the report.
2. Order-swap becomes a DETERMINISTIC Latin square over the 400 items: each policy appears in
   each of the 10 positions exactly 40 times (400/10). This removes random position imbalance
   and lets L1 measure semantic stability rather than a random-position artifact.
3. Parse rule (locked BEFORE the run): cell-micro agreement INCLUDES every validly parsed
   single-policy cell (a dropped key for one policy does not void the other nine cells);
   SEPARATELY report the strict 10-key JSON success rate AND the per-position / per-policy
   missing-key rate. The missing-key/position/N/A-transition tables are DIAGNOSTIC only and
   do NOT enter the L1 gate.
4. canonical and repeat remain INDEPENDENT calls at temperature 0, same model/env; NO retries,
   NO patching of teacher output, NO second confirmatory run.
5. Thresholds (repeat >=97%, order_swap >=90%, paraphrase >=90% cell-micro), the 400-item
   data, and the 10,000-replicate seeded bootstrap are UNCHANGED.

Core principle: fix only confirmed measurement bugs and the position assignment; never move a
threshold; never run a second confirmatory pass. If L1 still fails under the repaired,
Latin-square, correctly-parsed protocol, that is the genuine reliability finding and the
publication fallback applies.

--------------------------------------------------------------------------------
## Layer L2 — Teacher target-label HETEROGENEITY (are the 10 policies distinct?)
Question: do the ten policies induce distinguishable target-label distributions on the same
held-out responses? This is the UPSTREAM evidence that the policy space is non-degenerate.

Metrics (LOCKED): pairwise Stuart-Maxwell marginal-homogeneity test over all 45 policy
pairs with Holm family-wise control; supporting mean JSD (base 2) and mean TV with 95%
item-cluster bootstrap CIs. (Implementation already in scripts/day3/analyze_g1.py and
statistically verified by PaperGuru: correct SM covariance/pinv/rank df and monotone Holm.)

Pre-registered L2 criterion: reject equal marginals (Holm-adjusted p < 0.05) for
>= 40/45 pairs. The first run gave 44/45 — L2 is ALREADY MET on the frozen data.
S2--S3 (verbosity vs structure) non-separation is EXPLICITLY ACCEPTED as an expected soft-
axis correlation and does not fail L2 (the criterion is >=40/45, not 45/45). See diagnostic
D-3 which will confirm S2/S3 are genuinely correlated soft axes, not a generation artifact.

--------------------------------------------------------------------------------
## Layer L3 — D0 CRITIC behavior heterogeneity (deferred to Day-4, needs the critic)
Question: does the trained frozen critic behave differently across policies at D0 (base)?
This is the ORIGINAL BRIEF F.2 second conjunct and plan_9day.md's "per-policy F1 CV > 0.15".
It CANNOT be evaluated at Day-3 because no critic exists yet. It is deferred until the D0
critic is trained (Day-4), and its definitions are LOCKED here to prevent post-hoc choice:

- Critic prediction unit: per (item, policy) three-state prediction {satisfied, violated,
  not_applicable}, produced by the D0 frozen critic on the SAME held-out responses.
- F1 definition (LOCKED): **macro-F1 with VIOLATED as the positive class**, computed
  per-policy over APPLICABLE items only (i.e. items where teacher label != not_applicable
  for that policy). Rationale: the safety-relevant event is "violation caught"; N/A items
  are not part of the discrimination task and are excluded from the per-policy F1 denominator.
- N/A handling (LOCKED): N/A is neither positive nor negative; excluded from F1 as above,
  but the critic's N/A-vs-applicable calibration is reported separately (descriptive).
- Heterogeneity statistic (LOCKED): coefficient of variation CV = std/mean of the ten
  per-policy F1 values. Pre-registered threshold: CV > 0.15 (from plan_9day.md), with a 95%
  item-cluster bootstrap CI; PASS iff point estimate > 0.15 AND CI excludes 0.15 from below.

--------------------------------------------------------------------------------
## Overall G1 verdict rule (LOCKED)
- G1 PASS requires L2 PASS (done: 44/45) AND L3 PASS (Day-4) AND L1 PASS.
- If L1 is PARTIAL/FAIL after the diagnostic-informed rerun, G1 does NOT silently pass:
  the paper's causal claims (P2/P3/P5) are narrowed and the "multi-policy LLM critic is
  sensitive to policy wording/order" reliability finding becomes a first-class reported
  result (see reports/day3_g1.md §finding and the publication fallback in the brief).
- No threshold in this document may be moved after observing the rerun. Full stop.
