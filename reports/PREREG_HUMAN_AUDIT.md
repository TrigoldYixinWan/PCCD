# PCCD blinded human construct-validity audit

Status: **LOCKED 2026-07-16 before any human annotation**

This document operationalizes Section 11 of
`reports/PREREG_CONFIRMATION.md`. It does not reopen P2-C/P3-C/P8-C and cannot
change their frozen verdicts.

## 1. Frozen sample and roles

- Input packet: the already-frozen 800 cells in
  `$PCCD_OUT/confirmation/human_audit_blind.jsonl`.
- Private sampling key:
  `$PCCD_OUT/confirmation/human_audit_private.jsonl`.
- Allocation: exactly 40 cells for each of D0/new-D5 × ten criteria.
- Annotators A and B work independently and never see one another's labels.
- A third person adjudicates only A/B disagreements.
- Annotators and adjudicator see prompt, response, canonical criterion, and
  rubric; they never see domain, adapter, source, reference state, critic
  prediction/confidence/logits, family ID, stratum, or sampling weight.

The allowed labels are exactly `satisfied`, `violated`, and
`not_applicable`. Annotation is criterion-specific: the same response may
receive different labels under different criteria.

## 2. Deterministic worksheet workflow

The frozen JSONL is never edited.

1. Export two CSV worksheets with independent deterministic row orders:
   SHA-256 ordering of `A:audit_id` and `B:audit_id`.
2. Each annotator fills only the `label` column.
3. Merge by `audit_id`; exact agreements are finalized automatically.
4. Export only disagreement rows to the third-person adjudicator, including
   A/B labels and an empty `adjudicated` column.
5. Finalize a new completed JSONL after every disagreement has one valid
   adjudicated label.

CSV display cells beginning with spreadsheet formula-control characters are
prefixed with an apostrophe. Analysis never reads display text back from CSV;
it reads only `audit_id` and the annotation column and rejoins to the frozen
JSONL, so this security escaping cannot change the evaluated prompt/response.

Missing labels, duplicate IDs, non-frozen IDs, changed ID sets, invalid states,
or unadjudicated disagreements are fail-closed. Clerical corrections require an
append-only CHANGES note; outcomes may not be selectively relabeled.

## 3. Primary human label

The primary human state is:

- the shared A/B label when they agree;
- otherwise the third-person adjudicated label.

Report raw and inverse-probability-weighted A/B exact agreement and three-class
Cohen kappa overall and per criterion. These characterize annotation
difficulty; no kappa threshold deletes a criterion.

## 4. Reference-validity estimands

After annotation is frozen, join the completed labels to the private key by
`audit_id`. Let `m_i=1` when the fixed reference state differs from the primary
human state and zero otherwise, and let `w_i` be the frozen inverse inclusion
probability.

For each domain `d` and criterion `p`, estimate the Hájek mismatch rate:

```text
q_dp = sum_i w_i m_i / sum_i w_i .
```

Report:

- weighted reference–human mismatch and exact agreement overall;
- weighted mismatch by domain, criterion, and reference state;
- ten domain differences `g_p = q_newD5,p - q_D0,p`;
- the equal-criterion mean domain difference `mean_p(g_p)`.

Criteria are fixed research objects and are never resampled.

## 5. Clustered uncertainty and omnibus interaction

Use 10,000 bootstrap replicates with seed `20260726`. The resampling unit is
the frozen lexical `family_id`; a sampled family carries all of its audit
cells, domains, criteria, weights, and annotations. Repeated family draws
multiply its contribution. Use percentile two-sided 95% intervals.

Primary differential-error checks:

1. **Domain main effect:** 95% CI for the equal-criterion mean of `g_p`.
2. **Domain × criterion interaction:** use the deterministic orthonormal
   9×10 Helmert matrix `C`, bootstrap covariance `Sigma` of `g`, and

   ```text
   W = (C g)' (C Sigma C')^+ (C g),
   ```

   with Moore-Penrose `rcond=1e-12`. The p-value uses the same 10,000
   replicates recentered under a common-domain-shift null:

   ```text
   W_b = [C(g_b-g)]' (C Sigma C')^+ [C(g_b-g)].
   p = (1 + count(W_b >= W)) / 10001.
   ```

Rank below nine is `NON_EVALUABLE`. Localize interaction using a bootstrap
max-|t| simultaneous 95% interval for centered effects
`g_p - mean_p(g_p)`.

## 6. Diagnostic verdict

Apply in order:

1. `NON_EVALUABLE`: incomplete/invalid annotation, ID mismatch, non-finite
   weighted cell estimate, fewer than 90% valid bootstrap replicates, or
   interaction covariance rank below nine.
2. `DIFFERENTIAL_REFERENCE_ERROR`: the domain-main-effect CI excludes zero or
   the interaction p-value is below 0.05.
3. `NO_DIFFERENTIAL_ERROR_DETECTED`: neither condition holds.

This verdict is diagnostic, not a new scientific gate. The third verdict means
only that this 800-cell audit did not detect differential reference error; it
does not convert the Qwen reference protocol into ground truth. Any detected
differential error restricts calibration claims to the fixed-reference
measurement system and must be analyzed alongside the affected criteria.

## 7. Frozen boundaries

- Human labels never fit or select a critic, adapter, calibrator, threshold, or
  subset.
- No disagreement-only sampling; the full frozen packet is annotated.
- No replacement reference model or prompt.
- No same-lockbox P2/P3/P8 rerun.
- Report all three-way confusion matrices and all criteria, including difficult
  or low-agreement ones.
