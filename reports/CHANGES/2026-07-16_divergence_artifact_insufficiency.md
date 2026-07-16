# Change: Token-level alternative divergences are not identifiable from frozen artifacts

**Severity:** Red  
**Date / base commit:** 2026-07-16 / `7975afe`  
**Status:** BLOCKED; no divergence was fitted and no frozen artifact was modified.

## Trigger

`reports/MECHANISM_NOTES.md` section A and the Day-8 instruction require a
token-level plug-in estimate of

```text
chi^2(P_adapt || P_base) = E_base[(p_adapt/p_base - 1)^2]
```

using only the already stored per-item log-ratio records and without calling the
policy/base models again.

The actual schema of every frozen `$PCCD_OUT/g2/D*_kl_items.jsonl` row is:

```json
{
  "id": "...",
  "tokens": 93,
  "log_ratio_sum": 141.51907840641798,
  "log_ratio_mean": 1.521710520499118,
  "adapted_logp_sum": -146.65877899147745,
  "base_logp_sum": -288.17785690426354
}
```

`src/compute_kl.py` lines 119-132 compute the vector of per-token differences
in memory but persist only its sum and mean.  A recursive artifact search found no
separate per-token log-ratio file.

## What I changed

Nothing in the estimator, data, model, metric, or gate.  Execution stopped before
computing any alternative-divergence value.  This report is the only change.

## Why the requested value cannot be recovered

Let `ell_t = log p_adapt(y_t|h_t) - log p_base(y_t|h_t)`.  The frozen artifact
provides `sum_t ell_t` and its count.  The token-level chi-squared plug-in needs a
second/exponential moment such as a mean of `exp(ell_t)` under the appropriate
sampling distribution.  In general,

```text
mean_t exp(ell_t) != exp(mean_t ell_t)
```

and infinitely many token-level vectors share the stored mean while having
different chi-squared, reverse-KL, and TV estimates.  Therefore the requested
statistics are not identifiable from the frozen schema.

## Alternatives considered

1. **Use `exp(log_ratio_mean)-1`. Rejected.** This is a geometric-mean
   likelihood-ratio proxy, not the registered token-level chi-squared divergence.
   Labeling it chi-squared would be a scientific error.
2. **Use sequence likelihood ratios from `log_ratio_sum`. Rejected without human
   approval.** Sequence-level importance identities can be written from the stored
   sum, but this changes the estimand and units, is confounded by variable response
   length, and exponentiates large log ratios into an extremely unstable estimator.
3. **Re-run teacher forcing on the same frozen responses. Feasible only with Red
   approval.** `compute_kl.py` can be extended to persist every `ell_t`, then the
   requested token-level plug-ins can be calculated without new generations,
   teacher labels, or critic scoring.  It nevertheless calls the frozen base and
   adapted policy models again, contradicting the explicit no-new-model-call rule.
4. **Skip the mechanistic divergence analysis. Scientifically safe.** The frozen G3
   result and the literature-backed explanation remain reportable without claiming
   an empirical chi-squared comparison.

## Recommended resolution

If PaperGuru wants the registered comparison, explicitly authorize option 3:

- use the exact existing D1-D6 response/token artifacts;
- add a lossless per-token log-ratio output to `compute_kl.py`;
- perform deterministic teacher forcing only (no new sampling or labels);
- hash and freeze the new token-ratio artifacts before any divergence fit;
- retain the original KL/G3 verdict and label the alternative analysis non-gating.

Otherwise skip the analysis.  Do not substitute a proxy after seeing its fit.

## Impact on propositions and gates

- P1-P6 and G1-G4: unchanged and frozen.
- P6/G3 remains FAIL with KL LODO R2 `0.631786`.
- No evidence for or against chi-squared superiority has been generated.
- P7/G5 has not run and remains awaiting pre-registration lock.

## Rollback

No runtime or data change exists to roll back.  Delete only this report if the
blocker record is superseded by a human-approved resolution.

## Reproduction

```bash
head -1 "$PCCD_OUT/g2/D5_kl_items.jsonl"
find "$PCCD_OUT" -type f \( -iname '*ratio*' -o -iname '*kl*item*' -o -iname '*token*log*' \)
rg -n "differences|log_ratio" src/compute_kl.py
```

