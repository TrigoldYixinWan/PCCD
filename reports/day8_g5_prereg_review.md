# Day 8 — G5/P7 pre-lock implementation review

`reports/PREREG_G5.md` is scientifically implementable after lock, but three
details must be fixed before execution so the code cannot make post-hoc choices.
No target split, temperature, or metric has been computed during this review.

## 1. Lock the target split size and deterministic algorithm

Section 2 fixes seed `20260722` but not the TARGET-CALIB/TARGET-TEST sizes or the
exact ID partition algorithm.  Recommended lock:

1. sort the 3,000 D5 prompt IDs lexicographically;
2. apply `numpy.random.default_rng(20260722).permutation` to row indices;
3. first 1,000 IDs = TARGET-CALIB, remaining 2,000 = TARGET-TEST;
4. nested budgets are the first 50/100/200/500 IDs in that frozen calibration
   order;
5. write and hash both ID manifests before fitting any temperature.

The paper should call these **newly frozen disjoint partitions of the previously
evaluated G2 data**, not previously unseen observations.  They are untouched for
temperature fitting, but G2/G4 already reported aggregates from the 3,000 rows.

## 2. Lock the hierarchical-shrinkage formula

Section 4 correctly requires the formula to be fixed before running, but the draft
does not yet define it.  A support-count weight such as `n/(n+k)` is uninformative
here because every policy has exactly `b` three-way training cells at a budget
(N/A is a real class).

Recommended parameter-free empirical-Bayes implementation on log temperature:

```text
tau_g       = fitted global target log-temperature
tau_p       = fitted per-policy target log-temperature
v_p         = inverse observed total-NLL curvature at tau_p
s2          = max(sample_variance_p(tau_p) - mean_p(v_p), 0)
w_p         = s2 / (s2 + v_p)
tau_shrink,p = w_p * tau_p + (1-w_p) * tau_g
T_shrink,p   = exp(clip(tau_shrink,p, log(0.05), log(20)))
```

Lock the finite-curvature floor (recommended `1e-8`) and use summed rather than
mean NLL curvature so `v_p` scales with information/budget.  Recompute the entire
formula within each calibration bootstrap replicate.

## 3. Clarify the underpowered fallback and comparison direction

The current `<10 fitting examples` fallback never activates: every policy receives
50-500 prompt cells and N/A participates in three-way NLL.  PaperGuru should either
remove it as redundant or define the intended effective-support quantity (for
example, applicable satisfied+violated cells) before lock.  It must not be chosen
after inspecting temperatures.

Also lock the P7 structure-benefit contrast as

```text
mean_p ECE(target-global-T) - mean_p ECE(target-per-policy-or-hierarchical-T)
```

with a paired 95% bootstrap CI whose lower bound must exceed zero.

## Status

- No fundamental compute blocker after these clarifications.
- All inputs already exist; G5 is CPU-only and does not require new teacher, critic,
  or policy calls.
- Implementation and execution remain prohibited until PaperGuru marks the revised
  pre-registration human-approved and locked.
