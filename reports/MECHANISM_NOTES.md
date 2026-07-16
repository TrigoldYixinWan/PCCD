# PCCD — Mechanism Notes & Literature-Informed Analysis Additions (2026-07-16)

After G3/G4 both FAILed, a targeted literature review surfaced concrete mechanisms that
explain WHY they failed and two cheap, high-value analyses to add. These sharpen the paper
from "we observed negatives" to "we explain the negatives with known mechanisms and test the
implied better alternatives." None of this changes any frozen gate verdict.

## Mechanism 1 — why a single scalar KL under-predicts degradation (P6/G3)
- Kim et al. 2025 (\cite{kim2025rethinking}) show reward-model benchmark accuracy correlates
  only weakly with downstream over-optimization behavior — a single aggregate score does not
  capture how an RM behaves on a shifted/optimized policy distribution. Our result is the
  calibration analogue: a single KL magnitude under-predicts per-policy calibration drift.
- Huang et al. 2024 (\cite{huang2024correcting}) argue KL-regularization has a FUNDAMENTAL
  limitation for characterizing over-optimization because KL under-penalizes heavy-tailed /
  out-of-distribution mass; they show a chi-squared (chi^2) divergence captures it better.
  This gives a PRINCIPLED reason our KL predictor missed LODO 0.70 and a concrete better
  candidate predictor to test (below).
- Qiu et al. 2024 (\cite{qiu2024reward}) and Yang et al. 2024 (\cite{yang2024regularizing})
  show reward generalization depends on the policy's representational/topological relation to
  the RM, not a scalar distance — consistent with our finding that adaptation OBJECTIVE and
  POLICY IDENTITY are necessary variables (per-policy slopes ranged +0.13 to -0.04).

Paper framing: "KL magnitude is insufficient ACROSS adaptation objectives" is now backed by
a known f-divergence limitation, not left as an unexplained miss.

## Mechanism 2 — why source-only temperature does not transfer (P4/G4)
- Cheng et al. 2025 (\cite{cheng2025signal}) show calibration across heterogeneous
  distributions needs a SHIFT-AWARE signal; a recalibrator fit on one distribution does not
  transfer to a shifted one. This directly explains G4: source-calib temperatures preserve
  discrimination and improve source NLL, but do not transfer to the adapted (or even
  support-enriched D0) distribution. It also motivates P7's TARGET-AWARE direction.

## Added analysis A (cheap, high value) — chi^2 / alternative-divergence predictor for G3
BEFORE running P7, add a NON-GATING re-analysis of the existing G2 data: recompute the
adaptation-strength predictor as chi^2(adapted||base) (and, as further sensitivities, a
reverse-KL and a total-variation estimate) on the SAME frozen 3,000-prompt outputs, using the
SAME per-item log-ratio records already saved (chi^2 is E_base[(p_adapt/p_base - 1)^2], and a
token-level plug-in from the stored per-token log-ratios is feasible). Re-fit the LOCKED G3
primary form with the alternative predictor under the SAME LODO/permutation protocol.

Interpretation rule (locked to avoid p-hacking): the KL-based G3 verdict remains FROZEN and
primary. The chi^2 analysis is REPORTED as an explanatory/mechanistic result: if chi^2
predicts materially better than KL (higher LODO R^2), the paper's contribution UPGRADES from
"KL is insufficient" to "we identify KL's specific weakness (heavy-tail under-penalization)
and show a chi^2 / shift-sensitive divergence is a better — though still imperfect —
predictor," citing \cite{huang2024correcting}. If chi^2 does NOT help, we report that too
(the shift is not merely a tail phenomenon), which still strengthens the mechanistic story.
This does NOT change the frozen KL/G3 verdict and is explicitly labeled exploratory-but-
pre-specified (specified here, before it is computed).

## Added framing B — P7 low-shot is the shift-aware remedy the literature predicts
P7 (PREREG_G5.md) is now explicitly positioned as the SHIFT-AWARE recalibration that Cheng
et al. 2025 imply is necessary: a small number of TARGET-distribution labels supplies the
shift signal that source-only scaling lacks. hierarchical shrinkage across per-policy target
temperatures is motivated by imbalanced/low-shot practice (\cite{gao2026comprehensive} for
imbalance; standard empirical-Bayes shrinkage) to handle sparse per-policy support.

## Boundaries (Red)
- The chi^2 re-analysis is pre-specified HERE, before computation; it is non-gating and does
  not alter the frozen KL-based G3 verdict.
- No frozen gate (P2..P6, L1..L3) verdict changes. New analyses only add explanation or test
  new narrower propositions (P7).
- Frozen D0 critic read-only; teacher label-only fixed-prompt; no new teacher/critic calls
  for the chi^2 re-analysis (it reuses stored per-item log-ratio records).
