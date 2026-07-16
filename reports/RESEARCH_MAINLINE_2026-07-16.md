# PCCD research mainline after delegated takeover

## Paper center

PCCD now studies whether calibration measured on a frozen safety critic's source
distribution survives adaptation of the separately instantiated policy model.
The central claim is deliberately scoped to the tested model family, adaptation
objectives, ten-criterion taxonomy, and fixed reference-annotation protocol.

The desired contribution is not "adaptation always makes safety critics miss
violations." That proposition failed decisively: the registered FN-asymmetry
effect reversed. The defensible and potentially publishable contribution is:

1. the critic can be well calibrated before deployment-like policy adaptation;
2. adaptation can produce a measurable calibration-transfer failure;
3. the size and direction of that failure can differ by criterion even when the
   same frozen critic and prompt distribution are used; and
4. low-shot target-aware structured recalibration may provide a bounded remedy.

## Frozen evidence

- P1 anchor: D0 base mean ECE `0.028904`; this is descriptive support for good
  source calibration, not a preregistered pass threshold.
- Original L3/P3: formal failure, violated-F1 CV `0.08057`, 95% CI
  `[0.06468, 0.11145]`, below `0.15`. This says the base critic is relatively
  uniformly accurate; it does not refute heterogeneous post-adaptation drift.
- Discovery P2/P3 at D5: mean delta-ECE `0.028831`
  `[0.022345, 0.033063]`; cross-criterion SD `0.037940`
  `[0.032001, 0.045160]`; RMS `0.047652`
  `[0.040522, 0.054596]`. These are strong within-study discovery effects but
  are not yet independent confirmation.
- P5: conclusive opposite-direction result, mean
  `delta-FN - delta-FP = -0.277811`
  `[-0.322418, -0.232448]`. It remains a prominent boundary finding.
- P6/G3: the registered KL prediction test failed (`LODO R^2=0.631786 < 0.70`),
  despite an aggregate positive association. Chi-square, reverse-KL, and TV did
  not establish a superior predictor under the registered reanalysis.
- P4/G4: source-only temperatures did not transfer. P7 established meaningful
  low-shot improvement at budget 500, but not the registered operational
  tolerance: per-criterion target temperature mean ECE `0.043698`, two-stage
  95% CI `[0.044935, 0.056267]`, with reduction versus raw
  `0.035797 [0.025269, 0.037051]` and preserved discrimination.

## New evidence ladder

### Tier 1 — mandatory confirmation

One new lexical-family-disjoint, outcome-blind 3,000-prompt lockbox and one new
D5 training seed test P2-C and P3-C. This is the minimum evidence needed to use
"adaptation-induced, criterion-specific calibration drift" in the title or
abstract without an explicit exploratory qualifier.

### Tier 2 — remedy

Published Structured Matrix Scaling is fit using only 500 new target-calibration
prompts and evaluated on the untouched 2,000-prompt confirmation test. Its
paired comparison against per-criterion temperature scaling is P8-C. A success
creates a diagnosis-to-remedy loop; a failure narrows the actionability claim
without invalidating Tier 1.

### Tier 3 — construct validity

A blinded human audit estimates reference-annotation error and tests whether
that error itself changes by adaptation domain or criterion. Until it is
complete, claims must say "relative to the fixed reference annotation
protocol," not "ground-truth safety violation."

## Manuscript outcomes

- Strong positive: P2-C and P3-C confirm, human audit supports construct
  validity, and P8-C succeeds. Position as an empirical failure mode plus a
  low-shot structured remedy.
- Publishable empirical result: P2-C/P3-C confirm and human audit supports them,
  but P8-C is partial or fails. Position as the first criterion-resolved
  calibration-transfer study with carefully delimited negative repair results.
- Narrow result: only P2-C confirms. Position around calibration certificates
  failing under adaptation; heterogeneity becomes discovery-only.
- Non-confirmation: P2-C fails on the lockbox. Report the original effects as
  evaluation-set-specific and pivot to a measurement/protocol paper or workshop;
  do not claim general calibration degradation.

All outcomes retain the complete P5/P6/P7 negative evidence. The new design
improves the probability that a genuinely reproducible positive effect is found;
it does not guarantee one.
