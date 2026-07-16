# PCCD — G5 / P7 Low-Shot Target-Aware Recalibration Pre-Registration (DRAFT for human lock)

Drafted 2026-07-16 after G4 FAIL and the human-approved second reframe (THESIS_REFRAME.md).
**Not locked until PaperGuru approves.** No temperature/curve fitting or target-split scoring
may run before approval. After approval, changing the splits, budgets, methods compared,
primary point/metric, or verdict rule is Red. This tests a NEW, narrower proposition and CANNOT
change the frozen G4 verdict.

## 1. Proposition P7 (new)
G4 established that ZERO target labels (a temperature map fit only on the base calib split)
does NOT recover calibration on adapted distributions. P7 asks the necessary-condition
question: **how many TARGET-distribution labels does per-policy recalibration need to recover
calibration, and does per-policy structure help over a global temperature?** A positive answer
restores a "problem (P2/P3) -> boundary (P4/P6 fail) -> actionable remedy (P7)" arc.

## 2. Data discipline (LOCKED) — new untouched target splits
The failure mode we must avoid is fitting a recalibrator on the same labels we evaluate on.
- For the PRIMARY adapted point D5 (and, budget permitting, D2/D4), construct NEW target
  calibration and target test splits from the frozen D5 evaluation outputs, by a fixed
  deterministic prompt-ID partition of the existing 3,000 G2 prompts into
  TARGET-CALIB (for fitting temperatures) and TARGET-TEST (for evaluating recovery),
  DISJOINT, seed 20260722. Record exact IDs and hashes.
- Teacher labels on these adapted outputs already exist (label-only, fixed prompt); NO new
  teacher call. Frozen D0 critic logits already exist; NO critic update.
- The base calib split (Day-2 calib) remains the source-T reference only.
- TARGET-TEST labels are NEVER used to fit any temperature; used only to report ECE/F1/AUROC.

## 3. Budgets and learning curve (LOCKED)
Label budgets b in {50, 100, 200, 500}, drawn deterministically from TARGET-CALIB (nested:
the 50-set is a subset of the 100-set, etc.; seed 20260722). At each budget, per policy the
available violated/satisfied/N/A support may be small — report per-policy support at each
budget; a policy-budget cell with < 10 fitting examples for a method is UNDERPOWERED and its
per-policy temperature falls back to the global target-T (documented, not dropped).

## 4. Methods compared (LOCKED)
All applied to the FROZEN D0 critic logits (read-only); none updates the critic:
1. **source-T** (G4 baseline): the 10 base-calib temperatures, zero target labels.
2. **target-global-T**: ONE temperature fit on the b target-calib labels (all policies pooled).
3. **target-per-policy-T**: 10 temperatures fit on the b target-calib labels, per policy.
4. **hierarchical-shrinkage**: per-policy target-T shrunk toward the global target-T with a
   shrinkage weight tied to per-policy support (empirical-Bayes-style; exact shrinkage formula
   fixed in code before running and logged). Motivated by ENCORE-style per-rule structure and
   standard shrinkage; addresses sparse per-policy support at low budgets.
All temperatures use the SAME parameterization/optimizer/bounds as G4 (T=exp(tau),
tau in [log0.05, log20], bounded scalar NLL minimization, xatol 1e-8).

## 5. Metrics (LOCKED)
On TARGET-TEST, per policy and mean over policies, for each method x budget:
- primary: 15-bin top-class 3-way ECE (incl. N/A) — same locked ECE as everywhere.
- recovery vs source-T: ECE(source-T) - ECE(method) (positive = better than the G4 baseline).
- absolute ECE and its reduction vs raw (un-scaled) logits.
- discrimination: violated-F1 (applicable only) and AUROC; must remain non-inferior
  (argmax-preserving per policy; report ties).
Two-stage bootstrap, 10,000 replicates, seed 20260722: resample target-calib rows (refit
temperatures) and independently resample target-test prompts; percentile 95% CIs. Policies
fixed, not bootstrapped. Holm-correct per-policy tests.

## 6. G5/P7 verdict (LOCKED)
Primary point D5. Define the SMALLEST budget b* at which **target-per-policy-T** achieves:
1. mean-over-policies ECE reduction vs raw with 95% CI lower bound > 0, AND
2. mean ECE at that budget with 95% CI upper bound <= the P1 base anchor region (<= 0.05,
   the well-calibrated regime), AND
3. discrimination non-inferiority (mean dF1 CI lower >= -0.005; mean dAUROC CI lower >= -0.01).
- **P7 SUPPORTED (actionable remedy):** such a b* exists at b <= 200, AND at b* target-per-
  policy-T (or hierarchical-shrinkage) beats target-global-T on mean ECE with CI excluding 0
  (per-policy structure helps).
- **P7 PARTIAL:** recovery achieved only at b = 500, or per-policy does not beat global.
- **P7 NEGATIVE:** no budget up to 500 recovers calibration — strengthens the cautionary
  conclusion (even modest target labels are insufficient); reported honestly, no rescue.

## 7. Boundaries (Red)
- Does NOT change the frozen G4/P4 verdict; P7 is a separate narrower claim.
- TARGET-TEST labels never fit a temperature; target-calib/target-test disjoint, IDs frozen.
- Frozen D0 critic read-only; teacher label-only fixed-prompt (no new teacher calls).
- Methods/budgets/verdict locked before running; no post-hoc budget or method addition to pass.

## 8. Outputs after approval
- `$PCCD_OUT/results/g5_lowshot.json` (per method x budget x policy metrics, CIs, hashes);
- `$PCCD_OUT/results/g5_learning_curve.png` + committed summary figure;
- `reports/day8_g5.md` (BRIEF §F.3), including split IDs/hashes and exact commands.
