# PCCD — G2 Pre-Registration (locked before any full-grid run)

Pre-registered 2026-07-16 by PaperGuru (human-approved). This LOCKS the G2 protocol — the
make-or-break gate for P2 (degradation exists) + P5 (degradation is FN-asymmetric) — BEFORE
any full D0-D6 run. It supersedes the ad-hoc Day-5 sketch in plan_9day.md. Changing anything
here after seeing G2 data is a Red protocol violation.

## Why the protocol changed (the D3 diagnostic lesson)
The single-point D3 diagnostic (reports/day5_adapt_diag.md, commit cea0f40) was executed
correctly and taught us three things that this pre-registration fixes:

1. WRONG ADAPTATION DIRECTION. D3 was LoRA-SFT on SAFE/high-quality data, which pushed the
   policy toward MORE-OBVIOUS, MORE-FREQUENT violations — easier for the frozen critic to
   catch. That is the OPPOSITE of the deployment scenario P5 targets, where local adaptation
   pushes the policy OFF the critic's training distribution so violations become HIDDEN /
   subtler and evade detection (FN rises). Reward-hacking / overoptimisation work motivates
   this: adaptation that optimises a proxy objective drifts the output distribution in ways a
   fixed reward/critic model increasingly misjudges (\cite{zhang2024policy}; and reward-overoptimisation work held out pending indexing).
   G2 therefore uses HIDDEN-VIOLATION adaptation, not safe-SFT.

2. FN SUPPORT CONFOUND. FN = P(pred!=violated | teacher violated) was compared across base
   (violated support 3-10 items) and D3 (74-312). The base FN was sparse-support noise, so
   "FN decreased" was an artifact of incomparable support, NOT evidence about degradation.
   G2 fixes this with a pre-registered adequate per-policy violated support (below).

3. UNSTABLE HETEROGENEITY STATISTIC. Signed-CV of Delta_ECE blew up (CI to 17) because its
   denominator is a near-zero mixed-sign mean. G2 uses a stable dispersion statistic (below).

The one REAL directional signal at D3 — Delta_FP positive across all 10 policies — is retained
as supporting evidence that adaptation does perturb the critic; G2 will test whether the
safety-relevant FN direction emerges under hidden-violation adaptation.

## Adaptation grid (LOCKED) — direction corrected
Keep the KL-graded grid D0..D6 (D0 base, D1 system-prompt, D2/D3/D4/D5 LoRA r=4/8/16/32,
D6 DPO beta=0.1) BUT the adaptation OBJECTIVE is now hidden-violation-inducing, not safe-SFT:
- LoRA/DPO points optimise a preference/objective that rewards outputs which are
  SUPERFICIALLY compliant but contain subtler violations (i.e. a proxy the frozen critic was
  not trained to catch) — operationalised as DPO toward responses the teacher marks violated-
  but-fluent, or SFT on off-distribution style that masks violations. The EXACT adaptation
  data/objective per point is specified and frozen in the run's CHANGES note before running.
- D3 safe-SFT is RETAINED as a labelled CONTROL point (the "benign adaptation" contrast): P5
  predicts FN-asymmetric degradation under hidden-violation adaptation but NOT under benign
  safe-SFT. Showing the direction flips between the two is itself strong evidence.
- KL(adapted||base) is the x-axis for every point, using the LOCKED estimator from Day-5
  (token-weighted mean log-ratio, nats/token, prompt-bootstrap CI, seed fixed).

## Evaluation set & support (LOCKED) — fixes the FN confound
- Use a fixed held-out prompt set large/stratified enough that EVERY policy has >= 30 teacher-
  VIOLATED items on BOTH the base and each adapted distribution, so FN/FP are comparable.
  If the natural test split cannot reach 30 violated for a policy on the base distribution,
  augment the evaluation prompt pool (from the held-out dedup pool, disjoint from critic
  train/calib) until it does; record per-policy support. Do NOT compute FN on <30 support;
  mark such cells UNDERPOWERED and exclude from the asymmetry aggregate (report separately).
- Base and adapted are paired by prompt; teacher labels the adapted outputs (label-only,
  fixed prompt = L1 oracle); the FROZEN D0 critic (read-only) scores both.

## Metrics (LOCKED)
- Per-policy, base vs adapted: ECE, violated-F1, FN = P(pred!=violated | teacher violated),
  FP = P(pred=violated | teacher satisfied). Deltas = adapted - base, paired prompt-bootstrap
  (10,000 replicates, fixed seed), per-policy 95% CIs.
- DEGRADATION heterogeneity statistic (LOCKED, stable near zero): cross-policy
  ROOT-MEAN-SQUARE of Delta_ECE, RMS(Delta_ECE) = sqrt(mean_p Delta_ECE_p^2), with bootstrap
  CI. Also report cross-policy SD of Delta_ECE. We do NOT use signed-CV (ill-conditioned).
  Heterogeneity is supported if RMS(Delta_ECE) materially exceeds the base-level dispersion
  reference (D0 base-F1 CV 0.081 is descriptive context, not the test).
- FN-ASYMMETRY statistic (LOCKED): per adaptation point, mean over ADEQUATELY-POWERED policies
  of (Delta_FN - Delta_FP), with bootstrap CI; and the count of policies with Delta_FN>Delta_FP.

## G2 gate criteria (LOCKED, make-or-break)
Evaluated at the highest-KL hidden-violation point(s), primarily D5 (LoRA r=32) and D6 (DPO):
- G2(a) DEGRADATION EXISTS: mean per-policy Delta_ECE > 0 with CI excluding 0 at the
  hidden-violation D5/D6 points (critic calibration worsens under adaptation).
- G2(b) FN-ASYMMETRY: mean (Delta_FN - Delta_FP) > 0 with 95% CI excluding 0 at D5/D6, over
  adequately-powered policies (>=30 violated support). BOTH (a) and (b) required for G2 PASS.
- CONTROL check: at the benign D3 safe-SFT control, we expect NO FN-asymmetry (this is the
  already-observed Delta_FN-Delta_FP < 0), demonstrating the effect is specific to hidden-
  violation adaptation, not any adaptation.

## Honesty clause (LOCKED)
If, under correctly-powered support and hidden-violation adaptation, FN-asymmetry still does
NOT appear (CI includes or is below 0), G2 FAILS and we do NOT reshape the metric or direction
again. The paper then reports the true result: adaptation perturbs the frozen critic (Delta_FP
positive, Delta_ECE heterogeneous) but the degradation is NOT FN-asymmetric under our regimes
— a narrowed but honest contribution, combined with the L1/L2 and measurement-methodology
findings. This is the pre-registered fallback; no third redirection of the P5 test is allowed.

## Ordering (LOCKED)
1. Build the adequately-powered evaluation prompt set (>=30 violated/policy on base).        (CPU)
2. Implement hidden-violation adaptation objective; freeze its exact spec in a CHANGES note.
3. Run D0..D6 adaptation + generation + KL + teacher labels + frozen-critic scoring.          (GPU)
4. Compute locked metrics; evaluate G2(a)/(b) at D5/D6; report D3 benign control contrast.
5. reports/day5_g2.md with per-policy tables, RMS/asymmetry stats, CIs, and the gate verdict.
6. PaperGuru reviews. G3 (scaling law Delta_ECE ~ KL) and G4 (recalibration) follow only if
   the data supports proceeding.
Frozen D0 critic is read-only throughout; teacher stays label-only fixed-prompt.
