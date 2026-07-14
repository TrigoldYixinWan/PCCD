# PCCD — Project Brief (Full Research Cognition Sync for Local Codex)

> Purpose: give the local Codex agent the SAME understanding of this project that
> PaperGuru has — not just *what* to run, but *why* every choice was made, the full
> reference landscape, the methodology, the experimental design and its rationale.
> Codex and PaperGuru differ ONLY in labor division (Codex executes on the GPUs;
> PaperGuru reviews + writes the paper). On the science, we must be identical.
>
> Read this together with `HANDOFF.md` (operational) and `configs/plan_9day.md` (timeline).
> Target venue: **AAAI** (full paper). Citation style: numeric/AAAI (aaai-style .bst).

---

## PART A — Research concern: what problem, and why it matters

### A.1 One-sentence thesis
When a shared, frozen critic (an independent safety/quality judge) is deployed to score
the outputs of a policy that has been **locally adapted** by a downstream party
(system-prompt, LoRA, or DPO), the critic's **calibration silently degrades on that
policy's output distribution** — and this degradation is (i) real and per-policy
measurable, (ii) **false-negative asymmetric** (unsafe outputs increasingly slip through
as "safe"), (iii) **predictable from the adaptation strength measured as KL(adapted‖base)**,
and (iv) **recoverable by lightweight per-policy temperature scaling**.

### A.2 Why this is important (the deployment story)
The realistic setting is *private-model customization*: a provider ships a base policy plus
a frozen critic (kept frozen for cost, auditability, and because the customer cannot
retrain it). Customers adapt the policy locally to their domain. Everyone assumes the
critic still "works" because it was well-calibrated at ship time. Our claim is that this
assumption fails **silently** — there is no error signal at deployment (labels are absent),
and the failure is **safety-relevant** because it is FN-biased: the critic becomes
over-permissive exactly where adaptation pushed the policy off-distribution. This is the
gap that makes the paper matter to AAAI's safety/alignment audience.

### A.3 The FN-asymmetry is the scientific heart (guard it)
A symmetric accuracy drop would be a mundane "distribution shift hurts classifiers" result.
Our contribution is that the drop is **asymmetric toward false negatives** (unsafe→passed),
which is the dangerous direction and is *not* what generic covariate-shift theory predicts.
G2 (Day 5) is make-or-break precisely because it must establish this asymmetry, not just
a drop. If Codex ever sees a symmetric drop, that is a real (possibly negative) result —
report it truthfully; do NOT reshape data/metrics to manufacture asymmetry (see §E, and
HANDOFF §8).

---

## PART B — Pivot history (so Codex understands why the RQ looks the way it does)

The project reached "Gap 1" after several reviewer-driven pivots. Codex should know the
discarded framings to avoid accidentally drifting back into them:

1. **Started broad**: "LLM alignment for private-model customization" — too diffuse; a
   simulated senior reviewer flagged no crisp, falsifiable claim.
2. **Considered** studying the *policy's own* confidence/UQ collapse under fine-tuning —
   rejected because that overlaps heavily with sycophancy/UQ-collapse work (e.g. Sahoo
   2026 "Calibration Collapse Under Sycophancy Fine-Tuning") and with reward-hacking papers.
3. **Considered** per-rater / annotator calibration — rejected: occupied by per-rater
   shrinkage work (e.g. Raj 2026 "PEBS"). Our unit is the **policy**, not the rater.
4. **Landed on Gap 1**: an *independent frozen critic* evaluated on the *policy output
   distribution*, per-policy, FN-asymmetric, KL-predictable, temperature-recoverable.
   This is defensibly novel against the nearest neighbors (§C) and is falsifiable via the
   4 gates.

Takeaway for Codex: the specificity ("independent", "frozen", "policy output distribution",
"per-policy", "FN-asymmetric", "KL-predictable", "temperature-recoverable") is deliberate
armor against reviewers. Do not soften it in code comments, logs, or reports.

---

## PART C — Reference landscape (nearest neighbors & how we differ)

These were retrieved via paper_search (2024–2026, newest-first) and belong in refs.bib.
For each, Codex must be able to state our delta (useful when writing experiment reports
and when a reviewer-style question arises).

### C.1 Closest neighbors — cite AND differentiate explicitly
- **Sahoo 2026, "Calibration Collapse Under Sycophancy Fine-Tuning"** (arXiv:2604.10585):
  fine-tuning breaks UQ. DELTA: theirs is the *policy's own* UQ collapse from reward
  hacking; ours is an *independent frozen critic* losing calibration on the *adapted
  policy's outputs*, with a controlled adaptation grid and a recalibration remedy.
- **Raj 2026, "PEBS: Per-rater Empirical-Bayes Shrinkage"** (arXiv:2606.27578): per-rater
  RM calibration. DELTA: our unit is **per-policy**, and our shift source is *policy
  adaptation*, not annotator heterogeneity.
- **Hiremath 2026, "Calibration Drift Under Reasoning"** (arXiv:2606.11211): drift from
  CoT budget. DELTA: our drift driver is **adaptation strength KL(adapted‖base)** and we
  fit a **scaling law** to it.
- **Leong 2026, "Online Shift Detection and Conformal Adaptation for Deployed Safety
  Classifiers"** (arXiv:2606.11949): safety classifiers drift silently; online conformal
  fix. DELTA: our shift is *induced by known local adaptation* (D0–D6), diagnosed
  per-policy, and fixed by **per-policy temperature scaling** (cheaper, auditable) rather
  than online conformal adaptation.
- **Siahkali 2026, "Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under
  Distribution Shift"** (arXiv:2602.14913): CP coverage under shift. Use for the
  recalibration/uncertainty framing in G4.

### C.2 Reward-model degradation / hacking context (background, not our object)
Wolf 2025 "Reward Model Overoptimisation in Iterated RLHF" (arXiv:2505.18126);
Zhang 2024 "Policy Filtration for RLHF to Mitigate Noise" (arXiv:2409.06957);
Xu 2025 "Learning a Pessimistic Reward Model in RLHF" (arXiv:2505.20556);
Duan 2026 "Mitigating Reward Hacking via Bayesian Non-negative Reward Modeling"
(arXiv:2602.10623); Wang 2026 "Reward Hacking in the Era of Large Models" (survey,
arXiv:2604.13602); Leng 2024 "Taming Overconfidence in LLMs: Reward Calibration in RLHF"
(arXiv:2410.09724); Lu 2024 "It Takes Two: Seamlessness between Reward and Policy Model"
(arXiv:2406.07971). These frame RM fragility broadly; NONE isolates a *frozen independent
critic's per-policy calibration transfer under a controlled local-adaptation grid*.

### C.3 Calibration / conformal methods (for the recalibration machinery, P4/G4)
Jonkers 2024 "Conformal Predictive Systems Under Covariate Shift" (arXiv:2404.15018);
Penso 2025 "Conformal Prediction ... Noisy Labels" (arXiv:2501.12749); Dong 2025 survey on
calibration under class imbalance (relevant: safe/unsafe is imbalanced). Classic tools we
actually use: temperature scaling (Guo et al. 2017 — add via paper_search when writing),
ECE/adaptive-ECE, and netcal/mapie implementations already installed.

> RULE: every entry above must enter refs.bib **verbatim** from a paper_search result, and
> the final refs.bib must pass `ref_verify` with 0 unverified. Do not hand-type fields.

---

## PART D — Central propositions & weights (the paper's spine)

Weights = share of the paper's claimed contribution; they drive how much experimental
evidence and narrative each needs.

- **P1 (10%, support)** — On the *base* output distribution the frozen critic is
  well-calibrated (baseline sanity; establishes the "silent" premise).
- **P2 (part of 35% main)** — Under local adaptation the frozen critic's calibration
  degrades on the policy's output distribution.
- **P3 (part of 35% main)** — This degradation is *measurable per-policy* (heterogeneous
  across the 10-policy stack), not a single global number.
- **P4 (10%, support)** — Per-policy temperature scaling recovers calibration (the remedy).
- **P5 (25%, main)** — The degradation is **false-negative asymmetric**.
- **P6 (20%, main)** — The degradation is **predictable from KL(adapted‖base)** — a scaling
  law with a fitted, reported functional form and goodness-of-fit.

Mapping to gates: G1↔P3 (heterogeneity), G2↔P2+P5 (degradation + asymmetry, make-or-break),
G3↔P6 (scaling law), G4↔P4 (recalibration). P1 is established during Day-2/Day-3 baseline.

---

## PART E — Methodology and the reasoning behind every design choice

### E.1 The three actors (and why they are separate)
- **Teacher** = Qwen2.5-32B-Instruct, **label-only**. It produces the *ground-truth-ish*
  labels (safe/unsafe, preference) used to (a) train/define the critic's target and (b)
  evaluate calibration. WHY label-only: if the teacher also wrote the "chosen" response,
  the critic could learn a spurious "longer/teacher-style = better" shortcut — the
  **More-is-Less** artifact. By forbidding the teacher from generating responses, the
  critic must judge the *policy's own* outputs on their merits. This is a correctness
  invariant, not a preference — do not add a teacher-generation path (HANDOFF §8).
- **Critic** = a frozen judge (7B-scale) trained/calibrated ONCE on the base
  distribution, then never updated. WHY frozen & independent: the whole phenomenon is
  "an independent critic that the customer cannot retrain." If the critic adapts with the
  policy, there is no phenomenon to study.
- **Policy** = 7B-scale model, adapted along the D0–D6 grid. Its outputs are what the
  critic scores.

### E.2 The 10-policy stack (why 10, why this mix)
Defined in `configs/policy_taxonomy.json`. H1–H5 hard policies (severity high/medium),
S1–S3 soft policies, T1–T2 task policies. WHY a stack rather than one policy: P3/G1 require
*heterogeneity* — we need the degradation to vary across policies so that "per-policy"
calibration is meaningful and so the scaling law (P6) has spread on the x-axis (different
policies reach different KL). Severity weights (high=4/med=2/low=1) let us weight the
FN-asymmetry by how dangerous the miss is. The soft (S) and task (T) policies provide
contrast: they should show weaker/less-dangerous degradation, sharpening the claim that
the *hard-safety* policies are where FN-asymmetry bites.

### E.3 The adaptation grid D0–D6 (why these knobs)
D0 base, D1 system-prompt, D2/D3/D4/D5 LoRA r=4/8/16/32, D6 DPO β=0.1. WHY a graded grid:
P6 needs a *continuous-ish* axis of adaptation strength. Each step increases the expected
KL(adapted‖base): prompt-only (small) → low-rank LoRA (moderate, increasing with rank) →
DPO (preference-shifted). WHY measure KL rather than "rank" as the x-axis: rank/β are not
comparable across method types, but KL(adapted‖base) on held-out prompts is a *unified,
method-agnostic* scalar — that is what makes P6 a real scaling law rather than a
per-method curve. Codex must compute KL consistently (same prompt set, same estimator)
across ALL of D0–D6 so the x-axis is coherent.

### E.4 Data (why PKU-SafeRLHF + UltraFeedback + self-built soft pairs)
- **PKU-SafeRLHF** → hard-safety supervision: prompt, response_0/1, is_response_0/1_safe,
  19 harm categories, severity_level. This anchors H1–H5 and the safe/unsafe label used
  for FN-asymmetry.
- **UltraFeedback** → quality/helpfulness signal: instruction, completions[] with
  helpfulness/honesty/instruction_following/truthfulness Ratings. Anchors S1–S3 and T1–T2.
- **Self-built soft-preference pairs** (`src/sample_data.py`): the APM dataset is not
  public, so we synthesize soft-preference pairs for the soft policies. WHY: we need
  preference structure the teacher can label without generating responses.
- Splits: train / calib / test / conflict / audit. `calib` is the held-out set for
  temperature scaling (must be disjoint from `test` used to report calibration — a leak
  here invalidates P4/G4; guard it).

### E.5 Recalibration mechanism (P4/G4)
Per-policy **temperature scaling**: fit a single scalar T_p per policy p on the `calib`
split, apply to the critic's logits, re-measure calibration on `test`. WHY temperature
scaling (vs online conformal like Leong 2026): it is the *minimal* intervention — one
scalar per policy, no critic retraining, auditable, and it directly tests whether the
degradation is a *calibration* problem (fixable by rescaling confidence) vs a
*discrimination* problem (would need more). If temperature scaling recovers calibration
but NOT discrimination (AUROC), that is itself an informative, reportable finding.

### E.6 Hardware/runtime rationale (already validated Day-1)
2× RTX PRO 6000 Blackwell 96GB. One 32B teacher fits per card → we run the teacher as two
independent shards (GPU0 = even indices, GPU1 = odd), no cross-GPU comm, no memory pooling
(that is why SYS-topology P2P at 25.6 GB/s is a non-issue). Disk MUST be on
/root/autodl-tmp (see HANDOFF §3). Installed stack differs from pinned (trl 0.19.1 etc.) —
adapt code to the installed APIs; do not downgrade blindly.

---

## PART F — Metrics, gate criteria, and the experiment-report format

### F.1 Metrics (report these, with uncertainty)
- **Calibration**: ECE and adaptive-ECE (use netcal), reliability diagrams; report Brier
  score too. Always with bootstrap CIs over the test split.
- **FN-asymmetry (the key metric for P5/G2)**: track FN rate (unsafe scored safe) and FP
  rate separately per policy and per adaptation step; report the **FN/FP ratio** and its
  trend vs D0. Weight FN by severity (high=4/med=2/low=1). An asymmetry claim needs FN
  growing significantly faster than FP — report a significance test (bootstrap or
  McNemar-style) and effect size, not just point estimates.
- **Discrimination**: AUROC / AUPRC of the critic per policy per step (to separate
  calibration loss from discrimination loss).
- **Adaptation strength**: KL(adapted‖base) per policy per step, same estimator throughout.
- **Scaling law (P6/G3)**: fit degradation-metric = f(KL); report the functional form,
  fitted params with CIs, and R²/goodness-of-fit; hold out some policies to test predictive
  (not just descriptive) power if time allows.

### F.2 Gate pass/fail criteria (be honest — HANDOFF §8)
- **G1 (P3)**: the 10 policies are statistically distinguishable in output distribution
  AND their D0 critic behavior differs — heterogeneity confirmed.
- **G2 (P2+P5, make-or-break)**: (a) calibration degrades from D0→higher-D (ECE up,
  significant), AND (b) the degradation is FN-asymmetric (FN/FP ratio rises significantly
  with adaptation). BOTH required to pass. A drop without asymmetry = partial/negative.
- **G3 (P6)**: degradation is well-predicted by KL (report R²; a pre-registered threshold,
  e.g. R² ≥ 0.6 on held-out policies, is the bar — adjust only with human sign-off).
- **G4 (P4)**: per-policy temperature scaling significantly reduces ECE on test toward the
  D0 level, without harming AUROC.

### F.3 Experiment-report format (so PaperGuru can ingest results directly)
For EACH gate/day, Codex commits a short markdown report under `reports/dayN_<gate>.md`
containing: (1) exact commands run + git commit hash of the code used; (2) the numbers
(tables with mean±CI, per-policy where relevant); (3) the pass/fail verdict against F.2
with the actual statistic; (4) any anomalies; (5) a pointer to the raw artifacts on the
data disk (paths under $PCCD_OUT). Keep raw arrays/plots on the data disk; commit only the
summary md + small plot PNGs. This is what PaperGuru turns into the paper's tables/figures,
so numbers in the report MUST match the raw artifacts exactly (no rounding-away of CIs).

---

## PART G — Autonomy & the mandatory change report (NEW authority granted to Codex)

Codex MAY now change project design to handle unforeseen errors/blockers, WITHOUT waiting,
within these bounds — but every such change REQUIRES a change report so PaperGuru can
review with full context.

### G.1 Green — change freely, log briefly
Implementation-level fixes that do not touch the science: adapting to installed library
APIs (trl/peft/vllm version differences), disk/path/OOM/batch-size/dtype fixes, download
robustness, sharding/parallelism, logging, seeds plumbing, script refactors. Note them in
the day report; no separate approval needed.

### G.2 Yellow — change if blocked, but WRITE A CHANGE REPORT and flag in the PR
Design-adjacent changes made to get unblocked: e.g. substituting a dataset field/mapping,
adjusting a split ratio, changing the KL estimator, altering how a policy is adapted if a
method won't run, reducing the D-grid or policy count for compute, changing the
recalibration variant. Allowed to proceed so the pipeline doesn't stall, but you MUST
create `reports/CHANGES/<date>_<slug>.md` (template in G.4) and call it out in the PR.

### G.3 Red — do NOT change without human/PaperGuru sign-off
The scientific identity of the project (HANDOFF §8): the 5 distinctions vs Shihab 2026 /
the neighbors; teacher stays label-only; the meaning of FN-asymmetry; proposition weights;
whether a gate "passed." If blocked here, implement the *minimal* workaround, mark results
as PROVISIONAL, write the change report, and STOP for review.

### G.4 Change-report template (`reports/CHANGES/<YYYY-MM-DD>_<slug>.md`)
```
# Change: <short title>    (severity: Green|Yellow|Red)
Date / commit: <iso date> / <git sha>
Trigger: what error or blocker forced this (paste the exact error).
What I changed: files + the design decision, before → after.
Why this and not an alternative: options considered, why this is least-damaging to the science.
Impact on propositions/gates: which of P1–P6 / G1–G4 are affected and how.
Reversibility: how to revert; is any result now PROVISIONAL?
Open question for PaperGuru: <if any>
```
Rule of thumb: if PaperGuru would need this to interpret a number in the final paper, it
belongs in a change report. When unsure whether something is Yellow or Red, treat it as Red.

