# PCCD — D0 Critic Architecture & Training Protocol (Pre-Registered)

Pre-registered 2026-07-15 by PaperGuru (human-approved direction: "decide after reading
ENCORE's multi-head implementation"). This LOCKS the frozen critic's architecture and its
D0 (base) training protocol BEFORE any critic training runs. Changing anything here after
seeing D0 results is a Red protocol violation requiring a CHANGES entry.

Evidence base: multi-attribute safety judging in the current literature uses a SHARED
backbone with PER-ATTRIBUTE heads — directly exemplified by ENCORE (Li et al. 2025,
arXiv:2503.20995, "Entropy-guided Reward Composition for Multi-head Safety Reward Models"),
which uses per-rule heads over a shared backbone and shows that high-rating-entropy rules
are less discriminative. We adopt ENCORE's shared-backbone / per-attribute-head STRUCTURE
but replace its Bradley-Terry scoring heads with 3-way classification heads, because our
task is per-policy compliance labeling + calibration, not preference ranking.

## Architecture (LOCKED)
- Backbone: **Qwen2.5-7B-Instruct**, shared across all policies. This is a DIFFERENT model
  and scale from the label-only 32B generative teacher, so the critic is genuinely
  INDEPENDENT of the teacher (no shared weights, no shared scale, different output form).
- Heads: **10 independent lightweight classification heads**, one per policy
  (H1..H5, S1..S3, T1, T2). Each head is a small MLP (Linear -> GELU -> Linear) on the
  backbone's final-token hidden state, producing **3-way logits** over
  {satisfied, violated, not_applicable}. Per-head, not one flat 30-way head, because:
  (a) per-policy temperature scaling (P4) needs per-policy logits — one temperature scalar
      T_p per head, applied to that head's 3 logits;
  (b) per-policy F1-CV (L3) needs per-policy predictions — trivially read from each head;
  (c) it matches the ENCORE multi-head safety-RM structure (evidence-backed) and avoids the
      parameter blow-up / sparse-policy overf_it of 10 fully-separate backbones.
- Pooling: last non-pad token hidden state (standard for causal-LM classification heads).
- The critic reads the SAME (prompt, response) pair the teacher labeled; it never sees the
  teacher's labels at inference. Input format mirrors the teacher's canonical prompt WITHOUT
  the teacher's answer, i.e. the critic judges the policy's response on its own.

## D0 training protocol (LOCKED)
- D0 = the BASE critic: trained ONCE on the base-distribution teacher labels (the Day-2
  train split labels), then FROZEN. It is never updated for D1..D6 — that frozen-ness is the
  whole phenomenon.
- Trainable params: **LoRA on the backbone (r=16, alpha=32, dropout=0.05, target =
  q,k,v,o + gate/up/down proj)** + the 10 classification heads (heads fully trained). LoRA
  rank 16 chosen as a mid-grid value consistent with the adaptation grid; heads are tiny so
  trained in full. Backbone base weights stay frozen (only LoRA deltas + heads learn).
- Loss: sum over the 10 heads of **3-way cross-entropy** on the teacher label for that
  policy, INCLUDING not_applicable as a real third class (the critic must learn when a policy
  does not apply — N/A is informative, not masked). Report per-head loss separately.
  Optionally (ablation, not required for D0) weight heads by ENCORE-style inverse rating
  entropy; NOT part of the locked D0 loss — plain equal-weighted sum is D0.
- Optimizer/schedule: AdamW, lr 1e-4 (LoRA) / 1e-3 (heads), cosine decay, warmup 3%,
  effective batch 32, bf16, max_len 4096, 1-3 epochs with early stop on calib-split macro-F1.
  Seeds fixed and recorded. These are standard PEFT values; any change is a Green note.
- Multi-GPU: EXPLICIT strategy only (torchrun/accelerate, one process per GPU) — NEVER
  implicit HF DataParallel (carried-forward constraint from the Day-4 smoke).
- Class imbalance: heads see the natural label distribution (H1/H5 are N/A-heavy). Do NOT
  resample to balance — that would distort the deployment distribution the critic must
  calibrate on. Report per-head class support.

## What D0 must produce (feeds L3, P1, and later P2/P4)
1. Per-(item,policy) 3-way logits on the base-distribution test split -> for P1 (critic is
   well-calibrated on base): ECE / adaptive-ECE per policy, reliability diagrams, with
   bootstrap CIs.
2. Per-policy macro-F1 with VIOLATED as positive class over APPLICABLE items (teacher label
   != N/A), N/A excluded from the F1 denominator — the LOCKED L3 metric. CV over the 10
   per-policy F1 values; L3 PASS iff CV>0.15 and its bootstrap CI excludes 0.15 from below.
3. The frozen D0 checkpoint + saved logits, so D1..D6 evaluation reuses the SAME frozen
   critic without retraining.

## Boundaries (Red — do NOT change without sign-off)
- Backbone family/scale, the per-policy-head structure, the 3-way (incl. N/A) target, the
  frozen-after-D0 rule, and the locked L3 metric. Hyperparameters (lr, rank, epochs) are
  Green and may be tuned with a note, provided the calib/test split discipline is preserved
  (calib is used ONLY for temperature scaling and early-stop; test is never seen in training).
