# PCCD — Frozen Critic Calibration Transfer under Local Adaptation

Research project studying how a **frozen, independent critic** loses calibration on the
**output distribution of a locally adapted policy**, and whether the degradation is
(i) policy-heterogeneous, (ii) false-negative asymmetric, (iii) predictable from adaptation
strength, and (iv) recoverable with lightweight per-policy recalibration.

Target venue: AAAI. Hardware: 2 × NVIDIA RTX PRO 6000 96GB, 9-day budget.

## Central propositions

| ID | Proposition | Weight |
|----|-------------|--------|
| P2+P3 | Per-policy heterogeneous + FN-asymmetric degradation | 35% (main) |
| P5 | Adaptation-strength → drift scaling law (ΔECE vs KL) | 25% (main) |
| P6 | Recalibration budget–risk frontier (per-policy) | 20% (main) |
| P1 | Degradation existence for an *independent* frozen critic | 10% (support) |
| P4 | Lightweight per-policy recalibration sample complexity | 10% (support) |

## Cut vs prior work (must appear in intro)
Shihab et al. 2026 (Continual Calibration) show a **policy's own** conformal coverage
degrades under continual fine-tuning. We study a **different** object: a **frozen independent
critic** on the adapted **policy output distribution**, at **per-policy** granularity, with
**FN-asymmetric** degradation, **predictable** from adaptation strength (KL), and recoverable
via **per-policy temperature scaling**.

## Repo layout
```
scripts/setup/     environment + model/data download
scripts/stress/    Day-1 dual-GPU stress test
configs/           policy taxonomy, teacher JSON schema, experiment configs
src/               labeling, critic training, adaptation, degradation eval, recalibration
outputs/           labels, checkpoints, results, figures  (gitignored except manifests)
logs/              run logs (gitignored)
```

## Resource verification (2026-07, all PASS)
- Models: Qwen2.5-32B-Instruct (65.5GB, teacher), Qwen2.5-7B-Instruct (15.2GB, critic/policy),
  Qwen2.5-14B-Instruct (29.5GB, cross-scale). Llama-3.1-8B is **gated** → replaced by Qwen-14B.
- Datasets: PKU-Alignment/PKU-SafeRLHF (19 harm categories + severity_level), openbmb/UltraFeedback
  (helpfulness/honesty/instruction_following/truthfulness), walledai/HarmBench, allenai/reward-bench.
- Libraries: trl 1.8.0, mapie 1.4.1, netcal 1.4.0, rewardbench 0.1.4, peft 0.19.1,
  transformers 5.13.1, vllm 0.25.0, unsloth.
- APM (Spohn 2026) not public on HF → self-built soft-preference test set.

## 9-day plan
See `configs/plan_9day.md`.
