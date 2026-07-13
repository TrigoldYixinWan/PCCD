# PCCD 9-day plan (2× RTX PRO 6000 96GB)

Core budget ~184 GPUh of 432 theoretical (43%); rest is buffer.

| Day | GPU0 | GPU1 | Gate |
|-----|------|------|------|
| 1 | stress + 32B pilot (200) | 7B dry-run + libs | JSON>=99%, 32B on 1 card, no throttle |
| 2 | teacher shard A (4k+0.5k) | teacher shard B (4k+0.5k+1k) | throughput>=500/hr/GPU |
| 3 | audit(400)+conflict(400) | critic seed1 | **G1** per-policy F1 CV>0.15 |
| 4 | critic seed2/3 | total-score s1-3 + base calib | G1 confirmed |
| 5 | D2/D3 adapt+resp | D0/D1 resp + critic/teacher infer | **G2** ΔECE>0 & ΔFN>ΔFP |
| 6 | D4/D5 adapt | D6 DPO + scaling-law fit | **G3** ΔECE~KL R²>0.7 |
| 7 | recalib 125 runs + full-retrain | scaling verify pts + CalArena | **G4** per-policy T @n=100 recovers |
| 8 | Qwen-14B distill+D3/D5 | online cascade latency | cross-scale |
| 9 | reruns + stats | figures export | 12 deliverables |

## Adaptation-strength grid (for P5)
D0 base, D1 system-prompt, D2 LoRA r=4, D3 r=8, D4 r=16, D5 r=32, D6 DPO β=0.1.

## Gates
- G1 policy heterogeneity (Day3-4)
- **G2 degradation + FN-asymmetry (Day5)** — paper make-or-break
- G3 scaling law fit (Day6)
- G4 recalibration effective (Day7)

## Deliverables
~16k teacher labels, 400 audit, 400 conflict, critic checkpoints (2 methods ×3 seeds),
7-point per-policy degradation, ΔECE–KL scaling law, recalibration budget–risk frontier,
online cascade, RewardBench2/HarmBench/CalArena, Qwen-14B cross-scale, error analysis.
