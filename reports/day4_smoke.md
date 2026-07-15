# Day 4 pre-training smoke: TRL/PEFT on Transformers 5

Date: 2026-07-15  
Code commit used: `a2c73f35e2566dd4210cead6b4a5c84fefb4f3e9`  
Verdict: **PASS** (hard precondition only; this is not a G1-G4 scientific verdict)

## Scope

This test verifies that the pinned runtime can execute one LoRA SFT optimizer
step and one LoRA DPO optimizer step. It uses a randomly initialized 4.86M-parameter
tiny Qwen2 model, the already-local Qwen tokenizer, two synthetic examples per
trainer, and no research split. It therefore does not select or validate the D0
critic architecture/training protocol, which remains Red-locked pending review.

## Exact commands

```bash
source scripts/setup/env.sh
python -m pip freeze > "$PCCD_OUT/env/pip_freeze_before_day4_smoke.txt"
python -m pip install trl==1.8.0
python -m pip uninstall -y rewardbench
python -m pip check
CUDA_VISIBLE_DEVICES=0 python scripts/day4/smoke_trl_peft.py \
  2>&1 | tee logs/day4_smoke.log
python -m pip freeze > "$PCCD_OUT/env/pip_freeze_after_day4_smoke.txt"
```

The single-GPU visibility is intentional: the first launch with both GPUs visible
triggered Transformers' implicit `DataParallel` path and failed because replica-1
token indices were sent to an embedding resident on `cuda:0`. That launch did not
complete an optimizer step and is retained at
`logs/day4_smoke_attempt1_dp_fail.log`. Restricting this API smoke to one GPU is a
launch/configuration fix, not a change to a scientific protocol.

## Environment

| Component | Version/value |
|---|---:|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.13.1 |
| TRL | 1.8.0 |
| PEFT | 0.19.1 |
| CUDA | 13.0 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Post-change `pip check` | No broken requirements found |

## Results

| Trainer path | Global step | Finite training loss | Trainable LoRA params | Total params | Result |
|---|---:|---:|---:|---:|---:|
| `SFTTrainer` + PEFT LoRA | 1 | 11.932379722595215 | 448 | 4,863,104 | PASS |
| `DPOTrainer` + PEFT LoRA | 1 | 0.6931471824645996 | 448 | 4,863,104 | PASS |

Both required paths imported, constructed a PEFT model, performed forward/backward
and optimizer work, and returned `global_step == 1` with finite loss. This clears
the TRL/PEFT@Transformers-5 API hard precondition. D0 and D2-D6 were not started.

## Integrity and raw artifacts

| Artifact | SHA-256 |
|---|---|
| `scripts/day4/smoke_trl_peft.py` | `36215a559271b74e8378be09765bc8a0769c69043869056eb4caf5010cdd5984` |
| `requirements.txt` | `772e763af47ba60f3664e855511f868d57e9ef3baf87152ed2aee9021e338c3a` |
| `logs/day4_smoke.log` | `fa8aa9d58d2b8b1c4201c86e694a3554f5664e97f128693693c55c059b6c20aa` |
| `$PCCD_OUT/smoke/day4_trl_peft/summary.json` | `3d2c95181d1f08b529f0b335ec0cbaae617791e0ff500ff37619aa6f09abe94a` |
| `$PCCD_OUT/env/pip_freeze_before_day4_smoke.txt` | `9bbc82011a5d2ffbcb6456aae7e8e01f9119003bc6b7610eee96c0763aee60e2` |
| `$PCCD_OUT/env/pip_freeze_after_day4_smoke.txt` | `e6609274e3a8bbb23229007c7a1771c993c2fd49e3d742bd5fd13381ef6d8185` |

Raw outputs are on AutoDL under
`/root/autodl-tmp/pccd/outputs/smoke/day4_trl_peft/` and package snapshots under
`/root/autodl-tmp/pccd/outputs/env/`. Repository logs are
`logs/day4_smoke.log`, `logs/day4_smoke_attempt1_dp_fail.log`,
`logs/day4_trl_upgrade.log`, and `logs/day4_pip_check.log`.

## Anomalies and decision

- TRL 0.19.1 was not compatible with the fixed Transformers 5.13.1 runtime;
  upgrading only TRL to 1.8.0 resolved the removed-symbol import failure.
- RewardBench 0.1.4 was still installed despite its documented isolation rule.
  Removing it restored a clean `pip check`; it remains deferred to a separate
  Transformers-4 environment on Day 9.
- The first two-GPU-visible launch exposed an implicit DataParallel device mismatch.
  The required single-GPU smoke then passed both paths. This does not authorize an
  implicit multi-GPU strategy for later training.

**Pre-training smoke verdict: PASS. Stop for PaperGuru confirmation before any D0
critic architecture choice, D0 training, or D2-D6 adaptation run.**
