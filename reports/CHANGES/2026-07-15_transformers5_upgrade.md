# Change: Upgrade transformers 4.51.0 -> 5.13.1 for vLLM 0.25.0 (severity: Yellow)

Date / commits: 2026-07-15 / `c453fce`..`a59f3fc` (env/runtime), recorded by PaperGuru review

Trigger: At Gate D bring-up, `import vllm` (vllm==0.25.0) failed under the installed
`transformers==4.51.0`; vLLM 0.25.0 explicitly removes Transformers-v4 support. Exact
failure captured in `/root/PCCD/logs/stress_03_import_blocked_20260715.log`.

What I changed:
- Upgraded transformers 4.51.0 -> 5.13.1 in the live env; updated `requirements.txt` pin.
- Corrected the `requirements.txt` trl pin to the actually-installed `0.19.1` (the file
  had a stale `1.8.0` that never matched the box).
- Marked `rewardbench==0.1.4` as INCOMPATIBLE with transformers 5.x and commented it out
  of the main env; it must be installed in an isolated env only for the Day-9 RewardBench
  phase. It is not used in Gate D or Day-2 labeling, so nothing here depends on it.
- Green runtime plumbing (env.sh): expose Miniconda Python + CUDA13/torch lib paths to
  detached SSH jobs; set `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer 0.6.x misdetects
  SM120/Blackwell during sampling-kernel JIT; attention stays on FlashAttention).
- Saved the pre-change environment: `$PCCD_OUT/env/pip_freeze_before_gate_d_fix.txt`.

Why this and not an alternative: vLLM 0.25.0 is required for the teacher throughput path
and cannot run on transformers 4.x. Downgrading vLLM would forfeit the validated Blackwell
support. Upgrading transformers is the supported direction. RewardBench is a Day-9-only
eval and is cleanly isolable, so its 4.x requirement does not justify holding back the core
stack.

Impact on propositions/gates: No change to P1–P6 definitions or G1–G4 criteria. Gate D was
measured on the exact operational config (transformers 5.13.1, FlashInfer sampler off) that
Day-2 labeling will use, so the throughput/JSON numbers are representative.
OPEN RISK (must verify BEFORE Day-4 training): trl 0.19.1 + peft 0.19.1 DPO/LoRA behavior
under transformers 5.x is not yet validated. A tiny smoke-train (1-step LoRA + 1-step DPO)
must pass before the D2–D6 adaptation grid is trusted. Tracked as the pre-training gate.

Reversibility: pin transformers back to 4.51.0 only if paired with a vLLM downgrade; both
recorded in the saved pip freeze. No scientific result depends on the version beyond the
representativeness note above.

Open question for PaperGuru: none — flagged the trl/peft@transformers5 verification as a
required pre-training check below.
