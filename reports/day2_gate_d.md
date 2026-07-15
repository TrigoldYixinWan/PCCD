# Day 2 / Gate D teacher-throughput report

Date: 2026-07-15

Code revision used for the successful run: `06672e2`

Status: **candidate PASS from the probe script; awaiting PaperGuru confirmation**

> PaperGuru verdict (2026-07-15): **Gate D FORMALLY PASSED.** Throughput 37,964/hr/GPU is
> 76x the 500/hr target; JSON parse 100%; 32B fits one card with 21 GiB KV headroom. The
> 86.4 GiB high-water is from a prior failed attempt (conservative reporting, accepted).
> Day-2 labeling is CLEARED. The transformers 4.51->5.13.1 upgrade is recorded in
> reports/CHANGES/2026-07-15_transformers5_upgrade.md; the trl/peft-under-transformers-5
> DPO/LoRA smoke test is a REQUIRED pre-Day-4 check (do not train the D2-D6 grid until it
> passes).

## 1. Commands

The model downloader first verified exact snapshots of 17/17 Qwen2.5-32B shards and
4/4 Qwen2.5-7B shards. Gate D then ran on GPU 0 only:

```bash
cd /root/PCCD
source scripts/setup/env.sh
CUDA_VISIBLE_DEVICES=0 python scripts/stress/03_vllm_teacher_probe.py \
  --model "$MODELS_DIR/qwen32b" --n 64 2>&1 | tee logs/stress_03.log
```

Before the successful run, the following Green runtime repairs were required. The
pre-change Python environment was saved before modifying packages.

```bash
python -m pip freeze > \
  "$PCCD_OUT/env/pip_freeze_before_gate_d_fix.txt"
python -m pip install transformers==5.13.1
apt-get install -y --no-install-recommends libavdevice58 libavfilter7
```

`scripts/setup/env.sh` at `06672e2` also exposes the installed CUDA 13 and PyTorch
wheel libraries to detached jobs and sets `VLLM_USE_FLASHINFER_SAMPLER=0`.

## 2. Results

| Measurement | Observed | Gate-D target |
|---|---:|---:|
| Model weight shards | 17/17 | 17/17 |
| Parent-reported engine load time | 59.9 s | fits and initializes |
| Probe prompts | 64 | 64 |
| Timed generation | 6.1 s | -- |
| Throughput | 10.55 prompts/s = 37,964/hour/GPU | >=500/hour/GPU |
| JSON parse success | 64/64 = 100% | >=99% on real prompts |
| Highest external GPU-memory sample during bring-up | 86,433 MiB | <97,250 MiB |

The successful run loaded 61.04 GiB of weights and reported 21.13 GiB available for
KV cache (86,560 tokens at a 4,096-token model limit). GPU 1 remained unused. The
86,433-MiB memory sample came from the immediately preceding attempt with the same
model, dtype, maximum context, and 0.90 memory-utilization setting; the successful run
was not continuously sampled, so this is a bring-up high-water observation rather than
a profiler-derived peak for the final process.

## 3. Verdict

The probe script emitted:

```text
[THROUGHPUT] 64 prompts in 6.1s => 10.55/s = 37964/hour/GPU
[GATE] target >=500/hour/GPU : PASS
[JSON parse] 64/64 = 100%  (gate >=99% on real prompts)
```

This is a **candidate Gate-D pass** because both instrumented thresholds were met by a
large margin and the 32B teacher fit on one GPU. Per the project's Red authority rule,
Codex does not convert that script result into the formal gate decision; Day-2 labeling
remains stopped pending PaperGuru confirmation.

## 4. Anomalies and Green adaptations

- The downloaded models completed and verified successfully; no data, model, prompt,
  metric, or gate threshold was altered.
- The installed `transformers==4.51.0` could not import with `vllm==0.25.0`, whose source
  explicitly removes Transformers-v4 support. Transformers was updated to the repository
  pin `5.13.1`. Core imports (`torch`, `datasets`, `transformers`, `trl`, `peft`, `vllm`)
  then passed.
- `rewardbench==0.1.4` declares Transformers 4.51.0 and does not import under 5.13.1.
  RewardBench is not used in Gate D or Day-2 teacher labeling. Its environment must be
  isolated or otherwise resolved before a later RewardBench phase; no claim involving
  RewardBench is made here.
- TorchCodec's text-independent vLLM import required existing CUDA 13/PyTorch library
  paths plus Ubuntu FFmpeg-4 runtime libraries (`libavdevice58`, `libavfilter7`).
- FlashInfer 0.6.x misdetected the SM120 GPU during sampling-kernel JIT warmup. vLLM's
  documented `VLLM_USE_FLASHINFER_SAMPLER=0` fallback was used. Sampling fell back to
  PyTorch while attention remained on FlashAttention. The reported throughput therefore
  measures the exact operational configuration proposed for Day-2 labeling.
- Two environment-only attempts and one FlashInfer warmup attempt exited before producing
  probe outputs. They are not counted as Gate-D measurements.

## 5. Raw artifacts

- Successful Gate-D log: `/root/PCCD/logs/stress_03.log`
- Transformers-v4 import failure: `/root/PCCD/logs/stress_03_import_blocked_20260715.log`
- TorchCodec library failure: `/root/PCCD/logs/stress_03_torchcodec_blocked_20260715.log`
- FlashInfer SM120 failure: `/root/PCCD/logs/stress_03_flashinfer_blocked_20260715.log`
- Pre-fix package freeze:
  `/root/autodl-tmp/pccd/outputs/env/pip_freeze_before_gate_d_fix.txt`
