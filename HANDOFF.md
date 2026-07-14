# PCCD — Handoff to Local Codex (on AutoDL)

This document hands the **execution** of the PCCD project to a local Codex agent running
on the AutoDL box (2× RTX PRO 6000, 96GB each). PaperGuru retains **scientific oversight
and final paper writing** and reviews everything via this GitHub repo.

Read this whole file before running anything. Section 8 lists decisions you may NOT make
alone — surface those back to the human / PaperGuru.

---

## 0. Division of labor

**You (local Codex) own execution:**
- Fix the current disk blocker, download models/data, run Gate D probe, Day-2 labeling,
  and all subsequent training/experiments on the GPUs.
- Keep code, logs, and results updated in the local clone in real time.
- Push every milestone to `github.com/TrigoldYixinWan/PCCD` on a feature branch → open a PR.

**PaperGuru retains oversight:**
- Reviews your PRs for logic, data-leakage, and alignment with the propositions below.
- Judges the 4 gate outcomes (esp. G2, the make-or-break).
- Writes the final AAAI paper (citations, figures, ref_verify).

**The human** rents the AutoDL box, watches progress, and makes the calls flagged in §8.

---

## 1. Research object (do NOT drift from this)

**Gap 1 — "Frozen Critic Calibration Transfer under Local Adaptation."**
We study how a **frozen, independent critic** loses **calibration** on the **output
distribution of a locally-adapted policy**. The precise cut vs. Shihab et al. 2026
(Continual Calibration) — our object is:
- an **independent frozen critic** (not the policy's own head), evaluated on the
  **policy's output distribution**,
- at **per-policy granularity**,
- the degradation is **FN-asymmetric** (false-negatives grow faster — unsafe passed as safe),
- **predictable from adaptation strength (KL divergence)** — a scaling law,
- **recoverable via per-policy temperature scaling**.

If a code/design choice would blur any of these five distinctions from Shihab 2026,
STOP and flag it (§8).

### Central propositions & weights
- **P2 + P3 = 35%** (main): critic degrades under local adaptation; degradation is
  measurable per-policy on the policy output distribution.
- **P5 = 25%** (main): degradation is FN-asymmetric.
- **P6 = 20%** (main): degradation predictable from KL(adapted ‖ base), i.e. a scaling law.
- **P1 = 10%** (support): the frozen critic is well-calibrated on the base distribution.
- **P4 = 10%** (support): per-policy temperature scaling recovers calibration.

---

## 2. Experimental design (fixed)

- **10-policy stack**: H1–H5 hard (severity high/medium), S1–S3 soft, T1–T2 task.
  Defined in `configs/policy_taxonomy.json` (severity_weights high=4/med=2/low=1,
  plus PKU/UF harm-category mappings).
- **Teacher is LABEL-ONLY**: the 32B teacher emits only labels/judgments per the schema
  in `configs/teacher_schema.json`. It MUST NEVER write "chosen" responses — this avoids
  the More-is-Less shortcut. Do not add any code path that has the teacher generate
  candidate responses.
- **Adaptation grid D0–D6** (for the scaling law): D0 base, D1 system-prompt,
  D2/D3/D4/D5 LoRA r=4/8/16/32, D6 DPO β=0.1.
- **Model families**: primary Qwen2.5 (32B teacher, 7B critic+policy). Second family for
  generalization = **Qwen2.5-14B** (Llama-3.1-8B is gated, avoid). APM dataset is not
  public → soft-preference pairs are **self-built** (see `src/sample_data.py`).

### 4 gates (go/no-go)
- **G1** (Day 3–4): policy heterogeneity — the 10 policies produce distinguishable
  behavior/output distributions.
- **G2** (Day 5, MAKE-OR-BREAK): critic degradation exists AND is FN-asymmetric.
- **G3** (Day 6): scaling law fits (degradation vs KL).
- **G4** (Day 7): per-policy temperature scaling recovers calibration.

Full timeline: `configs/plan_9day.md`.

---

## 3. CURRENT BLOCKER — disk full (fix this FIRST)

Root cause: models were downloaded to `/root/models`, but `/root` is the **small AutoDL
system disk**. It filled up → `No space left on device` → `/tmp` unusable → even
`import torch`/`import datasets` fail (dill needs a tempdir) → git lock fails.

The big disk on AutoDL is **`/root/autodl-tmp`**. Everything heavy must live there.

**Already fixed in this branch (`day2/teacher-labeling`):**
- `scripts/setup/env.sh` — single source of truth for paths. `source` it before any step.
  It points MODELS_DIR, DATA_DIR, PCCD_OUT, HF_HOME, HUGGINGFACE_HUB_CACHE, TMPDIR,
  VLLM_CACHE_ROOT, TRITON_CACHE_DIR all under `$PCCD_DISK/pccd/...` where
  `PCCD_DISK=/root/autodl-tmp`.
- `scripts/setup/cleanup_systemdisk.sh` — removes `*.incomplete` and misplaced
  `/root/models`, `/root/data`, `/root/.cache/huggingface`.
- `download_models.sh` / `download_data.sh` — serial (`--max-workers 1`,
  `HF_HUB_ENABLE_HF_TRANSFER=0`; the parallel fetch was a blowup factor), endpoint
  fallback (huggingface.co → hf-mirror), resumable.
- `scripts/day2/run_day2.sh` — all pool/labels on the data disk; only small logs in repo.

### Fix procedure
```bash
cd /root/PCCD && git checkout day2/teacher-labeling && git pull
df -h                                       # CONFIRM /root/autodl-tmp size & free
bash scripts/setup/cleanup_systemdisk.sh    # answer y to reclaim system disk
source scripts/setup/env.sh                 # must print PCCD_DISK=/root/autodl-tmp
```
**If `/root/autodl-tmp` does not exist or is < ~200GB free, STOP and flag (§8)** — we may
need to drop the 14B family or resize the disk. 32B(~66GB)+7B(~15GB) alone need ~85GB plus
cache/tmp headroom (budget ~120GB).

---

## 4. Bring-up sequence (after disk fixed)

```bash
source scripts/setup/env.sh
# 1) download (background, resumable). 32B + 7B first; 14B optional (WITH_14B=1).
nohup bash scripts/setup/download_models.sh > logs/dl_models.log 2>&1 &
nohup bash scripts/setup/download_data.sh   > logs/dl_data.log   2>&1 &
watch -n 30 'df -h $PCCD_DISK | tail -1; ls $MODELS_DIR/qwen32b/*.safetensors 2>/dev/null | wc -l'
# expect 17 shards for 32B, 4 for 7B, then datasets present.

# 2) sanity: torch/datasets import (must work now that TMPDIR is on data disk)
python -c "import torch,datasets,transformers,trl,peft,vllm; print('imports OK')"

# 3) Gate D — 32B teacher vLLM probe (never ran yet)
CUDA_VISIBLE_DEVICES=0 python scripts/stress/03_vllm_teacher_probe.py \
  --model $MODELS_DIR/qwen32b --n 64 2>&1 | tee logs/stress_03.log

# 4) Day-2 teacher labeling + static audit
bash scripts/day2/run_day2.sh 2>&1 | tee logs/day2_full.log
```

---

## 5. Installed runtime on AutoDL (differs from pinned — this is fine, adapt to it)
Python 3.12.3 (miniconda). torch 2.11.0+cu130, transformers 4.51.0, **trl 0.19.1**
(NOT 1.8.0 — APIs differ, check before using DPO/SFT trainers), peft 0.19.1, vllm 0.25.0,
mapie 1.4.1, netcal 1.4.0, datasets 5.0.0. GPUs: 2× RTX PRO 6000 Blackwell, cc12.0,
CUDA 13.0, driver 580.119.02. P2P over SYS (no NVLink, 25.6 GB/s) — fine, we never pool
memory (one 32B teacher per card, independent shards).

## 6. Dataset schemas (confirmed)
- **PKU-SafeRLHF**: `prompt`, `response_0/1`, `is_response_0/1_safe`, 19 harm categories,
  `severity_level`.
- **UltraFeedback**: `instruction`, `completions[]` with annotations
  (helpfulness/honesty/instruction_following/truthfulness Ratings).

## 7. Git workflow (keep this)
- Work on feature branches: `dayN/<topic>` (current: `day2/teacher-labeling`).
- Each milestone → PR → PaperGuru reviews → merge to `main`.
- **PR #1 MERGED** (Day-1 stress verdict + NCCL fix). **PR #2 OPEN** = this branch
  (Day-2 pipeline + disk fix). Update PR #2's description as you land the disk fix result.
- Remote already has the PAT. Do NOT commit secrets, model weights, datasets, or anything
  under `$PCCD_DISK/pccd/` — `.gitignore` keeps `outputs/`, models, data out; only small
  logs/results summaries belong in the repo.

## 8. Decisions you (Codex) may NOT make alone — flag to human/PaperGuru
1. Any change that blurs the 5 distinctions vs Shihab 2026 (§1).
2. Making the teacher generate responses (must stay label-only).
3. Changing policy taxonomy, adaptation grid D0–D6, or gate criteria.
4. Dropping/replacing the 2nd model family, or changing datasets.
5. If a gate FAILS (esp. G2) — do not "fix" data/metrics to pass; report the true result.
6. If data disk is too small to hold the planned assets (§3).
7. Reinterpreting proposition weights or what counts as FN-asymmetry.

When in doubt: push your work + a note in the PR describing the question, and stop.
```
