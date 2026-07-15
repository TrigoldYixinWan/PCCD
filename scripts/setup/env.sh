#!/usr/bin/env bash
# PCCD central paths — source this before any download/train/label step:
#   source scripts/setup/env.sh
#
# CRITICAL on AutoDL: /root (system disk) is small (~30-50GB). ALL heavy data
# (models, datasets, HF cache, tmp, outputs) MUST live on the big data disk,
# which on AutoDL is /root/autodl-tmp. This file points everything there and
# also redirects TMPDIR so /tmp on the system disk never fills up.

# AutoDL exposes its managed Python through Miniconda in interactive login
# shells, but detached/non-login SSH jobs do not always inherit that PATH.
if [ -x /root/miniconda3/bin/python ]; then
  export PATH="/root/miniconda3/bin:$PATH"
fi

# Detached AutoDL jobs also miss the CUDA 13 and PyTorch wheel library paths.
# vLLM imports TorchCodec even for text-only models, which needs libnvrtc.so.13.
for _pccd_lib in \
  /root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib \
  /root/miniconda3/lib/python3.12/site-packages/torch/lib; do
  if [ -d "$_pccd_lib" ]; then
    export LD_LIBRARY_PATH="$_pccd_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
unset _pccd_lib

# flashinfer 0.6.x mis-parses SM120 during JIT sampling-kernel warmup and
# reports the Blackwell GPU as older than SM75. Use vLLM's supported PyTorch
# sampling fallback; attention remains on FlashAttention.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Pick the data disk: prefer autodl-tmp, fall back to autodl-fs, else /root.
if [ -d /root/autodl-tmp ]; then
  export PCCD_DISK=/root/autodl-tmp
elif [ -d /root/autodl-fs ]; then
  export PCCD_DISK=/root/autodl-fs
else
  export PCCD_DISK=/root
  echo "[env] WARNING: no /root/autodl-tmp found; using /root (system disk may be small!)"
fi

export MODELS_DIR="$PCCD_DISK/pccd/models"
export DATA_DIR="$PCCD_DISK/pccd/data"
export PCCD_OUT="$PCCD_DISK/pccd/outputs"
export HF_HOME="$PCCD_DISK/pccd/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$PCCD_DISK/pccd/tmp"
export TEMP="$TMPDIR"; export TMP="$TMPDIR"

mkdir -p "$MODELS_DIR" "$DATA_DIR" "$PCCD_OUT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

# vLLM / triton scratch also on data disk
export VLLM_CACHE_ROOT="$PCCD_DISK/pccd/vllm"
export TRITON_CACHE_DIR="$PCCD_DISK/pccd/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

echo "[env] PCCD_DISK=$PCCD_DISK"
echo "[env] MODELS_DIR=$MODELS_DIR"
echo "[env] DATA_DIR=$DATA_DIR"
echo "[env] PCCD_OUT=$PCCD_OUT"
echo "[env] HF_HOME=$HF_HOME  TMPDIR=$TMPDIR"
df -h "$PCCD_DISK" | tail -1
