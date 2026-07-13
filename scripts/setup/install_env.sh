#!/usr/bin/env bash
# PCCD setup — project deps on top of an existing PyTorch/CUDA base image (AutoDL).
# Run: bash scripts/setup/install_env.sh
set -euo pipefail
echo "PCCD env install (assumes torch+CUDA base image already present)"

pip install -U pip
# pin the verified versions; torch is assumed provided by the base image
pip install transformers==5.13.1 trl==1.8.0 peft==0.19.1 vllm==0.25.0 \
            accelerate bitsandbytes datasets
pip install mapie==1.4.1 netcal==1.4.0 rewardbench==0.1.4
pip install numpy scipy scikit-learn pandas matplotlib seaborn tqdm huggingface_hub hf_transfer

echo -e "\nVerify:"
python - <<'PY'
for m in ["torch","transformers","trl","peft","vllm","mapie","netcal","rewardbench"]:
    try:
        mod=__import__(m); print(f"OK {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
PY
echo "DONE env install."
