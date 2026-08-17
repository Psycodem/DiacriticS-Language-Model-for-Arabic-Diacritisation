#!/bin/bash
# setup_env.sh — one-time environment build on KAUST Ibex.
#
#   bash setup_env.sh
#
# Run this ON A COMPUTE NODE, not the login node — building torch/bitsandbytes
# on glogin will get the process killed for CPU use. Grab an interactive slot:
#
#   srun --time=1:00:00 --gpus=1 --cpus-per-gpu=8 --mem=32G --pty bash
#
# ASSUMPTIONS TO VERIFY (I could not reach Ibex to confirm these):
#   - `module load` names below exist; check with `module avail cuda` / `module avail miniconda`
#   - your scratch is /ibex/scratch/$USER
# Correct them and re-run if they are wrong.

set -euo pipefail

SCRATCH="/ibex/scratch/${USER}"
ENV_DIR="${SCRATCH}/envs/diacritics"
CACHE_DIR="${SCRATCH}/hf_cache"

mkdir -p "${SCRATCH}/envs" "${CACHE_DIR}"

module purge
module load cuda/12.2 || module load cuda/11.8 || echo "[warn] no cuda module loaded"

# Prefer a plain venv on top of the system python — fewer moving parts than conda,
# and Ibex's python3 is recent enough.
if [ ! -d "${ENV_DIR}" ]; then
    python3 -m venv "${ENV_DIR}"
fi
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

pip install --upgrade pip wheel

# torch first, matched to the CUDA module above, so nothing else drags in a
# mismatched build as a transitive dependency.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install \
    "transformers>=4.45" \
    "datasets>=2.20" \
    "accelerate>=0.34" \
    "peft>=0.13" \
    "bitsandbytes>=0.44" \
    sentencepiece \
    pyarabic \
    jiwer \
    pandas \
    matplotlib \
    tqdm \
    huggingface_hub

python - <<'PY'
import torch, transformers, peft, bitsandbytes
print("torch       ", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB",
          "bf16:", torch.cuda.is_bf16_supported())
PY

cat <<EOF

Environment ready: ${ENV_DIR}
HF cache:          ${CACHE_DIR}

google/gemma-4-E4B-it is a GATED repo. Before its run will work:
  1. accept the licence on its Hugging Face model page
  2. create a NEW token (the one in git history must be revoked)
  3. store it on Ibex, readable only by you:
       printf '%s' 'hf_yourNewToken' > ~/.hf_token && chmod 600 ~/.hf_token
     The sbatch scripts read that file; the token never enters the repo.
EOF
