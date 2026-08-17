#!/bin/bash
#SBATCH --job-name=diac-ft
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL

# QLoRA fine-tuning on Ibex.
#
#   sbatch --job-name=qwen4b  train.sbatch Qwen/Qwen3.5-4B
#   sbatch --job-name=gemma   train.sbatch google/gemma-4-E4B-it
#
# Extra flags pass straight through to finetune_qlora.py:
#   sbatch train.sbatch Qwen/Qwen3.5-4B --epochs 3 --train-subset 20000
#
# ASSUMPTIONS TO VERIFY: --constraint=a100 and the cuda module name. Check what
# your allocation actually offers with `sinfo -o "%20N %10c %10m %25f %10G"`.
# A 4B QLoRA fits on a 32GB V100 at batch-size 1-2; drop the constraint line if
# a100 nodes queue too long.

set -euo pipefail

MODEL="${1:?usage: sbatch train.sbatch <hf-model-id> [extra finetune_qlora.py args]}"
shift || true

SCRATCH="/ibex/scratch/${USER}"
ENV_DIR="${SCRATCH}/envs/diacritics"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the "Train Related" dir
SLUG="${MODEL//\//__}"
RUN_DIR="${SCRATCH}/runs/${SLUG}"

mkdir -p logs "${RUN_DIR}"

module purge
module load cuda/12.2 || module load cuda/11.8 || true
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

export HF_CACHE_DIR="${SCRATCH}/hf_cache"
export HF_HOME="${SCRATCH}/hf_home"
export TOKENIZERS_PARALLELISM=false
# Read the token from a 0600 file rather than baking it into the script.
[ -f "${HOME}/.hf_token" ] && export HF_TOKEN="$(cat "${HOME}/.hf_token")"

echo "=============================================="
echo "job      : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "model    : ${MODEL}"
echo "run dir  : ${RUN_DIR}"
echo "started  : $(date -Is)"
echo "=============================================="
nvidia-smi

# --resume is safe on a fresh dir (Trainer ignores it when no checkpoint exists),
# so a requeued job picks up where the last one was cut off.
srun python "${PROJECT}/finetune_qlora.py" \
    --model "${MODEL}" \
    --out-dir "${RUN_DIR}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --resume \
    "$@"

echo "finished : $(date -Is)"
echo "adapter  : ${RUN_DIR}/final_adapter"
