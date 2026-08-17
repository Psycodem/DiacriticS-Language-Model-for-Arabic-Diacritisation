#!/bin/bash
#SBATCH --job-name=gemma-lora-50pct
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@kaust.edu.sa

# gemma-4-E4B-it — LoRA on 50% of the Sadeed_Tashkeela train split (data-scaling point).
#
# Part of a 4-point curve per model: 10% / 30% / 50% / 100%. The subsets are
# NESTED — the script shuffles with a fixed seed (42) and takes a prefix, so the
# 10% rows are inside the 30%, which are inside the 50%, inside the 100%. Without
# that, the four points would be unrelated samples rather than a curve, and a
# plain head-slice would also have been biased by source book (Sadeed_Tashkeela is
# ordered by `filename`).
#
# DIAC_RUN_TAG puts outputs in their own directories, so the four runs of the same
# model cannot overwrite each other's adapter or metrics_summary.csv.
#
# --time derivation (measured 7.21 s/step at effective batch 96 on 3 GPUs):
#     steps      = 1,042,697 x 0.5 / 96 = 5,430
#     training   = 10.9 h
#     evals      = 27 x ~1.5 min = 0.7 h
#     load+score = 3.1 h  (score is 4,185 generations, constant across fractions)
#     expected   = 14.7 h   -> +1 h requested -> --time=16:00:00

set -euo pipefail

WORKDIR="${HOME}/gemma4-lora"
ENV_DIR="${HOME}/envs/diac_clean"
SCRIPT="${WORKDIR}/Train_LoRA_Gemma_4_Diacritization.py"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
[ -f "${SCRIPT}" ] || { echo "[FATAL] ${SCRIPT} missing" >&2; exit 1; }

module purge
module load machine_learning/2026.02
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

export HF_CACHE_DIR="${WORKDIR}/hf_cache"
export HF_HOME="${HOME}/hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
[ -f "${HOME}/.hf_token" ] && export HF_TOKEN="$(cat "${HOME}/.hf_token")"

export DIAC_TRAIN_FRACTION=0.5
export DIAC_RUN_TAG=50pct

echo "=================================================="
echo "job     : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "model   : gemma-4-E4B-it"
echo "fraction: 50% of train split"
echo "started : $(date -Is)"
echo "=================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch, transformers, peft; print(f'[INFO] torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}')"
echo

torchrun --standalone --nnodes=1 --nproc_per_node=3 "${SCRIPT}"

echo
echo "finished: $(date -Is)"
cat "${WORKDIR}/eval_outputs-50pct/metrics_summary.csv" 2>/dev/null || true
