#!/bin/bash
#SBATCH --job-name=qwen-lora-3gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=39:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@kaust.edu.sa

# Qwen3.5-4B LoRA — 1 epoch on 3 GPUs (DDP).
#
# --ntasks=1 with torchrun, NOT --ntasks=3 with srun. torchrun spawns and ranks
# the 3 worker processes itself; letting SLURM also create 3 tasks would start
# three independent torchruns, i.e. 9 processes fighting over 3 GPUs.
# --cpus-per-task covers all 3 ranks' dataloader workers, hence 24 not 8.
#
# FAIR-COMPARISON CONTRACT (must match the gemma-4-E4B run):
#   effective batch 96, 1 epoch, lr 2e-4 cosine, warmup 300, LoRA r16/a32/d0.05
#   on all 7 projections, MAX_SEQ_LENGTH 1024, seed 42, same splits.
#   Only PER_DEVICE_TRAIN_BATCH_SIZE differs (memory), with accumulation derived
#   so the product stays 96.

set -euo pipefail

WORKDIR="${HOME}/qwen35-lora"
ENV_DIR="${HOME}/envs/diac_clean"
SCRIPT="${WORKDIR}/Train_LoRA_Qwen_3_5_4B_Diacritization.py"
NGPU=3

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

echo "=================================================="
echo "job     : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "gpus    : ${NGPU}"
echo "started : $(date -Is)"
echo "=================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch, transformers, peft; print(f'[INFO] torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} | visible gpus {torch.cuda.device_count()}')"
echo

# --standalone: single-node rendezvous, no master addr/port juggling.
torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} "${SCRIPT}"

echo
echo "finished: $(date -Is)"
echo "adapter : ${WORKDIR}/qwen3.5-4b-lora-diacritization/final_adapter"
cat "${WORKDIR}/eval_outputs/metrics_summary.csv" 2>/dev/null || true
