#!/bin/bash
#SBATCH --job-name=diac-eval
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL

# Score a model on Train / Test / SadeedDiac-25 — the three rows the sheet's
# Fine-Tuning Models tab wants per run.
#
#   # fine-tuned adapter, all three splits
#   sbatch eval.sbatch google/gemma-4-E4B-it adapter gemma-4-E4B-it-QLoRA QLoRA
#
#   # plain baseline, benchmark only
#   sbatch eval.sbatch Qwen/Qwen3.5-4B base "" "" sadeed
#
# Chain it behind training so it starts the moment the adapter exists:
#   JID=$(sbatch --parsable --job-name=gemma train.sbatch google/gemma-4-E4B-it)
#   sbatch --dependency=afterok:$JID eval.sbatch google/gemma-4-E4B-it adapter gemma-4-E4B-it-QLoRA QLoRA
#
# Generation is checkpointed every 50 paragraphs, so hitting the time limit
# costs at most one chunk — just resubmit and it resumes.

set -euo pipefail

MODEL="${1:?usage: sbatch eval.sbatch <hf-model-id> <adapter|base> [label] [method] [splits]}"
MODE="${2:-base}"
LABEL="${3:-}"
METHOD="${4:-}"
SPLITS="${5:-train,test,sadeed}"

SCRATCH="/ibex/scratch/${USER}"
ENV_DIR="${SCRATCH}/envs/diacritics"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG="${MODEL//\//__}"
OUT_DIR="${SCRATCH}/sadeed_outputs"

mkdir -p logs "${OUT_DIR}"

module purge
module load cuda/12.2 || module load cuda/11.8 || true
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

export HF_CACHE_DIR="${SCRATCH}/hf_cache"
export HF_HOME="${SCRATCH}/hf_home"
export TOKENIZERS_PARALLELISM=false
[ -f "${HOME}/.hf_token" ] && export HF_TOKEN="$(cat "${HOME}/.hf_token")"

ARGS=(--model "${MODEL}" --out-dir "${OUT_DIR}" --cache-dir "${HF_CACHE_DIR}"
      --splits "${SPLITS}")

if [ "${MODE}" = "adapter" ]; then
    ADAPTER="${SCRATCH}/runs/${SLUG}/final_adapter"
    if [ ! -d "${ADAPTER}" ]; then
        echo "FATAL: no adapter at ${ADAPTER} — did training finish?" >&2
        exit 1
    fi
    ARGS+=(--adapter "${ADAPTER}")
    ARGS+=(--label "${LABEL:-${MODEL}-QLoRA}")
elif [ -n "${LABEL}" ]; then
    ARGS+=(--label "${LABEL}")
fi

[ -n "${METHOD}" ] && ARGS+=(--method "${METHOD}")

echo "=============================================="
echo "job    : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "model  : ${MODEL}  mode=${MODE}  splits=${SPLITS}  method=${METHOD:-auto}"
echo "started: $(date -Is)"
echo "=============================================="
nvidia-smi

srun python "${PROJECT}/eval_sadeed.py" "${ARGS[@]}"

echo "finished: $(date -Is)"
echo "results : ${OUT_DIR}/*__results.csv"
