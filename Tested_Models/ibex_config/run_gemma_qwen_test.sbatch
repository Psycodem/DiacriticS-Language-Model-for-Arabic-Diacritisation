#!/bin/bash
#SBATCH --job-name=gemma-qwen-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@kaust.edu.sa

# Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, google/gemma-4-E4B-it -- zero-shot, on the
# full 1,200-paragraph SadeedDiac-25 benchmark, one model after another.
# Uses the CORRECTED scorer (Tested_Models_v2/Gemma_Qwen_test.py).
#
#   sbatch run_gemma_qwen_test.sbatch
#
# --time is an ESTIMATE (3 models x 1,200 generations, up to 9B params) --
# not measured. This is the heaviest of the five test scripts; consider
# running it last / with the longest queue slot.
#
# google/gemma-4-E4B-it is GATED -- accept its licence on the model page with
# the account behind ~/.hf_token, or that one model's checkpoint fails.
#
# RESUME: the script checkpoints per-model predictions to
# sadeed_outputs/<model>__preds.csv every 50 paragraphs and resumes from them
# automatically. That ONLY works if WORKDIR stays the same across
# resubmissions (fixed below), so a killed job is resumed by just
# resubmitting this file -- do not job-ID-namespace WORKDIR.
#
# NO OVERWRITE: sadeed_outputs/results.csv is a FIXED path (needed for the
# resume behaviour above and so hf_cache/ is reused instead of re-downloading
# gemma-4-E4B-it and Qwen3.5-9B every run). Any existing results.csv is
# copied into results_archive/ with a timestamp BEFORE this run starts, so a
# later resubmission never destroys a completed run's numbers. The per-model
# prediction checkpoints are NOT touched, so resume still works.

set -euo pipefail

WORKDIR="${HOME}/tested_models_v2/gemma_qwen"
ENV_DIR="${HOME}/envs/diac_clean"
SCRIPT="${WORKDIR}/Gemma_Qwen_test.py"
RESULT_CSV="sadeed_outputs/results.csv"

mkdir -p "${WORKDIR}/logs" "${WORKDIR}/results_archive" "${WORKDIR}/sadeed_outputs"
cd "${WORKDIR}"
[ -f "${SCRIPT}" ] || { echo "[FATAL] ${SCRIPT} not found -- copy it up before submitting." >&2; exit 1; }

if [ -f "${RESULT_CSV}" ]; then
    STAMP="$(date +%Y%m%dT%H%M%S)_job${SLURM_JOB_ID:-manual}"
    cp "${RESULT_CSV}" "results_archive/${STAMP}_results.csv"
    echo "[INFO] archived previous ${RESULT_CSV} -> results_archive/${STAMP}_results.csv"
fi

module purge
module load machine_learning/2026.02
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
echo "[INFO] python: $(which python) ($(python --version 2>&1))"

export HF_HOME="${HOME}/hf_home"
export TOKENIZERS_PARALLELISM=false

if [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token")"
    echo "[INFO] HF token loaded from ${HOME}/.hf_token"
else
    echo "[WARN] ${HOME}/.hf_token not found -- gemma-4-E4B-it is gated and will fail to load."
fi

echo "=================================================="
echo "job     : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "workdir : ${WORKDIR}"
echo "started : $(date -Is)"
echo "=================================================="
nvidia-smi
echo

srun python "${SCRIPT}"

echo
echo "finished: $(date -Is)"
echo "results : ${WORKDIR}/${RESULT_CSV}"
echo "archive : ${WORKDIR}/results_archive/"
