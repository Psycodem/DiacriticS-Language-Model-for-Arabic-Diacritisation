#!/bin/bash
#SBATCH --job-name=fanar-model-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@kaust.edu.sa

# Fanar-1-9B-Instruct, zero-shot, 4-bit quantized, on SadeedDiac-25 (MSA + CA).
# Uses the CORRECTED scorer (Tested_Models_v2/Fanar_Model_test.py).
#
#   sbatch run_fanar_model_test.sbatch
#
# --time is an ESTIMATE (1,200 generations on one 9B 4-bit model, ~1-2 h
# realistic + model load) -- not measured, unlike the LoRA training jobs'
# smoke-tested limits. If it's tight, resubmit; there is no mid-run
# checkpoint in this script.
#
# NO OVERWRITE: the script writes large_models_results.csv to a FIXED path
# (needed so ./model_cache is reused across resubmissions instead of
# re-downloading the 9B model every time). To protect against that fixed
# path being clobbered by a later run, any existing large_models_results.csv
# is copied into results_archive/ with a timestamp BEFORE this run starts.

set -euo pipefail

WORKDIR="${HOME}/tested_models_v2/fanar"
ENV_DIR="${HOME}/envs/diac_clean"
SCRIPT="${WORKDIR}/Fanar_Model_test.py"
RESULT_CSV="large_models_results.csv"

mkdir -p "${WORKDIR}/logs" "${WORKDIR}/results_archive"
cd "${WORKDIR}"
[ -f "${SCRIPT}" ] || { echo "[FATAL] ${SCRIPT} not found -- copy it up before submitting." >&2; exit 1; }

if [ -f "${RESULT_CSV}" ]; then
    STAMP="$(date +%Y%m%dT%H%M%S)_job${SLURM_JOB_ID:-manual}"
    cp "${RESULT_CSV}" "results_archive/${STAMP}_${RESULT_CSV}"
    echo "[INFO] archived previous ${RESULT_CSV} -> results_archive/${STAMP}_${RESULT_CSV}"
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
    echo "[WARN] ${HOME}/.hf_token not found -- proceeding unauthenticated."
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
