#!/bin/bash
#SBATCH --job-name=large-models-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@kaust.edu.sa

# CohereLabs/aya-expanse-8b, moonshotai/Moonlight-16B-A3B-Instruct -- zero-shot,
# on the full 1,200-paragraph SadeedDiac-25 benchmark.
# Uses the CORRECTED scorer (Tested_Models_v2/Large_Models_test.py).
#
#   sbatch run_large_models_test.sbatch
#
# --time is an ESTIMATE (2 models, one a 16B MoE, x 1,200 generations) --
# not measured. There is no mid-run checkpoint in this script; a kill loses
# whatever model was in progress.
#
# KNOWN RISK, not something this file fixes: the script pins
# transformers==4.48.2 via its own pip-install-at-import step, which may
# conflict with the diac_clean venv's newer transformers. Watch the first
# few minutes of the log for a pip resolution error.
#
# aya-expanse-8b is GATED (Cohere) -- accept its licence with the account
# behind ~/.hf_token, or it lands in the script's own skip log rather than
# crashing the whole run.
#
# NO OVERWRITE: instruct_models_results.csv is a FIXED path (needed so
# ./model_cache is reused across resubmissions instead of re-downloading an
# 8B and a 16B model every time). Any existing results CSV is copied into
# results_archive/ with a timestamp BEFORE this run starts.

set -euo pipefail

WORKDIR="${HOME}/tested_models_v2/large_models"
ENV_DIR="${HOME}/envs/diac_clean"
SCRIPT="${WORKDIR}/Large_Models_test.py"
RESULT_CSV="instruct_models_results.csv"

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
    echo "[WARN] ${HOME}/.hf_token not found -- aya-expanse-8b is gated and will be skipped."
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
