#!/bin/bash
#SBATCH --job-name=flan-t5-byt5-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@kaust.edu.sa

# Abdou/arabic-tashkeel-flan-t5-small, glonor/byt5-arabic-diacritization --
# zero-shot seq2seq, on the full 1,200-paragraph SadeedDiac-25 benchmark.
# Uses the CORRECTED scorer (Tested_Models_v2/Flan-T5-based_ByT5-based_test.ipynb).
#
#   sbatch run_flan_t5_byt5_test.sbatch
#
# --time is generous for two small seq2seq models; should finish well inside
# the limit. No mid-run checkpoint in this notebook.
#
# Executed via `jupyter nbconvert --execute`, not a plain `python` call --
# this notebook has no .py counterpart.
#
# NO OVERWRITE: unlike the other four test scripts this notebook has no
# to_csv() call at all -- its only output is the executed notebook's cell
# output (the printed result tables). The executed copy is written to
# executed_runs/<timestamp>_<job_id>.ipynb, a fresh filename every run, so
# nothing is ever overwritten. Model weights still cache to a SHARED,
# job-independent HF_HOME, so resubmissions don't re-download.

set -euo pipefail

WORKDIR="${HOME}/tested_models_v2/flan_t5_byt5"
ENV_DIR="${HOME}/envs/diac_clean"
NOTEBOOK="${WORKDIR}/Flan-T5-based_ByT5-based_test.ipynb"
STAMP="$(date +%Y%m%dT%H%M%S)_job${SLURM_JOB_ID:-manual}"
OUT_NOTEBOOK="executed_runs/${STAMP}.ipynb"

mkdir -p "${WORKDIR}/logs" "${WORKDIR}/executed_runs"
cd "${WORKDIR}"
[ -f "${NOTEBOOK}" ] || { echo "[FATAL] ${NOTEBOOK} not found -- copy it up before submitting." >&2; exit 1; }

module purge
module load machine_learning/2026.02
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
echo "[INFO] python: $(which python) ($(python --version 2>&1))"

python -c "import nbconvert" 2>/dev/null || pip install -q nbconvert ipykernel

export HF_HOME="${HOME}/hf_home"
export TOKENIZERS_PARALLELISM=false

if [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token")"
    echo "[INFO] HF token loaded from ${HOME}/.hf_token"
else
    echo "[WARN] ${HOME}/.hf_token not found -- proceeding unauthenticated (both models are ungated)."
fi

echo "=================================================="
echo "job     : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "workdir : ${WORKDIR}"
echo "output  : ${OUT_NOTEBOOK}"
echo "started : $(date -Is)"
echo "=================================================="
nvidia-smi
echo

srun jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "${OUT_NOTEBOOK}" \
    "${NOTEBOOK}"

echo
echo "finished: $(date -Is)"
echo "executed notebook (with output tables) : ${WORKDIR}/${OUT_NOTEBOOK}"
