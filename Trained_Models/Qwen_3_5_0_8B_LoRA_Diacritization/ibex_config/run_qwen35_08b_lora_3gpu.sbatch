#!/bin/bash
# Transcribed from ~/ibex_diacritics_qwen35_08b/run_qwen35_08b_lora_3gpu.sbatch (job 50791419),
# which was still PENDING when this run was executed on Modal instead. The four `export` lines
# marked CONTRACT PATCH below are NOT in the original -- they are the settings that were
# unreachable from the environment until this package's copy of the training script made them
# env-settable. Without them an Ibex run of this script silently violates the fairness contract.
#SBATCH --job-name=qwen08-lora-r16
#SBATCH --output=/home/alnafjst/ibex_diacritics_qwen35_08b/logs/%x_%j.out
#SBATCH --error=/home/alnafjst/ibex_diacritics_qwen35_08b/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
# No --partition line on purpose: Ibex refuses an explicit gpu4/gpu24 and routes this
# automatically to gpu,gpu24,gpu72. 12h keeps all three eligible (gpu24 caps at 24h).

set -euo pipefail

WORKDIR=/home/alnafjst/ibex_diacritics_qwen35_08b
ENV_DIR=/home/alnafjst/envs/diacritics_qwen35_08b_py312
CUDA_SHIM=/home/alnafjst/cuda_shim
SCRIPT="$WORKDIR/src/train_lora_qwen35_08b.py"
RUN_NAME=qwen3.5-0.8b-lora-ibex-r16-full-epoch-v1
NGPU=3

module purge
module load machine_learning/2026.02
source "$ENV_DIR/bin/activate"

mkdir -p "$WORKDIR/logs" "$WORKDIR/runs" "$WORKDIR/hf_cache" "$WORKDIR/eval_outputs" "$WORKDIR/hf_home"
cd "$WORKDIR"

export HF_HOME="$WORKDIR/hf_home"
export HF_HUB_CACHE="$WORKDIR/hf_cache/hub"
export HF_CACHE_DIR="$WORKDIR/hf_cache"
export OUTPUT_ROOT="$WORKDIR/runs/$RUN_NAME"
export EVAL_ROOT="$WORKDIR/eval_outputs/$RUN_NAME"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Sadeed_Tashkeela is gated:manual. train_lora_qwen35_08b.py's hf_login() only prints a WARNING
# when the token is missing and then 401s during data loading -- i.e. AFTER the GPUs are held.
# The token also lives at $HF_HOME/token, which is what makes an already-queued job work.
[ -f "$WORKDIR/.hf_secret" ] && source "$WORKDIR/.hf_secret"

# causal_conv1d was compiled against a hand-built CUDA prefix ($CUDA_SHIM) because the
# machine_learning module ships nvcc but no CUDA headers and no unversioned lib*.so symlinks.
# Its RPATH already resolves every dependency, so NOTHING needs to be added to LD_LIBRARY_PATH.
#
# Do NOT put $CUDA_SHIM/lib64 on LD_LIBRARY_PATH. It symlinks the module's whole CUDA stack,
# including libcublas.so.12 (12.9.1.4), while torch here is cu128 and ships its own cuBLAS.
# Fronting torch's bundled copy makes RoPE's batched sgemm die with
#     RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE ... cublasSgemmStridedBatched
# on the FIRST forward pass. Verified on an A100 (job 50792430 failed this way; 50793142 A/B).
# Do still keep $CUDA_SHIM on disk -- deleting it breaks the RPATH and the kernels stop loading.
export NCCL_DEBUG=WARN

# Fair-comparison recipe from LoRA_Fine_Tuning_Config_Comparison.md.
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4
export PER_DEVICE_EVAL_BATCH_SIZE=8
export LORA_R=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.05
export GRADIENT_CHECKPOINTING=1
export EVAL_STEPS=200
export SAVE_TOTAL_LIMIT=2
export NUM_TRAIN_EPOCHS=1
export MAX_STEPS=-1
export LEARNING_RATE=2e-4
export OPTIM=adamw_torch
export TRAIN_DATASET_REVISION=c10bcbb3b50dc96551f62c472389de666a8c1c4e
export BENCHMARK_DATASET_REVISION=aa311213e44e4cab6cc3f2848daacd753adc1ce1

# ---- CONTRACT PATCH: not in the original sbatch -------------------------------------------
# The original exported EVAL_STEPS=200 above, but the shipped script never READ it -- it scaled
# the interval to run length via N_EVALS=20, so an Ibex run would have evaluated every 543 steps.
# The other three were module-level constants with no environment hook at all. All four are now
# env-settable in Train_LoRA_Qwen_3_5_0_8B_Diacritization.py, defaults unchanged.
export WARMUP_RATIO=0.05            # contract: 5% of total steps. Script shipped 0.03.
export LOAD_BEST_MODEL_AT_END=0     # contract: keep the LAST checkpoint. Script forced True.
export IN_TRAINING_EVAL_SUBSET=500  # contract: 500 rows of the test split. Script shipped 1000.
# EVAL_STEPS=200 above is now actually honoured by schedule_for().
# -------------------------------------------------------------------------------------------

printf 'job=%s host=%s started=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Fail in ~20s on a broken env instead of 65s in with 3 A100s held (job 50776872).
bash "$WORKDIR/preflight.sh"

python -c "import torch,transformers,datasets,peft,accelerate; print('torch',torch.__version__,'transformers',transformers.__version__,'datasets',datasets.__version__,'peft',peft.__version__,'accelerate',accelerate.__version__)"

# NOTE: "$ENV_DIR/bin/python" -m torch.distributed.run, NOT bare `torchrun`. The module's
# torchrun has the MODULE python in its shebang, so its workers lost the venv and could not
# import jiwer -- that is exactly how job 50776872 died.
"$ENV_DIR/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPU" "$SCRIPT" --train-only
python "$SCRIPT" --eval-only --model "$OUTPUT_ROOT/final_model" --splits benchmark --label final

printf 'finished=%s\n' "$(date -Is)"
