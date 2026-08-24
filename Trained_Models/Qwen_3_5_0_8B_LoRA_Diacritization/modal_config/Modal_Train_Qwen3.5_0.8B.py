"""Modal runner for a fresh, full-dataset Qwen3.5-0.8B LoRA fine-tune.

The default job trains one complete epoch over the pinned Sadeed_Tashkeela split. There is no
default step cap or cost stop. Outputs use a new Volume namespace, so the earlier partial run is
preserved and cannot be resumed accidentally. Retries inside this namespace resume only complete
checkpoints from this new run.

Authentication is intentionally external: Modal credentials stay in the local Modal profile and
the Hugging Face token stays in the Modal Secret named ``huggingface`` under ``HF_TOKEN``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

import modal

APP_NAME = "diacritics-qwen35-08b-lora-r16-eb96"
PKG_DIR = "/pkg"          # this directory, mounted read-only into the container
SCRATCH = "/scratch"      # the Volume
RUN_NAME = os.environ.get("DIACRITICS_RUN_NAME", "qwen3.5-0.8b-lora-r16-eb96-contract-v1")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", RUN_NAME):
    raise ValueError("DIACRITICS_RUN_NAME must be a 1-100 character path-safe slug")
RUN_ROOT = f"{SCRATCH}/runs/{RUN_NAME}"

SCRIPT = "train_lora_qwen35_08b.py"

# ==================================================================================================
# THE FAIRNESS CONTRACT -- LoRA_Fine_Tuning_Config_Comparison.md, "Identical across both models".
#
# This package exists to run Qwen3.5-0.8B under that contract, so the contract's values are the
# DEFAULTS here rather than flags someone has to remember. The sibling Train-Qwen-0.8B-LoRA/
# package keeps its own r32 / effective-batch-64 recipe; nothing is shared between them.
#
# Three of these were NOT reachable in the upstream script and were made env-settable in this
# package's copy of train_lora_qwen35_08b.py (defaults there unchanged, so the r32 run still
# reproduces):
#   WARMUP_RATIO           script hardcoded 0.03; the contract pins 0.05.
#   LOAD_BEST_MODEL_AT_END script forced True; the contract requires the LAST checkpoint.
#   EVAL_STEPS             script scaled the interval to run length and never read EVAL_STEPS,
#                          so the Ibex sbatch's `export EVAL_STEPS=200` was a no-op.
# ==================================================================================================
EFFECTIVE_BATCH = int(os.environ.get("EFFECTIVE_BATCH", "96"))

CONTRACT_ENV = {
    "LORA_ALPHA": "32",                 # alpha/r = 2
    "LORA_DROPOUT": "0.05",
    "LEARNING_RATE": "2e-4",            # cosine; LR_SCHEDULER_TYPE is already "cosine"
    "WARMUP_RATIO": "0.05",             # 5% of total steps, as a RATIO not a step count
    "OPTIM": "adamw_torch",             # not adamw_torch_fused -- the contract names this one
    "GRADIENT_CHECKPOINTING": "1",      # forced ON for every model, to remove the throughput
                                        # confound; see the contract's own note on this
    "EVAL_STEPS": "200",                # logging 20 / eval 200 / save 200
    "SAVE_TOTAL_LIMIT": "2",
    "LOAD_BEST_MODEL_AT_END": "0",      # keep the LAST checkpoint, not the best eval_loss
    "IN_TRAINING_EVAL_SUBSET": "500",   # contract: 500 rows of the test split
    "NUM_TRAIN_EPOCHS": "1",
    "MAX_STEPS": "-1",                  # epoch count controls the run, not a step cap
}
CONTRACT_RANK = 16
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")

app = modal.App(APP_NAME)

# Persistent scratch: HF cache (1.77GB of weights + the corpus + the tokenized Arrow shards),
# checkpoints, eval outputs. Shared with the 4B and full-FT packages on purpose -- same cache,
# same datasets, same pinned revisions. Modal gives 1 TiB of Volume storage free per month, then
# $0.09/GiB-month.
#
# Checkpoints are cheap here on purpose. An adapter is ~110MB (vs ~9GB for a full-FT checkpoint),
# so save_total_limit=5 (train_lora_qwen35_08b.py's SAVE_TOTAL_LIMIT) keeps ~550MB resident, not
# ~45GB. Delete a finished run's checkpoints anyway once scored:
#     modal volume rm -r diacritics-scratch runs/<run-name>/checkpoint-XXXX
scratch = modal.Volume.from_name("diacritics-scratch", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface")  # must contain HF_TOKEN

# CUDA *devel* base (not runtime): causal-conv1d ships sdist only and compiles CUDA from source,
# so nvcc must be on PATH at image-build time.
#
# THE BASE IMAGE'S CUDA AND TORCH'S CUDA MUST MATCH, AND BOTH ARE PINNED ON PURPOSE.
# causal-conv1d's setup.py calls torch.utils.cpp_extension._check_cuda_version, which hard-fails
# when nvcc's version differs from the one torch was built against. The 4B build died once
# exactly there: the base was 12.8.1-devel while a bare `pip_install("torch")` resolved to a
# CUDA 13.0 wheel, giving
#     RuntimeError: The detected CUDA version (12.8) mismatches the version that was used to
#     compile PyTorch (13.0)
# after ~10 minutes of compiling. A bare "torch" tracks whatever CUDA PyPI defaults to THAT DAY,
# so the pairing below is written out on both sides -- image tag and wheel index -- rather than
# left to a default that can move underneath the build.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
    # build-essential, not just git: the CUDA image has no guaranteed g++, and the extension
    # below is compiled from source. See the CXX/CC pin further down for the other half of this.
    .apt_install("git", "build-essential")
    # Must precede the source-built extension below: causal-conv1d's setup.py imports torch.
    # index_url pins the CUDA build; the version pin keeps the pair reproducible.
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130")
    # `peft` IS used here (unlike the full-FT package this image was forked from, where it was
    # only a hash-stability placeholder) -- kept at the exact same version and the exact same
    # position in this list for the same reason it was there originally: changing this layer's
    # hash forces a fresh 10-30 minute causal-conv1d compile below.
    .pip_install(
        "transformers==5.14.1",
        "datasets==5.0.1",
        "accelerate==1.14.0",
        "peft==0.20.0",
        "liger-kernel==0.8.2",
        "jiwer==4.0.0",
        "pandas",
        "matplotlib",
        "tqdm",
        "hf_transfer",
    )
    # NOT optional. Qwen3.5-0.8B is a hybrid model: 18 of its 24 layers are Gated-DeltaNet
    # linear attention (config.json `layer_types`, full_attention_interval=4), only 6 are full
    # attention. Without these two packages, modeling_qwen3_5.py's `is_fast_path_available` is
    # False and those 18 layers silently fall back to `torch_chunk_gated_delta_rule`, a
    # pure-PyTorch chunked scan. It only logs `warning_once`, so the run looks healthy while
    # costing several times more GPU time. Installing them is the single highest-leverage cost
    # fix in this file.
    # causal-conv1d ships sdist only, so this compiles CUDA from source. Pin the arch list to the
    # GPUs we actually rent (9.0 = H100/H200, 10.0 = B200) instead of every arch nvcc knows, and
    # cap MAX_JOBS so nvcc doesn't OOM the builder.
    #
    # CXX/CC are NOT boilerplate -- without them this build fails with a compiler error that
    # names a compiler nothing here uses:
    #     RuntimeError: The current installed version of clang++ (0.0.0) is less than the
    #     minimum required version by CUDA 13.0 (7.0)
    # Modal's add_python= installs a python-build-standalone CPython, which was itself compiled
    # with clang, so its sysconfig records CXX=clang++ and setuptools inherits that. clang is not
    # installed in the nvidia/cuda image, so torch's shutil.which('clang++') returns None, reports
    # the version as 0.0.0, and then validates it against the CLANG bounds (min 7.0) instead of
    # the GCC ones -- while a perfectly good g++ sits unused on PATH. Setting CXX explicitly makes
    # distutils.customize_compiler override the sysconfig value. gcc 11 on ubuntu 22.04 is
    # comfortably inside CUDA 13.0's accepted gcc range of 6.0-16.0.
    .env({
        "MAX_JOBS": "8",
        "TORCH_CUDA_ARCH_LIST": "9.0;10.0",
        "CXX": "g++",
        "CC": "gcc",
    })
    .pip_install("causal-conv1d", extra_options="--no-build-isolation")
    .pip_install("flash-linear-attention")  # Triton-based, wheel available, installs fast
    .env(
        {
            # All three cache vars, not just HF_HOME. HF_HOME alone puts snapshot_download's
            # output in $HF_HOME/hub/, but the script passes cache_dir=$HF_CACHE_DIR straight
            # through to from_pretrained -- a different directory. Without all three the
            # warm_cache download is a cache MISS at training time and the weights are re-pulled
            # with the GPU sitting idle. HF_CACHE_DIR is also where datasets writes its .map()
            # fingerprint cache, which is what makes `prepare` save the GPU job any work.
            "HF_HOME": f"{SCRATCH}/hf_cache",
            "HF_HUB_CACHE": f"{SCRATCH}/hf_cache",
            "HF_CACHE_DIR": f"{SCRATCH}/hf_cache",
            # Persist Triton/Liger JIT artifacts across Modal containers. Without this, every
            # retry recompiles each padded sequence bucket on an H100 (roughly 25-120s apiece).
            "TRITON_CACHE_DIR": f"{SCRATCH}/triton_cache",
            "OUTPUT_ROOT": RUN_ROOT,
            "EVAL_ROOT": f"{RUN_ROOT}/eval_outputs",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",   # ~4x faster download
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            # Evaluation falls back to the ordinary Transformers causal-LM loss for Qwen3.5
            # and can leave large transient allocations behind. Expandable segments reduce
            # fragmentation between the fused training path and that unfused eval path.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # ignore= is not tidiness. Run the training script locally even once and its defaults create
    # ./hf_cache, ./qwen3.5-0.8b-lora-diacritization and ./eval_outputs right here -- and
    # ./hf_cache is ~1.8GB of weights, which would then be uploaded into the image on every
    # single deploy.
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path=PKG_DIR,
        ignore=[
            "**/__pycache__", "**/*.pyc",
            "**/hf_cache", "**/eval_outputs", "**/results",
            "**/checkpoint-*",
            "**/*-lora-diacritization",
        ],
    )
)

HOUR = 60 * 60
FINAL_MODEL_DIR = f"{RUN_ROOT}/final_model"

# One place, so a change cannot apply to three functions and miss the fourth.
CPU_SPEC = dict(cpu=8, memory=32 * 1024)
# Billed on max(requested, used) at $0.008/GiB/hr, so 96GiB would add ~$0.77/hr on top of the
# $3.95 GPU -- ~15%, for RAM this job never touches. The model is 1.77GB of weights streamed to
# the GPU, and datasets writes tokenized Arrow shards to disk incrementally rather than holding
# the corpus resident.
GPU_SPEC = dict(gpu="H100:1", **CPU_SPEC)


def _run(cmd: list[str], commit_every: int = 0, env_overrides: dict | None = None) -> None:
    """Shell out to the training script, streaming output, failing loudly.

    `commit_every` (seconds, 0 = off) commits the Volume periodically WHILE the subprocess runs.
    `env_overrides` layers extra env vars onto the subprocess -- used by `train` to pass
    MAX_TRAIN_COST_USD through without every caller having to know that's how the cost budget
    is threaded into the script.

    Without it, the only commit is the one in each function's `finally`, and that is not enough
    for a multi-hour job. Volume writes are buffered until commit, so an uncommitted checkpoint
    does not exist as far as the next container is concerned. The `finally` covers the ordinary
    failures — a crash in the script makes _run sys.exit, which unwinds through it — but not a
    hard kill: hitting Modal's function timeout, or the container being reclaimed, can take the
    process out without running it.

    That is exactly the case the resume machinery is built for. latest_checkpoint() ranks by step
    number and requires trainer_state.json specifically so an interrupted run can pick up from
    the newest COMPLETE checkpoint — but it can only ever see checkpoints that were committed.
    Uncommitted, a 4-hour run that is reclaimed at hour 3 resumes from nothing.

    10 minutes is well under the ~40-minute checkpoint interval at N_SAVES=20 over a full epoch,
    so every checkpoint gets committed shortly after it is written. Committing is cheap and
    incremental; it is not a full re-upload of the Volume.
    """
    print("+ " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["MODEL_ID"] = MODEL_ID
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.Popen(cmd, cwd=PKG_DIR, env=env)

    if not commit_every:
        returncode = proc.wait()
    else:
        last_commit = time.time()
        while True:
            try:
                returncode = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                if time.time() - last_commit >= commit_every:
                    try:
                        scratch.commit()
                        print(f"[modal] volume committed "
                              f"(+{(time.time() - last_commit) / 60:.0f} min)", flush=True)
                    except Exception as e:
                        # Never take the training run down over a commit failure — the next
                        # one, or the finally, may well succeed.
                        print(f"[modal] volume commit failed ({type(e).__name__}: {e}); "
                              f"continuing", flush=True)
                    last_commit = time.time()

    if returncode != 0:
        sys.exit(returncode)


# --------------------------------------------------------------------------------------------
# CPU-only steps. These cost ~$0.63/hr -- never do any of them inside a GPU container.
# --------------------------------------------------------------------------------------------


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=3 * HOUR, **CPU_SPEC)
def warm_cache():
    """Pull the model (1.77GB) and both datasets into the Volume's HF cache on a CPU box.

    Downloading inside the GPU container would burn GPU time at 7x the rate. This is also the
    cheapest possible check that HF_TOKEN is valid AND that Misraj/Sadeed_Tashkeela access has
    actually been approved -- it is `gated: manual` and a human at Misraj approves each request,
    which can take days. Failing here costs pennies; failing after a GPU is allocated does not.

    (Qwen/Qwen3.5-0.8B and -0.8B-Base are both ungated -- verified against the HF API -- so the
    only gate in this pipeline is the training corpus.)
    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    sys.path.insert(0, PKG_DIR)
    import train_lora_qwen35_08b as t

    try:
        path = snapshot_download(MODEL_ID, token=os.environ["HF_TOKEN"])
        print(f"model cached at {path}", flush=True)

        # Pinned revisions, same as the training run will request -- otherwise this warms the
        # wrong snapshot and the GPU job downloads all over again.
        load_dataset(t.TRAIN_DATASET, cache_dir=t.CACHE_DIR, revision=t.TRAIN_DATASET_REVISION)
        print(f"cached {t.TRAIN_DATASET} @ {t.TRAIN_DATASET_REVISION}", flush=True)
        load_dataset(t.BENCHMARK_DATASET, cache_dir=t.CACHE_DIR,
                     revision=t.BENCHMARK_DATASET_REVISION)
        print(f"cached {t.BENCHMARK_DATASET} @ {t.BENCHMARK_DATASET_REVISION}", flush=True)
    finally:
        # Commit even on failure: a partially-pulled snapshot is still worth keeping, since
        # snapshot_download resumes per-file rather than restarting the whole pull.
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=3 * HOUR, **CPU_SPEC)
def analyze():
    """Corpus composition by filename + measured token economics. No GPU, no training.

    Answers the two questions RUN_REPORT.md §9 leaves open, and answers them with measurements
    rather than estimates:

      - what is actually IN Sadeed_Tashkeela, by source file. The MSA deficit (11.07 vs CA 2.84)
        is created by training on a Classical-heavy corpus, and you cannot build a balanced rung
        without knowing the composition. Writes corpus_composition.csv.
      - how the token budget splits between prompt overhead, input and target. The 4B run spent
        19% of all compute re-encoding the same Arabic instruction 96,000 times.

    Run this BEFORE deciding whether the plain full-corpus rung is the right next run.
    """
    try:
        _run([sys.executable, SCRIPT, "--analyze"])
    finally:
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=4 * HOUR, **CPU_SPEC)
def prepare(train_size: int = 0, include_filename: str = "", exclude_filename: str = ""):
    """Tokenize the full corpus into the Volume's datasets cache. CPU only.

    Worth its own phase for two reasons:

      1. It is 15-25 minutes of pure CPU work that would otherwise happen with an H100 idle --
         $0.25 here versus ~$1.70 there.
      2. It is a full pass over all 1,042,698 rows, which is the ONLY thing that finds the kind
         of defect RUN_REPORT.md §5.3 describes: exactly one null row, 0.0001% of the corpus,
         which killed a real training run 12 minutes in with the GPU already rented. A 200-row
         smoke test cannot find it. Finding it here costs cents.

    Pass the same --include/--exclude-filename you intend to train with, or the fingerprint
    differs and the GPU job re-tokenizes anyway (harmlessly -- it just loses the saving).
    """
    cmd = [sys.executable, SCRIPT, "--prepare-only"]
    if train_size:
        cmd += ["--train-size", str(train_size)]
    if include_filename:
        cmd += ["--include-filename", include_filename]
    if exclude_filename:
        cmd += ["--exclude-filename", exclude_filename]
    try:
        _run(cmd)
    finally:
        scratch.commit()


# --------------------------------------------------------------------------------------------
# GPU steps.
#
# One H100-80 throughout, and there is no case for more. The unfused rank-64 recipe OOMed at
# microbatch 16 because Qwen's 248,320-way logits dominated memory. Liger fused linear CE removes
# that tensor; the measured optimized recipe fits microbatch 32, accumulation 2, rank 32 and runs
# at 0.81 s/optimizer-step after the persistent Triton cache is warm. Sharding this across GPUs
# would add all-gather traffic to a job that already fits on one.
# --------------------------------------------------------------------------------------------


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=1 * HOUR, **GPU_SPEC)
def smoke_test(rank: int = CONTRACT_RANK, batch_size: int = 32,
               gradient_accumulation_steps: int = 3,
               eval_batch_size: int = 8,
               use_liger: bool = True, torch_compile: bool = False,
               pad_to_multiple_of: int = 64):
    """12,000 rows, 60 steps, no checkpoints; defaults to the Liger speed experiment.

    IT IS THE GATE, in two independent ways.

    CORRECTNESS. It exercises every part that can be silently wrong: the linear-attention
    fast-path check, the thinking-disabled prompt assertion, the text-only model class
    assertion, the LoRA target-count assert (must print exactly 150), the fp32-adapter-dtype
    assertion, and the label masking. The gate is the loss: it must be finite, non-zero and
    trending down. A loss near 0.0 at step 1 means the masking is inverted -- stop. The script
    raises on that itself, but read the printed losses anyway.

    COST. It prints a [THROUGHPUT] line measured over the last 40 of its 60 steps, discarding
    the first 20. That discard is not caution, it is necessary: step 1 carries ~120s of Triton
    JIT for the Gated-DeltaNet kernels, and the length-grouped sampler sorts each 50-step
    megabatch longest-first so early steps are the slowest in their cycle. RUN_REPORT.md §5.6
    records two bad cost estimates made by extrapolating from the first few steps. That
    [THROUGHPUT] line is the only estimate in this package you should budget from.

        rank    Defaults to 32, the measured speed/quality compromise. Pass 64 to reproduce the
                earlier high-capacity recipe or 16 for the fastest/smallest adapter probe.

        batch_size / gradient_accumulation_steps
                Defaults to the measured 32 x 2, preserving effective batch 64.
        use_liger
                Enables only Liger fused linear cross-entropy. Other Liger patches stay off.
        torch_compile
                Separate opt-in stage after Liger succeeds; compilation can add a long first-step
                warmup and may graph-break around Qwen's custom linear-attention kernels.
        pad_to_multiple_of
                Bounds Liger's shape-specialized JIT variants. 64 means at most 16 padded
                sequence lengths under the 1,024-token cap; 0 restores exact dynamic padding.
    """
    if batch_size <= 0 or gradient_accumulation_steps <= 0 or eval_batch_size <= 0:
        raise ValueError("batch sizes and gradient_accumulation_steps must be positive")
    if batch_size * gradient_accumulation_steps != EFFECTIVE_BATCH:
        raise ValueError(f"speed comparisons must preserve effective batch "
                         f"{EFFECTIVE_BATCH} (got {batch_size} x "
                         f"{gradient_accumulation_steps})")
    if pad_to_multiple_of < 0:
        raise ValueError("pad_to_multiple_of must be >= 0")
    cmd = [sys.executable, SCRIPT, "--smoke-test"]
    if rank:
        cmd += ["--rank", str(rank)]
    env = {
        "PER_DEVICE_TRAIN_BATCH_SIZE": str(batch_size),
        "GRADIENT_ACCUMULATION_STEPS": str(gradient_accumulation_steps),
        "PER_DEVICE_EVAL_BATCH_SIZE": str(eval_batch_size),
        "SMOKE_RUN_EVAL": "1",
        "USE_LIGER_KERNEL": "1" if use_liger else "0",
        "TORCH_COMPILE": "1" if torch_compile else "0",
        "PAD_TO_MULTIPLE_OF": str(pad_to_multiple_of),
    }
    env.update(CONTRACT_ENV)   # the contract wins over any per-call convenience setting
    print(f"[modal] contract={CONTRACT_ENV}", flush=True)
    print(f"[modal] smoke train_batch={batch_size} eval_batch={eval_batch_size} "
          f"accumulation={gradient_accumulation_steps} "
          f"rank={rank or 64} liger={use_liger} compile={torch_compile} "
          f"pad_multiple={pad_to_multiple_of}", flush=True)
    try:
        _run(cmd, env_overrides=env)
    finally:
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=24 * HOUR, **GPU_SPEC)
def train(max_steps: int = 0, max_cost_usd: float = 0.0, rank: int = CONTRACT_RANK,
          train_size: int = 0,
          include_filename: str = "", exclude_filename: str = "",
          batch_size: int = 32, gradient_accumulation_steps: int = 3,
          eval_batch_size: int = 8,
          use_liger: bool = True, torch_compile: bool = False,
          pad_to_multiple_of: int = 64):
    """LoRA fine-tune for one full epoch by default. Score separately with `evaluate`.

    Split from evaluation on purpose: Modal caps a function at 24h and containers can be
    preempted, and a train followed by an eval in one function means a preemption during the eval
    loses the training result too. The script resumes from the newest COMPLETE checkpoint on the
    Volume (ranked by step number and requiring trainer_state.json, not by mtime -- a job
    preempted mid-write leaves a partial checkpoint holding the newest mtime, and resuming from
    that dies on a truncated shard).

        max_cost_usd    0 disables the optional wall-clock cost stop (default). A positive value
                        deliberately enables it for this container.
        max_steps       0 uses the script default: -1, meaning NUM_TRAIN_EPOCHS=1 controls the
                        run. A positive value deliberately makes a partial run.

        rank            Defaults to 32, the smoke-tested speed/quality compromise. Pass 64 to
                        reproduce the earlier partial run's adapter capacity.
        train_size      0 = the whole split.
        include/exclude_filename
                        regex over the corpus `filename` column, for a domain-balanced rung.
                        Run `analyze` first -- do not guess a pattern.
        batch_size / gradient_accumulation_steps
                        Defaults to the fastest smoke-tested geometry while preserving effective
                        batch 64. Change only after a successful smoke at the same values.
        use_liger       Enables fused linear cross-entropy, required for batch sizes above 8.
        torch_compile   Optional; leave off unless its separate smoke is faster and stable.
        pad_to_multiple_of
                        Bounds Triton's shape variants; 64 is the tested default.
    """
    if batch_size <= 0 or gradient_accumulation_steps <= 0 or eval_batch_size <= 0:
        raise ValueError("batch sizes and gradient_accumulation_steps must be positive")
    if batch_size * gradient_accumulation_steps != EFFECTIVE_BATCH:
        raise ValueError(f"training must preserve effective batch {EFFECTIVE_BATCH} "
                         f"(got {batch_size} x {gradient_accumulation_steps}) -- it is the "
                         f"quantity the optimizer steps on and the whole basis of the "
                         f"comparison")
    if pad_to_multiple_of < 0:
        raise ValueError("pad_to_multiple_of must be >= 0")
    cmd = [sys.executable, SCRIPT, "--train-only"]
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    if rank:
        cmd += ["--rank", str(rank)]
    if train_size:
        cmd += ["--train-size", str(train_size)]
    if include_filename:
        cmd += ["--include-filename", include_filename]
    if exclude_filename:
        cmd += ["--exclude-filename", exclude_filename]
    # MAX_TRAIN_COST_USD=0 from the CLI means "disable", matching the script's own convention.
    env = {
        "MAX_TRAIN_COST_USD": str(max_cost_usd),
        "PER_DEVICE_TRAIN_BATCH_SIZE": str(batch_size),
        "GRADIENT_ACCUMULATION_STEPS": str(gradient_accumulation_steps),
        "PER_DEVICE_EVAL_BATCH_SIZE": str(eval_batch_size),
        "USE_LIGER_KERNEL": "1" if use_liger else "0",
        "TORCH_COMPILE": "1" if torch_compile else "0",
        "PAD_TO_MULTIPLE_OF": str(pad_to_multiple_of),
    }
    env.update(CONTRACT_ENV)   # the contract wins over any per-call convenience setting
    print(f"[modal] contract={CONTRACT_ENV}", flush=True)
    print(f"[modal] run={RUN_NAME} full_dataset={train_size == 0} max_steps="
          f"{'epoch default' if max_steps == 0 else max_steps} max_cost_usd={max_cost_usd} "
          f"train_batch={batch_size} eval_batch={eval_batch_size} "
          f"accumulation={gradient_accumulation_steps} rank={rank or 64} "
          f"liger={use_liger} compile={torch_compile} pad_multiple={pad_to_multiple_of}",
          flush=True)
    try:
        # Commit every 10 min so checkpoints survive a hard kill, not just a clean failure.
        _run(cmd, commit_every=10 * 60, env_overrides=env)
    finally:
        scratch.commit()  # persist checkpoints even if the run dies or is preempted


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=12 * HOUR, **GPU_SPEC)
def evaluate(model: str = FINAL_MODEL_DIR, label: str = "lora",
             splits: str = "benchmark"):
    """Generation + DER/WER on train / test / SadeedDiac-25, into eval_outputs/.

    `model` is an ADAPTER DIRECTORY (or "none" for the zero-shot base) -- load_model_for_eval()
    reloads the frozen bf16 base and merges the adapter before generating. `splits` defaults to
    "benchmark" only here (not "train,test,benchmark" like the full-FT package): benchmark alone
    (1,200 rows) is the cheap way to get the one number that goes in the results table, and this
    package's budget does not comfortably cover scoring the 500-row train sample and the full
    2,485-row test split too. Pass --splits train,test,benchmark explicitly if the budget allows.

    The benchmark is generated ONCE and scored three ways -- CA, MSA and pooled -- plus a macro
    mean, and each of those by BOTH scorers. Use the macro mean, not the pooled number: pooling
    is diacritic-weighted and CA paragraphs are longer, so it flatters the result by
    over-weighting the easy domain.

    Which column to read:
        DER_noce_corr  -> corrected scorer. CITE THIS. Compares to the full-FT run's 2.58 and
                          the 4B LoRA run's 8.48.
        DER_noce       -> frozen 4B-identical scorer. Kept for diffing only; do not cite it.
    They are the same predictions under two scorers, not two runs. Never mix them.

    Re-invocable: predictions are checkpointed to CSV every ~200 rows, so a timeout or preemption
    picks up where it left off instead of regenerating.

    To score the ZERO-SHOT baseline through this identical pipeline (same prompt, same
    enable_thinking=False, same NFC scoring) -- the only honest comparison for the fine-tune:

        modal run modal_app.py::evaluate --model none --label base --splits benchmark

    `label` MUST differ between runs. It namespaces the prediction checkpoints, and those
    checkpoints are what a re-invocation resumes from: reuse "lora" for the base model and it
    will happily "resume" the fine-tuned model's cached predictions and report them as the
    baseline -- indistinguishable from "fine-tuning changed nothing". RUN_REPORT.md §5.5 (in
    ../Train-Qwen) caught that one before it fired.
    """
    try:
        # Same reasoning as `train`: run_inference checkpoints predictions to CSV every ~200
        # rows, and those are what a re-invocation resumes from — but only once committed.
        _run([sys.executable, SCRIPT, "--eval-only", "--model", model,
              "--label", label, "--splits", splits], commit_every=10 * 60)
    finally:
        scratch.commit()


@app.local_entrypoint()
def main():
    print(__doc__)
