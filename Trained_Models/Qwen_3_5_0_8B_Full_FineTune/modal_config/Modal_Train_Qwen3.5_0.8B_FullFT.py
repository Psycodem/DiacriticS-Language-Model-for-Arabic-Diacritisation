"""
Modal runner for the Qwen3.5-0.8B FULL fine-tune.

Thin wrapper: it does NOT reimplement anything. Each Modal function sets up the environment
train_full_ft_qwen35_08b.py expects (HF cache + output dirs on a persistent Volume, $HF_TOKEN)
and shells out to that one script with a phase flag.

The image recipe is copied from ../Train-Qwen/modal_app.py DELIBERATELY BYTE-FOR-BYTE through
the causal-conv1d layer, including the `peft` pin this pipeline does not use. Modal caches image
layers by definition hash: keep those lines identical and the 10-30 minute CUDA compile of
causal-conv1d is a cache hit from the 4B app instead of a fresh build. Change one character in
the pip list above it and you pay for that compile again. The comments on those layers are kept
too — they encode three separate build failures that each cost ~10 minutes to diagnose.

One-time setup:
    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_...   # needs APPROVED Sadeed_Tashkeela access

Run order — cheap things first, and every one of them earns its place:
    modal run modal_app.py::warm_cache                 # CPU,  ~15 min, pulls 1.77GB + datasets
    modal run modal_app.py::analyze                    # CPU,  ~10 min, corpus + token economics
    modal run modal_app.py::prepare                    # CPU,  ~25 min, tokenizes the full corpus
    modal run modal_app.py::smoke_test                 # 1xH100, ~10 min -- THE GATE
    modal run --detach modal_app.py::train             # 1xH100, ~3-5 h
    modal run --detach modal_app.py::evaluate --label ft
    modal run --detach modal_app.py::evaluate --model none --label base --splits benchmark

Everything lands on the `diacritics-scratch` Volume, which survives between runs and is SHARED
with the 4B LoRA package -- so its HF cache and its tokenized-dataset cache are reused, and
outputs are namespaced under runs/qwen3.5-0.8b-fullft/ so the two cannot collide:
    modal volume ls  diacritics-scratch runs/qwen3.5-0.8b-fullft
    modal volume get diacritics-scratch runs/qwen3.5-0.8b-fullft/eval_outputs ./results

------------------------------------------------------------------------------------------------
WHAT THIS COSTS, BEFORE YOU START IT
------------------------------------------------------------------------------------------------
Rates (modal.com/pricing, as recorded in the 4B package on 2026-08-13): H100 SXM5 $0.001097/s =
$3.95/hr, physical core $0.0000131/s = $0.047/hr, memory $0.00000222/s = $0.008/GiB/hr. Modal
bills max(requested, used), so the decorators below ARE the bill:

    3.95 + 8(0.047) + 32(0.008) = ~$4.58 / hr        <- every GPU function here
    0    + 8(0.047) + 32(0.008) = ~$0.63 / hr        <- warm_cache, analyze, prepare

  step          wall clock       cost
  ------------- ---------------- -----------
  image build   0 (cache hit)    $0        (or 10-30 min / ~$0.50 if the 4B image is gone)
  warm_cache    ~15 min          ~$0.15
  analyze       ~10 min          ~$0.10
  prepare       ~25 min          ~$0.25
  smoke_test    ~10 min          ~$0.80
  train         3-5 h  + ~40 min $14-26     <- the +40 min is 20 checkpoint writes of ~9GB
  evaluate ft   0.5-1 h          $2-5
  evaluate base 0.5 h            ~$2
  ------------- ---------------- -----------
  FULL RUN                       $19-35, most likely ~$27

Compare: the 4B LoRA run cost ~$11 and covered 9.2% of one epoch. This is ~2.3x the money for
~11x the data and 35x the trainable parameters.

WHERE THE TRAIN ESTIMATE COMES FROM, AND WHY YOU SHOULD NOT TRUST IT YET. The 4B LoRA run
measured 6,624 tok/s on one H100 at ~16% MFU with gradient checkpointing on. This model is
~751M text params against 4.23B, and gradient checkpointing is off, so FLOPs/token drop ~7.5x;
MFU will fall too (hidden 1024 vs 2560 makes every matmul smaller and more launch-bound), so
call the net ~4-5x => ~30k tok/s. A full epoch is ~384M tokens => ~3.5 h.

That is arithmetic, not a measurement. RUN_REPORT.md §5.6 records two premature cost estimates
made exactly this way. `smoke_test` runs 60 steps and prints a [THROUGHPUT] line computed from
the last 40 of them -- past the ~120s Triton JIT and across a full 50-step length-grouped
megabatch cycle. THAT number is the estimate. Budget from it:

    hours = total_steps * s_per_step / 3600      # smoke_test prints both
    cost  = hours * 4.58

Modal caps a function at 24h. At ~3.5h that is not a concern, but `train` resumes from the
newest COMPLETE checkpoint on the Volume anyway if it is ever interrupted.

`evaluate` generates for 500 train + 2,485 test + 1,200 benchmark rows ~= 4,185 paragraphs. It
checkpoints predictions to CSV as it goes, so re-invoking after a timeout resumes rather than
regenerating.

To train a different base model, set MODEL_ID -- both this file and the script read it:
    MODEL_ID=Qwen/Qwen3.5-0.8B-Base modal run modal_app.py::train
(the script switches itself to the plain-prompt path when the model has no chat template, and
says so loudly, because that breaks comparability with the 4B numbers).
"""

import os
import subprocess
import sys
import time

import modal

APP_NAME = "diacritics-qwen35-08b-fullft"
PKG_DIR = "/pkg"          # this directory, mounted read-only into the container
SCRATCH = "/scratch"      # the Volume
RUN_ROOT = f"{SCRATCH}/runs/qwen3.5-0.8b-fullft"

SCRIPT = "train_full_ft_qwen35_08b.py"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")

app = modal.App(APP_NAME)

# Persistent scratch: HF cache (1.77GB of weights + the corpus + the tokenized Arrow shards),
# checkpoints, eval outputs. Shared with the 4B LoRA package on purpose -- same cache, same
# datasets, same pinned revisions. Modal gives 1 TiB of Volume storage free per month, then
# $0.09/GiB-month.
#
# BUDGET THE CHECKPOINTS. A full-FT checkpoint here is fp32 weights (~3.0GB) + fp32 AdamW state
# (~6.0GB) ~= 9GB, versus 108MB for a LoRA adapter. save_total_limit=2 keeps that at ~18GB, and
# final_model/ adds ~3GB. Delete a finished run's checkpoints:
#     modal volume rm -r diacritics-scratch runs/qwen3.5-0.8b-fullft/checkpoint-XXXX
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
    # `peft` is NOT used by this pipeline -- there are no adapters in a full fine-tune. It is
    # kept because removing it would change this layer's hash and force a fresh 10-30 minute
    # causal-conv1d compile below. An unused 3MB pure-Python wheel is the cheaper option.
    .pip_install(
        "transformers==5.14.1",
        "datasets==5.0.1",
        "accelerate==1.14.0",
        "peft==0.20.0",
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
            "OUTPUT_ROOT": RUN_ROOT,
            "EVAL_ROOT": f"{RUN_ROOT}/eval_outputs",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",   # ~4x faster download
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # ignore= is not tidiness. Run the training script locally even once and its defaults create
    # ./hf_cache, ./qwen3.5-0.8b-fullft-diacritization and ./eval_outputs right here -- and
    # ./hf_cache is ~1.8GB of weights, which would then be uploaded into the image on every
    # single deploy.
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path=PKG_DIR,
        ignore=[
            "**/__pycache__", "**/*.pyc",
            "**/hf_cache", "**/eval_outputs", "**/results",
            "**/*-fullft-diacritization",
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


def _run(cmd: list[str], commit_every: int = 0) -> None:
    """Shell out to the training script, streaming output, failing loudly.

    `commit_every` (seconds, 0 = off) commits the Volume periodically WHILE the subprocess runs.

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
    import train_full_ft_qwen35_08b as t

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
# One H100-80 throughout, and there is no case for more. The full-FT memory budget at 751M
# trainable params, batch 8 x 1024 tokens:
#     fp32 weights + grads + AdamW m,v      12.0 GB
#     autocast bf16 weight copies          ~ 1.5 GB
#     logits + fp32 CE upcast + gradient   ~20.0 GB   <- the real consumer, vocab is 248,320
#     activations (gradient checkpointing OFF)  ~6.0 GB
#                                          --------
#                                          ~40 GB of 80
# Sharding this across GPUs would add all-gather traffic to a job that fits twice over on one.
# --------------------------------------------------------------------------------------------


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=1 * HOUR, **GPU_SPEC)
def smoke_test():
    """12,000 rows, 60 steps, no checkpoints. Run this before paying for a real run.

    IT IS THE GATE, in two independent ways.

    CORRECTNESS. It exercises every part that can be silently wrong: the linear-attention
    fast-path check, the thinking-disabled prompt assertion, the text-only model class
    assertion, the fp32-master-weights assertion, and the label masking. The gate is the loss:
    it must be finite, non-zero and trending down. A loss near 0.0 at step 1 means the masking
    is inverted -- stop. The script raises on that itself, but read the printed losses anyway.

    COST. It prints a [THROUGHPUT] line measured over the last 40 of its 60 steps, discarding
    the first 20. That discard is not caution, it is necessary: step 1 carries ~120s of Triton
    JIT for the Gated-DeltaNet kernels, and the length-grouped sampler sorts each 50-step
    megabatch longest-first so early steps are the slowest in their cycle. RUN_REPORT.md §5.6
    records two bad cost estimates made by extrapolating from the first few steps. That
    [THROUGHPUT] line is the only estimate in this package you should budget from.
    """
    try:
        _run([sys.executable, SCRIPT, "--smoke-test"])
    finally:
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=24 * HOUR, **GPU_SPEC)
def train(max_steps: int = 0, train_size: int = 0,
          include_filename: str = "", exclude_filename: str = ""):
    """Full fine-tune, one epoch over the whole corpus. Training only -- score with `evaluate`.

    Split from evaluation on purpose: Modal caps a function at 24h and containers can be
    preempted, and a train followed by an eval in one function means a preemption during the eval
    loses the training result too. The script resumes from the newest COMPLETE checkpoint on the
    Volume (ranked by step number and requiring trainer_state.json, not by mtime -- a job
    preempted mid-write leaves a partial checkpoint holding the newest mtime, and resuming from
    that dies on a truncated shard).

    Defaults are the point of this package: max_steps=0 here means the script's MAX_STEPS=-1,
    i.e. ONE FULL EPOCH over all 1,042,693 usable rows. The 4B LoRA run saw 9.2% of one.

        max_steps       0 = full epoch. Set it only to build a cheaper rung.
        train_size      0 = the whole split.
        include/exclude_filename
                        regex over the corpus `filename` column, for a domain-balanced rung.
                        Run `analyze` first -- do not guess a pattern.
    """
    cmd = [sys.executable, SCRIPT, "--train-only"]
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    if train_size:
        cmd += ["--train-size", str(train_size)]
    if include_filename:
        cmd += ["--include-filename", include_filename]
    if exclude_filename:
        cmd += ["--exclude-filename", exclude_filename]
    try:
        # Commit every 10 min so checkpoints survive a hard kill, not just a clean failure.
        _run(cmd, commit_every=10 * 60)
    finally:
        scratch.commit()  # persist checkpoints even if the run dies or is preempted


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=12 * HOUR, **GPU_SPEC)
def evaluate(model: str = FINAL_MODEL_DIR, label: str = "ft",
             splits: str = "train,test,benchmark"):
    """Generation + DER/WER on train / test / SadeedDiac-25, into eval_outputs/.

    The benchmark is generated ONCE and scored three ways -- CA, MSA and pooled -- plus a macro
    mean, and each of those by BOTH scorers. Use the macro mean, not the pooled number: pooling
    is diacritic-weighted and CA paragraphs are longer, so it flatters the result by
    over-weighting the easy domain.

    Which column to read:
        DER_noce_corr  -> corrected scorer. CITE THIS. Compares to RESULTS.md's 8.48.
        DER_noce       -> frozen 4B-identical scorer. Compares to RUN_REPORT.md's 6.96, and to
                          nothing else -- RESULTS.md retired it.
    They are the same predictions under two scorers, not two runs. Never mix them.

    Re-invocable: predictions are checkpointed to CSV every ~200 rows, so a timeout or preemption
    picks up where it left off instead of regenerating.

    To score the ZERO-SHOT baseline through this identical pipeline (same prompt, same
    enable_thinking=False, same NFC scoring) -- the only honest comparison for the fine-tune:

        modal run modal_app.py::evaluate --model none --label base --splits benchmark

    `label` MUST differ between runs. It namespaces the prediction checkpoints, and those
    checkpoints are what a re-invocation resumes from: reuse "ft" for the base model and it will
    happily "resume" the fine-tuned model's cached predictions and report them as the baseline --
    indistinguishable from "fine-tuning changed nothing". RUN_REPORT.md §5.5 caught that one
    before it fired.
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
