"""
Modal runner for the Qwen3.5-0.8B double-pass char-tagger -- the plan's MAIN test.

Shares the `diacritics-scratch` Volume with every other package here: the HF cache (already
warm from ../Train-Tagging's run -- Sadeed_Tashkeela / SadeedDiac-25 are already pulled, no
warm_cache needed), the label vocab (already scanned + verified against the full corpus at
runs/arabert-tagging/labels.json -- SAME labels.py, SAME corpus, so it is reused rather than
rederived), and the warm-start checkpoint (uploaded once via `modal volume put` from the local
Train-Qwen-0.8B-Full-FT/.../Best-Step-4070 directory -- see this package's README).

The image is copied BYTE-FOR-BYTE from ../Train-Qwen-0.8B-Full-FT/modal_app.py through the
causal-conv1d / flash-linear-attention layers (see that file's docstring for why: Modal caches
image layers by definition hash, so identical lines are a cache hit instead of a fresh 10-30
minute CUDA compile -- though on THIS Modal account neither Qwen image has been built yet, so
the first build here pays that cost regardless).

Run order:
    modal run modal_app.py::prepare                        # CPU  — Phase 1b, double-pass tokenize
    modal run modal_app.py::smoke_test                      # GPU  — 60 steps, THE GATE
    modal run --detach modal_app.py::train
    modal run --detach modal_app.py::evaluate
    modal volume get diacritics-scratch runs/qwen-tagging/eval_outputs ./results
"""

import os
import subprocess
import sys
import time

import modal

APP_NAME = "diacritics-qwen-tagging"
PKG_DIR = "/pkg"
SCRATCH = "/scratch"
# The 2,000-step rung (macro DER_noce 3.55, evaluated and left in place as a reference point) is
# at runs/qwen-tagging/. This full-epoch run is namespaced separately so neither run's
# checkpoints/eval_outputs can collide with or overwrite the other's.
RUN_ROOT = f"{SCRATCH}/runs/qwen-tagging-fullepoch"
DATA_DIR = f"{SCRATCH}/tagging_data_qwen"
LABELS_PATH = f"{SCRATCH}/runs/arabert-tagging/labels.json"   # reused, not rederived
WARM_START_DIR = f"{SCRATCH}/runs/qwen3.5-0.8b-fullft/final_model"  # the 2.58-DER checkpoint

app = modal.App(APP_NAME)
scratch = modal.Volume.from_name("diacritics-scratch", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

# Copied byte-for-byte from ../Train-Qwen-0.8B-Full-FT/modal_app.py through flash-linear-attention
# -- see that file's docstring for the three build traps this pinning avoids.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(
        "transformers==5.14.1",
        "datasets==5.0.1",
        "accelerate==1.14.0",
        "peft==0.20.0",
        "jiwer==4.0.0",
        "pandas",
        "hf_transfer",
    )
    .env({
        "MAX_JOBS": "8",
        "TORCH_CUDA_ARCH_LIST": "9.0;10.0",
        "CXX": "g++",
        "CC": "gcc",
    })
    .pip_install("causal-conv1d", extra_options="--no-build-isolation")
    .pip_install("flash-linear-attention")
    .env({
        "HF_HOME": f"{SCRATCH}/hf_cache",
        "HF_HUB_CACHE": f"{SCRATCH}/hf_cache",
        "HF_CACHE_DIR": f"{SCRATCH}/hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_DISABLE_XET": "1",  # xet transfers stalled repeatedly in local testing
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "MODEL_ID": WARM_START_DIR,
    })
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path=PKG_DIR,
        ignore=["**/__pycache__", "**/*.pyc", "**/hf_cache", "**/results", "**/eval_outputs"],
    )
)

HOUR = 60 * 60
CPU_SPEC = dict(cpu=8, memory=32 * 1024)
GPU_SPEC = dict(gpu="H100:1", **CPU_SPEC)  # NOT A10G -- 0.8B full body + double-pass seq len


def _run(cmd: list, commit_every: int = 0):
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=PKG_DIR, env=dict(os.environ))
    if not commit_every:
        rc = proc.wait()
    else:
        last = time.time()
        while True:
            try:
                rc = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                if time.time() - last >= commit_every:
                    try:
                        scratch.commit()
                        print(f"[modal] volume committed (+{(time.time() - last) / 60:.0f} min)",
                              flush=True)
                    except Exception as e:
                        print(f"[modal] commit failed ({type(e).__name__}: {e}); continuing",
                              flush=True)
                    last = time.time()
    if rc != 0:
        sys.exit(rc)


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=1 * HOUR, **CPU_SPEC)
def check_warm_start():
    """Confirm the warm-start checkpoint (uploaded via `modal volume put`) is actually on the
    Volume before anything downstream assumes it. Cheap CPU check, no model load."""
    import os as _os
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not _os.path.exists(f"{WARM_START_DIR}/{f}")]
    if missing:
        raise RuntimeError(
            f"{WARM_START_DIR} is missing {missing}. Upload the checkpoint first: "
            f"modal volume put diacritics-scratch <local-checkpoint-dir> "
            f"runs/qwen3.5-0.8b-fullft/final_model"
        )
    if not _os.path.exists(LABELS_PATH):
        raise RuntimeError(
            f"{LABELS_PATH} not found -- run ../Train-Tagging's modal_app.py::scan_labels first "
            f"(this package reuses that vocab rather than rederiving it)."
        )
    print(f"[OK] warm-start checkpoint present at {WARM_START_DIR}")
    print(f"[OK] label vocab present at {LABELS_PATH}")


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=2 * HOUR, **CPU_SPEC)
def prepare(limit: int = 0):
    """Phase 1b: double-pass tokenize + char-align the full corpus. CPU only."""
    cmd = [sys.executable, "prepare_data_tagging_qwen.py", "--labels-path", LABELS_PATH,
           "--out", DATA_DIR, "--verify-input-column"]
    if limit:
        cmd += ["--limit", str(limit)]
    try:
        _run(cmd)
    finally:
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=1 * HOUR, **GPU_SPEC)
def smoke_test():
    """60 steps on a 12K-row slice. Gate on [SANITY] (not near 0) and is_fast_path_available
    (must not be False -- see model_tagging_qwen.py's assert_fast_path).

    DO NOT budget the real run from the [THROUGHPUT] s/step line. Measured: entering
    trainer.train() costs a FIXED ~190s here (Triton JIT of the Gated-DeltaNet / causal-conv1d
    kernels on the first step, dataloader worker spawn, CUDA allocator warmup) before any
    steady-state step happens. Over 60 steps that fixed cost IS 75% of the wall clock, so the
    smoke test reports ~3.5-4.5 s/step against a true steady state of ~1.07 s/step -- a 3-4x
    overestimate that priced the full epoch at $73 instead of $23. Solve it from two runs of
    DIFFERENT length at the same geometry (total = F + steps*r), or just use the 2,000-step
    rung's 1.283 s/step, which already amortizes F down to 8%."""
    try:
        _run([sys.executable, "train_tagging_qwen.py", "--data-dir", DATA_DIR,
              "--labels-path", LABELS_PATH, "--out", f"{RUN_ROOT}/smoke", "--smoke-test"])
    finally:
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=12 * HOUR, **GPU_SPEC)
def train(max_steps: int = 0, num_epochs: float = 1.0):
    """Phase 3, full epoch: max_steps=0 (the default -- THE POINT of this invocation) disables
    the step cap and trains for num_epochs=1.0 over the full 1,040,468-row corpus (~16,258
    steps at effective batch 64), exactly the "whole epoch" convention
    Train-Qwen-0.8B-Full-FT's generative run used for its own MAX_STEPS=-1/NUM_TRAIN_EPOCHS=1.

    Per-device batch is 16 (grad_accum 4, same effective batch 64 as the earlier 2,000-step
    rung, so eval_loss stays comparable): the 2,000-step run measured only 33.2/80GB H100 memory
    used, so this doubles the per-device batch to use more of that headroom. Do NOT re-judge it
    from smoke_test -- see that function's docstring for why 60 steps cannot measure this.

    Expected cost, from the 2,000-step run's solved steady state of 1.074 s/step (which is the
    generative full-FT's 1.046 s/step excluding eval -- the same speed, not 3x slower):
    ~16,258 steps -> ~5.0 h -> ~$23 at $4.58/hr, i.e. the same ballpark as the 4h48m the
    generative full FT took over the same corpus at the same effective batch."""
    try:
        _run([sys.executable, "train_tagging_qwen.py", "--data-dir", DATA_DIR,
              "--labels-path", LABELS_PATH, "--out", RUN_ROOT,
              "--max-steps", str(max_steps), "--num-epochs", str(num_epochs)],
             commit_every=5 * 60)
    finally:
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=4 * HOUR, **CPU_SPEC)
def diagnose_coverage(limit: int = 0):
    """Is the MSA gap a vocabulary-coverage problem? CPU only (~$0.63/hr, no GPU).

    Counts every bare word form in the training corpus, then buckets the benchmark's words by
    how often training saw them and reports CA vs MSA error rate WITHIN each bucket. Matched
    frequency is the whole point: MSA words are rarer here by construction, so "MSA errors are on
    rare words" proves nothing on its own. Decides whether sourcing MSA data is the right spend."""
    out_dir = f"{RUN_ROOT}/eval_outputs"
    try:
        _run([sys.executable, "diagnose_msa_coverage.py",
              "--predictions", f"{out_dir}/predictions_tagging_qwen.csv",
              "--out", f"{out_dir}/msa_coverage_diagnostic.csv",
              "--limit", str(limit)], commit_every=5 * 60)
    finally:
        scratch.commit()


@app.function(image=image, volumes={SCRATCH: scratch}, secrets=[hf_secret],
              timeout=1 * HOUR, **GPU_SPEC)
def evaluate(model_dir: str = f"{RUN_ROOT}/final_model"):
    """Phase 4. Same corrected metric + CA/MSA convention as the 2.58 baseline and the AraBERT
    control, so the printed macro DER_noce is directly comparable to both."""
    out_dir = f"{RUN_ROOT}/eval_outputs"
    try:
        _run([sys.executable, "evaluate_tagging_qwen.py", "--model-dir", model_dir,
              "--out", f"{out_dir}/metrics_tagging_qwen.csv",
              "--predictions-out", f"{out_dir}/predictions_tagging_qwen.csv"])
    finally:
        scratch.commit()


@app.local_entrypoint()
def main():
    print(__doc__)
