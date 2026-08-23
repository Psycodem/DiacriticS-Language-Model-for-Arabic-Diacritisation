"""
Modal runner for the 10% LoRA fine-tuning of Qwen3.5-4B and gemma-4-E4B-it.

Runs the SAME training scripts as Ibex, unmodified, driven by the same DIAC_*
environment variables the sbatch files set. Nothing about the training config is
re-specified here — that would be a second place for it to drift.

    modal run modal_train.py --model gemma
    modal run modal_train.py --model qwen

Config parity with run_{gemma,qwen}_lora_10pct.sbatch:
    GPUs                3 x A100-80GB      (matches --gpus=3 --constraint=a100)
    launcher            torchrun --nproc_per_node=3
    DIAC_TRAIN_FRACTION 0.1
    DIAC_RUN_TAG        10pct
    effective batch     96 (per_device x accum x world_size, derived in-script)
    warmup              5% ratio, resolved to steps in-script
    everything else     read from the training script itself

Environment pinned to what was VERIFIED working on Ibex. torch 2.11 specifically:
2.10 + transformers 5.15 cannot run gemma-4-E4B on CUDA at all (every dtype dies
in CUBLAS inside the per-layer-embeddings projection, while CPU works).
"""
import os
import subprocess
import modal

APP_NAME = "diacritization-lora"
VOLUME_NAME = "diac-outputs"
HF_SECRET = "huggingface-token"      # create with: modal secret create huggingface-token HF_TOKEN=hf_...

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = {
    "gemma": (os.path.join(REPO, "gemma4_lora_diacritization",
                           "Train_LoRA_Gemma_4_Diacritization.py"),
              "Train_LoRA_Gemma_4_Diacritization.py"),
    "qwen":  (os.path.join(REPO, "qwen35_4b_lora_diacritization",
                           "Train_LoRA_Qwen_3_5_4B_Diacritization.py"),
              "Train_LoRA_Qwen_3_5_4B_Diacritization.py"),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",              # NOT 2.10 — see module docstring
        "transformers==5.15.0",
        "peft==0.20.0",
        "accelerate",
        "datasets",
        "sentencepiece",
        "pyarabic",
        "jiwer",
        "pandas",
        "matplotlib",
        "tqdm",
        "huggingface_hub",
        "tabulate",                   # missing it cost two full Ibex runs at the final print
        "tiktoken",
    )
    .add_local_file(SCRIPTS["gemma"][0], f"/workspace/{SCRIPTS['gemma'][1]}")
    .add_local_file(SCRIPTS["qwen"][0],  f"/workspace/{SCRIPTS['qwen'][1]}")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="A100-80GB:3",
    # 16h ceiling. The 10h original was set before qwen-lora-50pct measured
    # 20h09 end-to-end on Ibex at the same 3x A100 shape. Generation over
    # 500+2485+1200 examples is a fixed cost that does not shrink with the
    # training fraction, so 10% is not a tenth of 50%. Unused hours are free;
    # hitting the ceiling mid-generation is not.
    timeout=60 * 60 * 16,
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def train(model: str, fraction: float = 0.1, tag: str = "10pct"):
    script = SCRIPTS[model][1]
    workdir = f"/outputs/{model}-{tag}"
    os.makedirs(workdir, exist_ok=True)
    # The script writes OUTPUT_DIR/EVAL_DIR relative to cwd, so run inside the
    # volume and everything lands on persistent storage as it goes.
    subprocess.run(["cp", f"/workspace/{script}", workdir], check=True)

    env = dict(os.environ)
    env.update({
        "DIAC_TRAIN_FRACTION": str(fraction),
        "DIAC_RUN_TAG": tag,
        "HF_CACHE_DIR": "/outputs/hf_cache",
        "HF_HOME": "/outputs/hf_home",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "8",
    })

    print(f"=== {model} {tag} on 3x A100-80GB ===", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader"], check=False)

    rc = subprocess.run(
        ["torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=3", script],
        cwd=workdir, env=env,
    ).returncode

    volume.commit()
    print(f"=== finished rc={rc}; outputs under {workdir} ===", flush=True)
    if rc != 0:
        raise RuntimeError(f"training exited {rc}")
    return workdir


@app.local_entrypoint()
def main(model: str = "gemma", fraction: float = 0.1, tag: str = "10pct"):
    """Fire-and-forget via .spawn().

    NOT train.remote(): that keeps the local entrypoint blocked for the whole
    job, so anything that disturbs the client (a timeout, the shell being
    reaped) tears the run down mid-training — which is exactly what killed the
    first two attempts. .spawn() hands the work to Modal and returns an id the
    run can be tracked by afterwards, independent of this process.
    """
    if model not in SCRIPTS:
        raise SystemExit(f"model must be one of {list(SCRIPTS)}")
    call = train.spawn(model, fraction, tag)
    print(f"SPAWNED {model} {tag}")
    print(f"CALL_ID={call.object_id}")
    print(f"track: python -m modal call-logs {call.object_id}")
