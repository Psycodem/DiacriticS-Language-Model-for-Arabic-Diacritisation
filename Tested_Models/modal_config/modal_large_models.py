"""Large instruct-model zero-shot benchmark on Modal, replacing Ibex job 50682687.

Ports tested_models_v2/large_models/Large_Models_test.py (queued on Ibex as
`large-models-test`, 1xA100, 18h) to Modal. Two general instruct models, prompted
zero-shot, over the full 1,200-paragraph SadeedDiac-25 benchmark (600 MSA + 600 CA):

    CohereLabs/aya-expanse-8b                 (gated -- needs an accepted licence)
    moonshotai/Moonlight-16B-A3B-Instruct     (MoE, trust_remote_code)

TRANSFORMERS IS PINNED TO 4.48.2 ON PURPOSE, and this is the one place this file
deliberately departs from the rest of the project (which runs 5.15). The Ibex
script hard-pins `transformers==4.48.2` in its own bootstrap, and Moonlight ships
`trust_remote_code` modelling code written against that API. Running it under 5.x
risks the remote code failing outright, and -- worse -- it would make these
numbers incomparable with whatever the Ibex run would have produced. torch is
pinned to 2.6.0 to match that era rather than the 2.11 the Gemma-4/Qwen runs need;
that constraint comes from Gemma-4's CUDA kernels and does not apply to these two
models.

Also baked in, each from a failure this project has already paid for:
  * `tiktoken` + `blobfile`: Moonlight's tokenizer needs them. Their absence made
    the model fail to load on Ibex and silently drop out of the results table.
  * `tabulate`: pandas .to_markdown(). Missing, it crashes AFTER all inference is
    done but BEFORE any CSV is written -- this exact script lost 1h18 of A100
    time to it once already.
  * Per-sample checkpointing to the volume, so an interrupted run resumes.
  * `.spawn()` rather than `.remote()` -- see the note on main() below.

USAGE
    modal run modal_large_models.py --probe 5 --wait     # smoke test, blocking
    modal run --detach modal_large_models.py             # full run, fire-and-forget
    bash fetch_large_models.sh                           # bring results down
"""
import os
import modal

_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_FNS = os.path.abspath(os.path.join(
    _HERE, "..", "..", "DiacriticS-Language-Model-for-Arabic-Diacritisation",
    "Evaluation_Functions_Corrected.py"))
# Guarded: Modal re-imports this module inside the container, where the local
# repo does not exist. An unguarded check aborts the container before it starts.
if modal.is_local() and not os.path.exists(EVAL_FNS):
    raise SystemExit(f"FATAL: canonical scorer not found at {EVAL_FNS}")

VOLUME_NAME = "diac-outputs"
APP_NAME = "diac-large-models"
RUN_DIR = "/outputs/large_models_v2"
CACHE_DIR = "/outputs/large_models_v2/model_cache"

SAMPLE_SIZE = 600           # 600 MSA + 600 CA = 1200, exactly as on Ibex

DIACRITIZE_INSTRUCTION = "قم بتشكيل هذا النص تشكيلاً كاملاً بدون أي إضافات أو شرح:\n{text}"

MODEL_REGISTRY = {
    "aya-expanse-8b": {
        "repo": "CohereLabs/aya-expanse-8b",
        "trust_remote_code": False,
        "system_message": None,
    },
    "Moonlight-16B-A3B-Instruct": {
        "repo": "moonshotai/Moonlight-16B-A3B-Instruct",
        "trust_remote_code": True,
        "system_message": "You are a helpful assistant provided by Moonshot-AI.",
    },
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.48.2",
        "accelerate", "datasets", "sentencepiece", "einops",
        "tiktoken", "blobfile",
        "jiwer", "pandas", "tqdm", "huggingface_hub", "tabulate",
    )
    .add_local_file(EVAL_FNS, "/root/Evaluation_Functions_Corrected.py")
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _load_benchmark(strip_diacritics, sample_size):
    """SadeedDiac-25 split into MSA and CA tracks -- same rule as the Ibex script."""
    from datasets import load_dataset
    print("Loading SadeedDiac-25 Dataset...", flush=True)
    dataset = load_dataset("Misraj/SadeedDiac-25", split="train")

    msa_samples, ca_samples = [], []
    for item in dataset:
        text_input = item.get("input", "") or ""
        text_output = item.get("output", "") or ""
        source = (item.get("filename", "") or "").lower()
        sample = {"ground_truth": text_output, "raw_input": strip_diacritics(text_input)}
        if "fadel" in source or source in ["religion", "classical_poetry", "hadith"]:
            ca_samples.append(sample)
        else:
            msa_samples.append(sample)

    print(f"[INFO] pool: {len(msa_samples)} MSA, {len(ca_samples)} CA", flush=True)
    return [("MSA (Modern)", msa_samples[:sample_size]),
            ("CA (Classical)", ca_samples[:sample_size])]


@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=60 * 60 * 24,
    volumes={"/outputs": vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_large_models(probe: int = 0, only: str = ""):
    import sys, csv, gc, time, importlib.util
    import torch, pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM

    spec = importlib.util.spec_from_file_location(
        "evalfns", "/root/Evaluation_Functions_Corrected.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["evalfns"] = ev
    spec.loader.exec_module(ev)
    print("[INFO] scorer: Evaluation_Functions_Corrected.py", flush=True)

    import transformers as _tf
    print(f"[INFO] torch {torch.__version__} | transformers {_tf.__version__}", flush=True)

    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_tok:
        from huggingface_hub import login
        login(token=hf_tok)
        os.environ["HF_TOKEN"] = hf_tok
        print("[INFO] HF token loaded (aya-expanse-8b is gated)", flush=True)
    else:
        print("[WARN] no HF token -- aya-expanse-8b will land in the skip log", flush=True)

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = f"{RUN_DIR}/hf_home"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    n = probe if probe > 0 else SAMPLE_SIZE
    if probe:
        print(f"\n*** PROBE MODE: {probe} samples per model per domain ***\n", flush=True)
    domains = _load_benchmark(ev.strip_diacritics, n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device} "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else ''}", flush=True)

    def ckpt_path(label, domain):
        safe = domain.split(" ")[0]
        tag = f"probe{probe}_" if probe else ""
        return f"{RUN_DIR}/preds_{tag}{label}_{safe}.csv"

    def load_ckpt(path):
        done = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    done[int(row["idx"])] = row["prediction"]
        return done

    def infer(label, meta, model, tokenizer, domain, test_set):
        """Batch-1 greedy generation with per-sample checkpointing to the volume."""
        path = ckpt_path(label, domain)
        done = load_ckpt(path)
        if done:
            print(f"  [resume] {label}/{domain}: {len(done)}/{len(test_set)} already done",
                  flush=True)

        new_file = not os.path.exists(path)
        fh = open(path, "a", encoding="utf-8", newline="")
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["idx", "prediction"])

        # Small enough that a short probe still commits something; a fixed 25
        # would write nothing at all on a 5-sample smoke test.
        flush_every = max(1, min(25, len(test_set) // 4))

        t0, generated = time.time(), 0
        for i, item in enumerate(test_set):
            if i in done:
                continue
            try:
                messages = []
                if meta["system_message"]:
                    messages.append({"role": "system", "content": meta["system_message"]})
                messages.append({"role": "user",
                                 "content": DIACRITIZE_INSTRUCTION.format(text=item["raw_input"])})

                mi = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt",
                    tokenize=True, return_dict=True).to(model.device)
                with torch.no_grad():
                    # explicit max_new_tokens: the default is far too short for
                    # a fully diacritized paragraph
                    out = model.generate(**mi, max_new_tokens=1024, do_sample=False)
                pred = tokenizer.decode(out[0, mi["input_ids"].shape[-1]:],
                                        skip_special_tokens=True).strip()
            except Exception as e:
                print(f"  [warn] {label}: sample {i} failed, empty prediction ({e})",
                      flush=True)
                pred = ""

            done[i] = pred
            writer.writerow([i, pred])
            generated += 1

            if generated % flush_every == 0:
                fh.flush()
                vol.commit()
                rate = (time.time() - t0) / generated
                left = (len(test_set) - len(done)) * rate
                print(f"  {label}/{domain}: {len(done)}/{len(test_set)}  "
                      f"{rate:.2f}s/sample  ~{left/60:.0f} min left", flush=True)

        fh.close()
        vol.commit()
        if generated:
            rate = (time.time() - t0) / generated
            print(f"  [THROUGHPUT] {label}/{domain}: {rate:.2f} s/sample "
                  f"over {generated} new samples", flush=True)

        predictions = [done.get(i, "") for i in range(len(test_set))]
        references = [x["ground_truth"] for x in test_set]
        return ev.compute_der_wer(predictions, references)

    registry = MODEL_REGISTRY
    if only:
        registry = {k: v for k, v in MODEL_REGISTRY.items() if k.lower() == only.lower()}
        if not registry:
            raise SystemExit(f"FATAL: --only {only!r} matches nothing in {list(MODEL_REGISTRY)}")
        print(f"[INFO] --only {only}: running {list(registry)}", flush=True)

    records, skipped = [], []
    for label, meta in registry.items():
        print(f"\n{'='*60}\n Loading model: {label}  ({meta['repo']})\n{'='*60}", flush=True)
        model = tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                meta["repo"], cache_dir=CACHE_DIR,
                trust_remote_code=meta["trust_remote_code"])
            model = AutoModelForCausalLM.from_pretrained(
                meta["repo"], torch_dtype="auto", device_map="auto",
                trust_remote_code=meta["trust_remote_code"], cache_dir=CACHE_DIR)
            model.eval()
            vol.commit()

            for domain, test_set in domains:
                metrics = infer(label, meta, model, tokenizer, domain, test_set)
                metrics.update({"Model": label, "Domain Track": domain})
                records.append(metrics)
                print(f"  -> {label} / {domain}: {metrics}", flush=True)
        except Exception as e:
            skipped.append((label, f"{type(e).__name__}: {e}"))
            print(f"Skipping {label}: {e}", flush=True)
        finally:
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Finished {label} - GPU cache cleared.\n", flush=True)

    df = pd.DataFrame(records)
    if df.empty:
        full = df
        print("No models produced results - check the skip log below.", flush=True)
    else:
        cols = ["DER_ce", "DER_noce", "WER_ce", "WER_noce"]
        mean = df.groupby("Model")[cols].mean().round(2).reset_index()
        mean.insert(1, "Domain Track", "Mean (MSA + CA)")
        full = pd.concat([df, mean], ignore_index=True)
        full["Domain Track"] = pd.Categorical(
            full["Domain Track"],
            categories=["MSA (Modern)", "CA (Classical)", "Mean (MSA + CA)"], ordered=True)
        full = full.sort_values(["Model", "Domain Track"]).reset_index(drop=True)

    print("\n=== INSTRUCT MODELS EVALUATION MATRIX (WITH MSA / CA MEAN ROW) ===", flush=True)
    print(full.to_markdown(index=False) if not full.empty else "No models processed.",
          flush=True)

    name = f"large_models_results{'_probe' + str(probe) if probe else ''}.csv"
    out_csv = f"{RUN_DIR}/{name}"
    full.to_csv(out_csv, index=False)
    vol.commit()
    print(f"\nResults saved to {out_csv}", flush=True)

    if skipped:
        print("\n================ SKIPPED / FAILED MODELS (explicit) ================",
              flush=True)
        for nm, reason in skipped:
            print(f"- {nm}: {reason}", flush=True)
    else:
        print("\nAll registered models ran successfully - no gaps in the table above.",
              flush=True)

    return full.to_csv(index=False)


@app.local_entrypoint()
def main(probe: int = 0, only: str = "", wait: bool = False):
    """Fire-and-forget by default.

    `.spawn()`, not `.remote()`. `--detach` keeps the APP alive after the client
    exits, but `.remote()` leaves the client blocking on the result, and when
    that client process is killed -- shell timeout, closed terminal, agent
    session ending -- Modal cancels the input along with it. That is exactly how
    the first small-models full run died 12 minutes in on 2026-08-20.
    `.spawn()` returns a handle immediately, so nothing is left to cancel.

    Pass --wait only for short runs you intend to babysit (e.g. --probe 5).
    """
    if wait:
        print(run_large_models.remote(probe=probe, only=only))
        return

    call = run_large_models.spawn(probe=probe, only=only)
    print(f"\nspawned call id : {call.object_id}")
    print(f"app id          : {app.app_id}")
    print("\nThe container now runs independently of this process.")
    print("Track it with:")
    print(f"    modal app logs {app.app_id}")
    print("    modal volume ls diac-outputs large_models_v2")
