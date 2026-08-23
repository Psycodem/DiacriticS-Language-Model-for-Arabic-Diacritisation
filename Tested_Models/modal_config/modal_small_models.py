"""Small-models (<500M) zero-shot benchmark on Modal, replacing Ibex job 50697409.

Ports tested_models_v2/small_models/Small_Models_test.py (queued on Ibex as
`small-models-test`, 1xA100, 14h) to Modal. Two models, zero-shot, over the full
1,200-paragraph SadeedDiac-25 benchmark (600 MSA + 600 CA):

    basharalrfooh/Fine-Tashkeel   (seq2seq)
    Etherll/Tashkeel-350M-v2      (causal, chat template)

WHAT IS DELIBERATELY IDENTICAL TO THE IBEX SCRIPT
  * A100-80GB, matching Ibex's --constraint=a100, so these numbers stay
    comparable with every other Tested_Models_v2 result.
  * Batch size 1, greedy, max_new_tokens=1024 / max_length=1024. Batching would
    be far cheaper but introduces left-padding effects that no other v2 model
    was scored under; comparability is the entire point of this benchmark, so
    the cost is accepted rather than the variable.
  * The same MSA/CA split rule, the same 600-per-domain head slice, the same
    mean-row construction and the same output columns.

WHAT IS DELIBERATELY DIFFERENT
  1. No runtime `pip install`. The Ibex script shells out to pip on startup and
     installs UNPINNED torch/transformers, which here would silently overwrite
     the pinned image and reintroduce version bugs this project already paid
     for. Dependencies are baked into the image instead.
  2. The scorer is IMPORTED from Evaluation_Functions_Corrected.py rather than
     inlined. The inlined copy was verified logic-identical (AST-equal modulo
     docstrings), and importing keeps it that way permanently.
  3. Per-sample checkpointing. The Ibex ancestor of this job hit its wall clock
     with ZERO CSVs written, losing the entire run. Here every prediction is
     appended to the volume as it is produced, so an interrupted run resumes
     instead of restarting -- which on metered Modal credits is the difference
     between a pause and a loss.

USAGE
    # 1. cheap throughput probe -- run FIRST, it sets the cost expectation
    modal run modal_small_models.py --probe 12

    # 2. full run, detached, resumable
    modal run --detach modal_small_models.py

    # 3. bring results down
    modal volume get diac-outputs /small_models_v2/small_models_results.csv .
"""
import os
import modal

# Resolved from this file's location, not the CWD: `modal run` can be invoked
# from anywhere, and a CWD-relative mount would silently pick up the wrong file
# (or fail) depending on where the command was typed.
#
# The existence check is guarded by modal.is_local(). Modal re-imports this
# module INSIDE the container, where the local repo does not exist -- an
# unguarded check there aborts the container before the function ever runs.
_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_FNS = os.path.abspath(os.path.join(
    _HERE, "..", "..", "DiacriticS-Language-Model-for-Arabic-Diacritisation",
    "Evaluation_Functions_Corrected.py"))
if modal.is_local() and not os.path.exists(EVAL_FNS):
    raise SystemExit(f"FATAL: canonical scorer not found at {EVAL_FNS}")

VOLUME_NAME = "diac-outputs"
APP_NAME = "diac-small-models"
RUN_DIR = "/outputs/small_models_v2"
CACHE_DIR = "/outputs/small_models_v2/model_cache"

SAMPLE_SIZE = 600           # 600 MSA + 600 CA = 1200, exactly as on Ibex

MODEL_REGISTRY = {
    "Fine-Tashkeel":    {"repo": "basharalrfooh/Fine-Tashkeel", "type": "seq2seq"},
    "Tashkeel-350M-v2": {"repo": "Etherll/Tashkeel-350M-v2",    "type": "causal_chat"},
}

# Pinned for the reasons documented in Requirements.txt: transformers < 5.15 does
# not know this project's newer architectures, torch 2.10 + transformers 5.15
# breaks CUDA for them, and a missing `tabulate` has already destroyed two
# otherwise-complete benchmark runs at the final .to_markdown() call.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.15.0",
        "accelerate", "datasets", "sentencepiece",
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
    timeout=60 * 60 * 20,
    volumes={"/outputs": vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_small_models(probe: int = 0, only: str = ""):
    import os, sys, csv, gc, time, importlib.util
    import torch, pandas as pd
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    # --- canonical scorer, imported not copied ---
    spec = importlib.util.spec_from_file_location(
        "evalfns", "/root/Evaluation_Functions_Corrected.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["evalfns"] = ev
    spec.loader.exec_module(ev)
    print("[INFO] scorer: Evaluation_Functions_Corrected.py", flush=True)

    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_tok:
        from huggingface_hub import login
        login(token=hf_tok)
        os.environ["HF_TOKEN"] = hf_tok

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
        """Returns {idx: prediction} for samples already generated."""
        done = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    done[int(row["idx"])] = row["prediction"]
        return done

    def infer(label, meta, model, tokenizer, domain, test_set):
        """Batch-1 generation with per-sample checkpointing to the volume."""
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

        # Flush often enough that a short probe still commits something. A fixed
        # interval of 25 writes nothing at all when the probe is 12 samples long,
        # which is exactly when the throughput number matters most.
        flush_every = max(1, min(25, len(test_set) // 4))

        t0, generated = time.time(), 0
        for i, item in enumerate(test_set):
            if i in done:
                continue
            try:
                if meta["type"] == "seq2seq":
                    inputs = tokenizer(item["raw_input"], return_tensors="pt",
                                       max_length=1024, truncation=True).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_length=1024)
                    pred = tokenizer.decode(out[0], skip_special_tokens=True)
                else:
                    # Prompt copied from the Etherll/Tashkeel-350M-v2 model card,
                    # INCLUDING the space before the colon. Small_Models_test.py
                    # (and therefore the Ibex run) omits that space; for a 350M
                    # model the tokenisation of "النص :" and "النص:" is not the
                    # same, so the card's exact string is what the model was
                    # instruction-tuned against.
                    mi = tokenizer.apply_chat_template(
                        [{"role": "user",
                          "content": "قم بتشكيل هذا النص " + ":\n" + item["raw_input"]}],
                        add_generation_prompt=True, return_tensors="pt",
                        tokenize=True, return_dict=True).to(model.device)
                    with torch.no_grad():
                        # explicit max_new_tokens: without it generate() falls back to
                        # a ~20-token default that truncates the diacritized output
                        out = model.generate(**mi, max_new_tokens=1024, do_sample=False)
                    pred = tokenizer.decode(out[0, mi["input_ids"].shape[-1]:],
                                            skip_special_tokens=True).strip()
            except Exception as e:
                print(f"  [warn] {label}: sample {i} failed, empty prediction ({e})", flush=True)
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
            raise SystemExit(f"FATAL: --only {only!r} matches nothing in "
                             f"{list(MODEL_REGISTRY)}")
        print(f"[INFO] --only {only}: running {list(registry)}", flush=True)

    records, skipped = [], []
    for label, meta in registry.items():
        print(f"\n{'='*60}\n Loading model: {label}  ({meta['repo']})\n{'='*60}", flush=True)
        model = tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(meta["repo"], cache_dir=CACHE_DIR)
            if meta["type"] == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    meta["repo"], cache_dir=CACHE_DIR).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    meta["repo"], device_map="auto", dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR)
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

    # ---- results table, same shape as the Ibex script ----
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
            categories=["MSA (Modern)", "CA (Classical)", "Mean (MSA + CA)"],
            ordered=True)
        full = full.sort_values(["Model", "Domain Track"]).reset_index(drop=True)

    print("\n=== SMALL MODELS EVALUATION MATRIX (WITH MSA / CA MEAN ROW) ===", flush=True)
    print(full.to_markdown(index=False) if not full.empty else "No models processed.",
          flush=True)

    name = f"small_models_results{'_probe' + str(probe) if probe else ''}.csv"
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

    THIS IS THE FIX FOR THE 2026-08-20 12:12 CANCELLATION. The first full run
    died 12 minutes in with:

        [modal-client] Received a cancellation signal while processing input

    `--detach` keeps the APP alive after the client exits, but `.remote()` makes
    the client sit and block on the function's result. When that client process
    is killed -- a shell timeout, a closed terminal, an agent session ending --
    Modal treats the death of the awaiting client as a cancellation and kills the
    input with it. Detach does not protect against that, because the problem is
    the outstanding await, not the app lifetime.

    `.spawn()` submits the call and returns a handle immediately, so the client
    has nothing to await and exits cleanly in seconds. Nothing remains that can
    be SIGTERMed into cancelling the work. Combined with `--detach`, the app and
    the container both outlive the launcher.

        modal run --detach modal_small_models.py            # fire and forget
        modal run modal_small_models.py --probe 5 --wait    # short smoke, block

    Pass --wait only for short runs you intend to babysit.
    """
    if wait:
        print(run_small_models.remote(probe=probe, only=only))
        return

    call = run_small_models.spawn(probe=probe, only=only)
    print(f"\nspawned call id : {call.object_id}")
    print(f"app id          : {app.app_id}")
    print("\nThe container now runs independently of this process.")
    print("Track it with:")
    print(f"    modal app logs {app.app_id}")
    print("    modal volume ls diac-outputs small_models_v2")
