"""Flan-T5 / ByT5 zero-shot benchmark on Modal, replacing Ibex job 50682686.

Ports tested_models_v2/flan_t5_byt5/Flan-T5-based_ByT5-based_test.ipynb (queued
on Ibex as `flan-t5-byt5-test`, 1xA100, 4h) to Modal. Two small seq2seq
diacritization models over the full 1,200-paragraph SadeedDiac-25 benchmark
(600 MSA + 600 CA):

    Abdou/arabic-tashkeel-flan-t5-small
    glonor/byt5-arabic-diacritization

THREE THINGS THIS PORT FIXES, all of which the notebook gets wrong:

  1. THE SCORER. The notebook carries a DRIFTED copy of the evaluation code:
     its clean_and_tokenize() is missing normalize_numerals() and
     strip_citation_refs(), so Eastern-Arabic digits are never folded to Western
     and Fadel-style '(41 / 251)' citations are never stripped. That is the same
     defect already fixed in Fanar_Model_test.py and Small_Models_test.py; the
     earlier audit only swept .py files, so the notebook was missed. This port
     imports Evaluation_Functions_Corrected.py directly, so it cannot drift
     again -- but it also means these numbers are NOT comparable with any
     previously published Flan-T5/ByT5 figure produced by the notebook.

  2. IT WRITES A CSV. The notebook has no to_csv() call at all -- its only
     output was cell output inside an executed .ipynb, which is why the sbatch
     had that whole paragraph about executed_runs/. Results here land in a real
     CSV like every other v2 script.

  3. IT LOADS EACH MODEL ONCE. The notebook loops domain-outer/model-inner, so
     every model is loaded twice (once for MSA, once for CA). Same predictions,
     twice the load cost. This runs model-outer/domain-inner.

Column names are the v2-standard DER_ce/DER_noce/WER_ce/WER_noce rather than the
notebook's prose headings, so the CSV joins cleanly with the other v2 results.

USAGE
    modal run modal_flan_byt5.py --probe 1 --wait     # smoke test, blocking
    modal run --detach modal_flan_byt5.py             # full run, fire-and-forget
"""
import os
import modal

_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_FNS = os.path.abspath(os.path.join(
    _HERE, "..", "..", "DiacriticS-Language-Model-for-Arabic-Diacritisation",
    "Evaluation_Functions_Corrected.py"))
if modal.is_local() and not os.path.exists(EVAL_FNS):
    raise SystemExit(f"FATAL: canonical scorer not found at {EVAL_FNS}")

VOLUME_NAME = "diac-outputs"
APP_NAME = "diac-flan-byt5"
RUN_DIR = "/outputs/flan_byt5_v2"
CACHE_DIR = "/outputs/flan_byt5_v2/model_cache"

SAMPLE_SIZE = 600           # 600 MSA + 600 CA = 1200, exactly as on Ibex

MODEL_REGISTRY = {
    "Flan-T5-Tashkeel-Small": {"repo": "Abdou/arabic-tashkeel-flan-t5-small"},
    "Glonor-ByT5-Arabic":     {"repo": "glonor/byt5-arabic-diacritization"},
}

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


@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=60 * 60 * 12,
    volumes={"/outputs": vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_flan_byt5(probe: int = 0, only: str = ""):
    import sys, csv, gc, time, importlib.util
    import torch, pandas as pd
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    spec = importlib.util.spec_from_file_location(
        "evalfns", "/root/Evaluation_Functions_Corrected.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["evalfns"] = ev
    spec.loader.exec_module(ev)
    print("[INFO] scorer: Evaluation_Functions_Corrected.py "
          "(notebook's drifted copy deliberately NOT used)", flush=True)

    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_tok:
        from huggingface_hub import login
        login(token=hf_tok)

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = f"{RUN_DIR}/hf_home"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    n = probe if probe > 0 else SAMPLE_SIZE
    if probe:
        print(f"\n*** PROBE MODE: {probe} samples per model per domain ***\n", flush=True)

    print("Loading SadeedDiac-25 Dataset...", flush=True)
    dataset = load_dataset("Misraj/SadeedDiac-25", split="train")
    msa_samples, ca_samples = [], []
    for item in dataset:
        source = (item.get("filename", "") or "").lower()
        sample = {"ground_truth": item.get("output", "") or "",
                  "raw_input": ev.strip_diacritics(item.get("input", "") or "")}
        # the notebook's own split rule, preserved verbatim
        if "fadel" in source or source in ["religion", "classical_poetry", "hadith"]:
            ca_samples.append(sample)
        else:
            msa_samples.append(sample)
    print(f"[INFO] pool: {len(msa_samples)} MSA, {len(ca_samples)} CA", flush=True)
    domains = [("MSA (Modern)", msa_samples[:n]), ("CA (Classical)", ca_samples[:n])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device} "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else ''}", flush=True)

    def ckpt_path(label, domain):
        tag = f"probe{probe}_" if probe else ""
        return f"{RUN_DIR}/preds_{tag}{label}_{domain.split(' ')[0]}.csv"

    def load_ckpt(path):
        done = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    done[int(row["idx"])] = row["prediction"]
        return done

    def infer(label, model, tokenizer, domain, test_set):
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
        flush_every = max(1, min(25, len(test_set) // 4))

        t0, generated = time.time(), 0
        for i, item in enumerate(test_set):
            if i in done:
                continue
            try:
                inputs = tokenizer(item["raw_input"], return_tensors="pt",
                                   max_length=1024, truncation=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=1024)
                pred = tokenizer.decode(out[0], skip_special_tokens=True)
            except Exception as e:
                print(f"  [warn] {label}: sample {i} failed, empty prediction ({e})",
                      flush=True)
                pred = ""

            done[i] = pred
            writer.writerow([i, pred])
            generated += 1
            if generated % flush_every == 0:
                fh.flush(); vol.commit()
                rate = (time.time() - t0) / generated
                print(f"  {label}/{domain}: {len(done)}/{len(test_set)}  {rate:.2f}s/sample  "
                      f"~{(len(test_set)-len(done))*rate/60:.0f} min left", flush=True)

        fh.close(); vol.commit()
        if generated:
            print(f"  [THROUGHPUT] {label}/{domain}: "
                  f"{(time.time()-t0)/generated:.2f} s/sample over {generated} new samples",
                  flush=True)

        return ev.compute_der_wer([done.get(i, "") for i in range(len(test_set))],
                                  [x["ground_truth"] for x in test_set])

    registry = MODEL_REGISTRY
    if only:
        registry = {k: v for k, v in MODEL_REGISTRY.items() if k.lower() == only.lower()}
        if not registry:
            raise SystemExit(f"FATAL: --only {only!r} matches nothing in {list(MODEL_REGISTRY)}")

    records, skipped = [], []
    for label, meta in registry.items():
        print(f"\n{'='*60}\n Loading model: {label}  ({meta['repo']})\n{'='*60}", flush=True)
        model = tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(meta["repo"], cache_dir=CACHE_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                meta["repo"], cache_dir=CACHE_DIR).to(device)
            model.eval()
            vol.commit()
            for domain, test_set in domains:
                metrics = infer(label, model, tokenizer, domain, test_set)
                metrics.update({"Model": label, "Domain Track": domain})
                records.append(metrics)
                print(f"  -> {label} / {domain}: {metrics}", flush=True)
        except Exception as e:
            skipped.append((label, f"{type(e).__name__}: {e}"))
            print(f"Skipping {label}: {e}", flush=True)
        finally:
            del model, tokenizer
            torch.cuda.empty_cache(); gc.collect()
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

    print("\n=== FLAN-T5 / ByT5 EVALUATION MATRIX (WITH MSA / CA MEAN ROW) ===", flush=True)
    print(full.to_markdown(index=False) if not full.empty else "No models processed.",
          flush=True)

    out_csv = f"{RUN_DIR}/flan_byt5_results{'_probe' + str(probe) if probe else ''}.csv"
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
    """`.spawn()` by default -- see modal_large_models.py for why `.remote()`
    gets the run cancelled when the launching client is killed."""
    if wait:
        print(run_flan_byt5.remote(probe=probe, only=only))
        return
    call = run_flan_byt5.spawn(probe=probe, only=only)
    print(f"\nspawned call id : {call.object_id}")
    print(f"app id          : {app.app_id}")
    print(f"    modal app logs {app.app_id}")
