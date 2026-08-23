"""
Fanar-1-9B-Instruct zero-shot benchmark on Modal, replacing Ibex job 50682684.

Ports tested_models_v2/fanar/Fanar_Model_test.py (queued on Ibex as
`fanar-model-test`, 1xA100, 6h) to Modal. One 9B instruct model, 4-bit NF4
quantized, over the full 1,200-paragraph SadeedDiac-25 benchmark.

FIXES THE BARE REPO ID. The Ibex script has:

    large_models = {"Fanar-1-9B-Instruct": "Fanar-1-9B-Instruct"}

i.e. a bare model name with no org prefix. There is no such repo on the Hub, so
from_pretrained() raises, the except branch appends to skipped_models, the loop
continues, and the job exits 0 with an EMPTY results table. That is exactly what
happened on Ibex once already -- reported COMPLETED in 26 seconds having tested
nothing. The correct id is QCRI/Fanar-1-9B-Instruct, used below.

Because that bug means the model has never actually been scored, there is no
prior Fanar number for these results to disagree with.

Everything else is kept faithful to the Ibex script:
  * 4-bit NF4 + double quant, bf16 compute (BitsAndBytesConfig, unchanged)
  * the same Arabic system prompt, greedy, max_new_tokens=512
  * .strip().split("\\n")[0] post-processing -- Fanar is chatty, and the script
    keeps only the first line
  * Fanar's OWN MSA/CA split rule, which is 'fadel' in filename vs not. Note
    this differs from the other v2 scripts, which also route religion /
    classical_poetry / hadith to CA. Kept as-is so the split matches what the
    Fanar job would have measured.
  * per-sample checkpointing to the volume, so an interrupted run resumes
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
APP_NAME = "diac-fanar"
RUN_DIR = "/outputs/fanar_v2"
CACHE_DIR = "/outputs/fanar_v2/model_cache"

SAMPLE_SIZE = 600           # 600 MSA + 600 CA = 1200, exactly as on Ibex

MODEL_LABEL = "Fanar-1-9B-Instruct"
MODEL_REPO = "QCRI/Fanar-1-9B-Instruct"     # NOT the bare name the Ibex script used
SYSTEM_PROMPT = "قم بتشكيل النص العربي التالي تشكيلاً كاملاً ودقيقاً دون أي إضافة:"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.15.0",
        "bitsandbytes>=0.45",        # required for the 4-bit NF4 path
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
def run_fanar(probe: int = 0):
    import sys, csv, gc, time, importlib.util
    import torch, pandas as pd
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = f"{RUN_DIR}/hf_home"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    n = probe if probe > 0 else SAMPLE_SIZE
    if probe:
        print(f"\n*** PROBE MODE: {probe} samples per domain ***\n", flush=True)

    print("Loading SadeedDiac-25 Dataset...", flush=True)
    dataset = load_dataset("Misraj/SadeedDiac-25", split="train")
    # Fanar's own split rule: fadel vs not. Deliberately NOT the broader rule the
    # other v2 scripts use -- see the module docstring.
    msa = [{"ground_truth": x["output"], "raw_input": ev.strip_diacritics(x["input"])}
           for x in dataset if "fadel" not in (x.get("filename", "") or "").lower()][:n]
    ca = [{"ground_truth": x["output"], "raw_input": ev.strip_diacritics(x["input"])}
          for x in dataset if "fadel" in (x.get("filename", "") or "").lower()][:n]
    print(f"[INFO] MSA {len(msa)}, CA {len(ca)}", flush=True)
    domains = [("MSA (Modern)", msa), ("CA (Classical)", ca)]

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: no CUDA device -- 4-bit quantized inference needs a GPU")
    print(f"[INFO] device=cuda {torch.cuda.get_device_name(0)}", flush=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    def ckpt_path(domain):
        tag = f"probe{probe}_" if probe else ""
        return f"{RUN_DIR}/preds_{tag}{MODEL_LABEL}_{domain.split(' ')[0]}.csv"

    def load_ckpt(path):
        done = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    done[int(row["idx"])] = row["prediction"]
        return done

    def infer(model, tokenizer, domain, test_set):
        path = ckpt_path(domain)
        done = load_ckpt(path)
        if done:
            print(f"  [resume] {domain}: {len(done)}/{len(test_set)} already done", flush=True)

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
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": item["raw_input"]}]
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = f"{SYSTEM_PROMPT}\n\nالنص: {item['raw_input']}\nالتشكيل:"

                # model.device, not a hardcoded "cuda": with device_map="auto" the
                # model may be sharded and this still resolves to the input device.
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    # greedy, so temperature is omitted rather than contradicted
                    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:],
                                       skip_special_tokens=True)
                pred = raw.strip().split("\n")[0]     # Fanar is chatty; keep line 1
            except Exception as e:
                print(f"  [warn] sample {i} failed, empty prediction ({e})", flush=True)
                pred = ""

            done[i] = pred
            writer.writerow([i, pred])
            generated += 1
            if generated % flush_every == 0:
                fh.flush(); vol.commit()
                rate = (time.time() - t0) / generated
                print(f"  {domain}: {len(done)}/{len(test_set)}  {rate:.2f}s/sample  "
                      f"~{(len(test_set)-len(done))*rate/60:.0f} min left", flush=True)

        fh.close(); vol.commit()
        if generated:
            print(f"  [THROUGHPUT] {domain}: "
                  f"{(time.time()-t0)/generated:.2f} s/sample over {generated} new samples",
                  flush=True)

        return ev.compute_der_wer([done.get(i, "") for i in range(len(test_set))],
                                  [x["ground_truth"] for x in test_set])

    records, skipped = [], []
    model = tokenizer = None
    print(f"\n{'='*60}\n Loading model: {MODEL_LABEL}  ({MODEL_REPO})\n{'='*60}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO, trust_remote_code=True, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO, quantization_config=quantization_config, device_map="auto",
            trust_remote_code=True, cache_dir=CACHE_DIR)
        model.eval()
        vol.commit()
        for domain, test_set in domains:
            metrics = infer(model, tokenizer, domain, test_set)
            metrics.update({"Model": MODEL_LABEL, "Domain Track": domain})
            records.append(metrics)
            print(f"  -> {MODEL_LABEL} / {domain}: {metrics}", flush=True)
    except Exception as e:
        skipped.append((MODEL_LABEL, f"{type(e).__name__}: {e}"))
        print(f"Skipping {MODEL_LABEL}: {e}", flush=True)
    finally:
        del model, tokenizer
        torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(records)
    if df.empty:
        full = df
        print("No results - check the skip log below.", flush=True)
    else:
        cols = ["DER_ce", "DER_noce", "WER_ce", "WER_noce"]
        mean = df.groupby("Model")[cols].mean().round(2).reset_index()
        mean.insert(1, "Domain Track", "Mean (MSA + CA)")
        full = pd.concat([df, mean], ignore_index=True)
        full["Domain Track"] = pd.Categorical(
            full["Domain Track"],
            categories=["MSA (Modern)", "CA (Classical)", "Mean (MSA + CA)"], ordered=True)
        full = full.sort_values(["Model", "Domain Track"]).reset_index(drop=True)

    print("\n=== FANAR EVALUATION MATRIX (WITH MSA / CA MEAN ROW) ===", flush=True)
    print(full.to_markdown(index=False) if not full.empty else "No models processed.",
          flush=True)

    out_csv = f"{RUN_DIR}/fanar_results{'_probe' + str(probe) if probe else ''}.csv"
    full.to_csv(out_csv, index=False)
    vol.commit()
    print(f"\nResults saved to {out_csv}", flush=True)

    if skipped:
        print("\n================ SKIPPED / FAILED (explicit) ================", flush=True)
        for nm, reason in skipped:
            print(f"- {nm}: {reason}", flush=True)
    else:
        print("\nModel ran successfully - no gaps in the table above.", flush=True)

    return full.to_csv(index=False)


@app.local_entrypoint()
def main(probe: int = 0, wait: bool = False):
    """`.spawn()` by default -- see modal_large_models.py for why `.remote()`
    gets the run cancelled when the launching client is killed."""
    if wait:
        print(run_fanar.remote(probe=probe))
        return
    call = run_fanar.spawn(probe=probe)
    print(f"\nspawned call id : {call.object_id}")
    print(f"app id          : {app.app_id}")
    print(f"    modal app logs {app.app_id}")
