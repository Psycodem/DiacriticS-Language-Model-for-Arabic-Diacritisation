# -*- coding: utf-8 -*-
"""benchmark_runner.py — shared harness for the v2 re-scoring scripts.

The v2 scripts exist to fix one thing: the v1 scripts each scored with their own
drifted copy of DER/WER (see models_functions.py). So the runner's job is to
reproduce v1 generation EXACTLY and change only the scoring path — otherwise a
moved number can't be attributed to the metric fix rather than a prompt change.

Concretely preserved from v1:
  * each model keeps its own Arabic prompt (Tashkeel-350M-v2's differs from the
    instruct models'; swapping them would move the scores on its own)
  * greedy decoding, do_sample=False
  * per-sample inference by default (BATCH_SIZE=1)
  * a failed sample becomes "" and is scored as wrong, rather than dropped

Changed on purpose:
  * scoring goes through models_functions.compute_der_wer
  * predictions are checkpointed to disk, so a crash costs one chunk not a run
    (Glonor-ByT5 took 5h15m for 600 paragraphs the first time)
  * results are written in the sheet's column order, with a pooled Mean row

Batching (--batch-size > 1) is available for speed but left OFF by default:
left-padding a batch can shift a few greedy outputs, which would muddy the
comparison this whole exercise is meant to make clean.
"""

import gc
import os
import re

import pandas as pd
from tqdm.auto import tqdm

from models_functions import compute_der_wer, strip_diacritics

DATASET_ID = "Misraj/SadeedDiac-25"
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./model_cache")
OUT_DIR = os.environ.get("BENCH_OUT_DIR", "./bench_v2_outputs")
SAMPLE_SIZE = int(os.environ.get("BENCH_SAMPLE_SIZE", "600"))   # per domain, as in v1
CHUNK_SIZE = 50

# Prompt used by the instruct/chat models in v1 (Mideum_/Large_/instruct_).
SYS_PROMPT = (
    "You are a specialized Arabic diacritizer. Re-write the given text with "
    "complete Arabic diacritical marks (Tashkeel). Return ONLY the diacritized text."
)
# Tashkeel-350M-v2 was prompted differently in Small_models_v1.py — kept as-is.
TASHKEEL_PROMPT = "قم بتشكيل هذا النص"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

skipped_models = []


# ============================================================
# Benchmark data — MSA / CA split, same convention as v1
# ============================================================
def load_test_sets(sample_size=SAMPLE_SIZE):
    from datasets import load_dataset

    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR)

    msa = [{"ground_truth": x["output"], "raw_input": strip_diacritics(x["input"])}
           for x in ds if "fadel" not in str(x.get("filename", "")).lower()][:sample_size]
    ca = [{"ground_truth": x["output"], "raw_input": strip_diacritics(x["input"])}
          for x in ds if "fadel" in str(x.get("filename", "")).lower()][:sample_size]

    print(f"loaded benchmark: MSA={len(msa)}  CA={len(ca)}")
    if not msa or not ca:
        print("[warn] one domain came back empty — check the 'filename' column exists")
    return msa, ca


# ============================================================
# Loaders
# ============================================================
def _dtype():
    import torch
    return torch.bfloat16 if (torch.cuda.is_available()
                              and torch.cuda.is_bf16_supported()) else torch.float16


def load_model(repo, label, kind, trust_remote_code=False):
    """Returns (handle, tokenizer). `handle` is a pipeline for kind='pipeline_chat'."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              AutoTokenizer, pipeline)

    print(f"Loading [{kind}] {label}  ({repo}) ...")
    try:
        if kind == "pipeline_chat":
            pipe = pipeline("text-generation", model=repo, dtype=_dtype(),
                            device_map="auto", trust_remote_code=trust_remote_code,
                            model_kwargs={"cache_dir": CACHE_DIR})
            return pipe, pipe.tokenizer

        tok = AutoTokenizer.from_pretrained(repo, cache_dir=CACHE_DIR,
                                            trust_remote_code=trust_remote_code)
        if kind == "seq2seq":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForSeq2SeqLM.from_pretrained(
                repo, cache_dir=CACHE_DIR,
                trust_remote_code=trust_remote_code).to(device)
        elif kind == "causal_chat":
            model = AutoModelForCausalLM.from_pretrained(
                repo, device_map="auto", dtype=_dtype(), cache_dir=CACHE_DIR,
                trust_remote_code=trust_remote_code)
        else:
            raise ValueError(f"unknown kind: {kind}")
        model.eval()
        return model, tok
    except Exception as e:
        skipped_models.append((label, f"{type(e).__name__}: {e}"))
        print(f"  SKIPPING {label}: {e}")
        return None, None


# ============================================================
# Inference — one sample, per model kind (mirrors v1 exactly)
# ============================================================
def _predict_one(handle, tok, kind, raw_input, prompt_style):
    import torch

    if kind == "seq2seq":
        inputs = tok(raw_input, return_tensors="pt",
                     max_length=1024, truncation=True).to(handle.device)
        with torch.no_grad():
            out = handle.generate(**inputs, max_length=1024)
        return tok.decode(out[0], skip_special_tokens=True)

    if kind == "pipeline_chat":
        msgs = [{"role": "user", "content": f"{SYS_PROMPT}\n\n{raw_input}"}]
        out = handle(msgs, max_new_tokens=1024, do_sample=False,
                     return_full_text=False)
        text = out[0]["generated_text"]
        return text if isinstance(text, str) else str(text)

    # causal_chat
    if prompt_style == "tashkeel":
        msgs = [{"role": "user", "content": TASHKEEL_PROMPT + ":\n" + raw_input}]
    else:
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": raw_input}]

    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", tokenize=True,
                                  return_dict=True).to(handle.device)
    with torch.no_grad():
        out = handle.generate(**enc, max_new_tokens=1024, do_sample=False)
    in_len = enc["input_ids"].shape[-1]
    return tok.decode(out[0, in_len:], skip_special_tokens=True)


def clean_output(text):
    text = _THINK_RE.sub("", text or "")
    return text.strip().strip('"“”').strip()


def infer(handle, tok, kind, test_set, label, domain, prompt_style, single_line):
    """Generate predictions for one domain, checkpointing as it goes."""
    import torch

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{label}__{domain}".replace("/", "__").replace(" ", "_")
    ckpt = os.path.join(OUT_DIR, tag + "__preds.csv")

    preds = [None] * len(test_set)
    if os.path.exists(ckpt):
        try:
            prev = pd.read_csv(ckpt, header=None, names=["idx", "pred"],
                               keep_default_na=False)
            for _, r in prev.iterrows():
                i = int(r["idx"])
                if 0 <= i < len(preds):
                    preds[i] = str(r["pred"])
            print(f"  resumed {sum(p is not None for p in preds)} cached predictions")
        except Exception as e:
            print(f"  could not read checkpoint ({type(e).__name__}); starting fresh")

    def save():
        done = [(i, preds[i]) for i in range(len(preds)) if preds[i] is not None]
        pd.DataFrame(done).to_csv(ckpt, index=False, header=False)

    todo = [i for i in range(len(test_set)) if preds[i] is None]
    since = 0
    for i in tqdm(todo, desc=f"{label} [{domain}]"):
        try:
            raw = _predict_one(handle, tok, kind, test_set[i]["raw_input"], prompt_style)
            raw = clean_output(raw)
            # v1 kept only the first line for the instruct models, because they
            # tended to append commentary. Preserved per-model, not globally.
            preds[i] = raw.split("\n")[0] if single_line else raw
        except Exception as e:
            print(f"  [warn] {label}: sample {i} failed ({type(e).__name__}: {e}); empty")
            preds[i] = ""
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        since += 1
        if since >= CHUNK_SIZE:
            save(); since = 0
    save()

    refs = [x["ground_truth"] for x in test_set]
    pd.DataFrame({"gold": refs, "pred": preds}).to_csv(
        os.path.join(OUT_DIR, tag + "__pairs.csv"), index=False, header=False)
    return [p or "" for p in preds], refs


# ============================================================
# Orchestration
# ============================================================
def run_registry(registry, results_name):
    """registry: {label: {repo, kind, prompt_style?, single_line?, trust_remote_code?}}"""
    import torch

    os.makedirs(OUT_DIR, exist_ok=True)
    msa, ca = load_test_sets()
    domains = [("MSA (Modern)", msa), ("CA (Classical)", ca)]

    rows = []
    results_path = os.path.join(OUT_DIR, results_name)

    for label, meta in registry.items():
        print(f"\n{'='*62}\n {label}  ({meta['repo']})\n{'='*62}")
        handle, tok = load_model(meta["repo"], label, meta["kind"],
                                 meta.get("trust_remote_code", False))
        if handle is None:
            continue

        pooled_preds, pooled_refs = [], []
        try:
            for domain, test_set in domains:
                if not test_set:
                    print(f"  skipping {domain}: empty")
                    continue
                preds, refs = infer(handle, tok, meta["kind"], test_set, label, domain,
                                    meta.get("prompt_style", "instruct"),
                                    meta.get("single_line", False))
                m = compute_der_wer(preds, refs)
                rows.append({**m, "Model": label, "Domain Track": domain})
                pooled_preds += preds
                pooled_refs += refs
                print(f"  {domain:16s} DER_ce={m['DER_ce']:6.2f} DER_noce={m['DER_noce']:6.2f} "
                      f"WER_ce={m['WER_ce']:6.2f} WER_noce={m['WER_noce']:6.2f}")

            if pooled_refs:
                # Pooled over every paragraph, not the mean of the two track rows —
                # those only coincide when MSA and CA have equal counts.
                m = compute_der_wer(pooled_preds, pooled_refs)
                rows.append({**m, "Model": label, "Domain Track": "Mean (MSA + CA)"})
                print(f"  {'Mean (MSA + CA)':16s} DER_ce={m['DER_ce']:6.2f} "
                      f"DER_noce={m['DER_noce']:6.2f} WER_ce={m['WER_ce']:6.2f} "
                      f"WER_noce={m['WER_noce']:6.2f}")
        except Exception as e:
            skipped_models.append((label, f"{type(e).__name__}: {e}"))
            print(f"  FAILED mid-run: {e}")
        finally:
            del handle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # rewrite after every model so a later crash never loses earlier work
        if rows:
            pd.DataFrame(rows)[["DER_ce", "DER_noce", "WER_ce", "WER_noce",
                                "Model", "Domain Track"]].to_csv(
                results_path, index=False, encoding="utf-8-sig")
            print(f"  saved -> {results_path}")

    print(f"\n{'='*62}\n RESULTS (scored via models_functions.py)\n{'='*62}")
    if rows:
        df = pd.DataFrame(rows)[["DER_ce", "DER_noce", "WER_ce", "WER_noce",
                                 "Model", "Domain Track"]]
        print(df.to_string(index=False))
    else:
        print("no results produced")

    if skipped_models:
        print(f"\n{'='*62}\n SKIPPED / FAILED\n{'='*62}")
        for name, reason in skipped_models:
            print(f"- {name}: {reason}")
    return rows
