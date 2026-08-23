"""Bisher/gemma-3-1b-pt-10k-diacritization on SadeedDiac-25, on Modal.

Ports C:\\Users\\Mahdi\\Downloads\\eval_gemma3_1b_diac_sadeed_(1).ipynb, with the
notebook's own inline scorer REPLACED by the canonical
Evaluation_Functions_Corrected.py, as requested.

WHY THE SCORER SWAP MATTERS. The notebook carries its own private DER/WER
implementation (its cell 13). Swapping in the canonical scorer is what makes this
number comparable with every other Tested_Models_v2 result -- but it also means
the figure here will NOT match whatever the notebook printed when run in Colab.
That is the intended direction: one scorer for the whole project.

WHAT IS KEPT FAITHFUL TO THE NOTEBOOK
  * The adapter-vs-full-model probe. Bisher/gemma-3-1b-pt-10k-diacritization may
    be an adapter-only repo; if it is, the base google/gemma-3-1b-pt is loaded
    and the adapter attached, then merged.
  * The PROMPT-FORMAT SEARCH. This is the substantive part of the notebook and
    the reason it exists. A -pt (pre-trained, not instruction-tuned) checkpoint
    has no canonical prompt, and the wrong one produces a model that continues
    the text instead of diacritising it -- which looks like a terrible model
    rather than a wrong harness. Seven candidate formats are scored on a handful
    of paragraphs and the best is used for the full run.
  * The ranking rule: DER_ce plus a penalty for output length drifting >10% off
    the gold word count. DER alone cannot see insertions, so a format whose stop
    string fails to cut trailing junk scores a deceptively perfect DER.
  * Batched generation at BATCH_SIZE=8 with LEFT padding, greedy, and the
    notebook's dynamic cap of min(input_len * 2.2 + 32, 1024).

The notebook also contained a live HF token in plaintext (cell 16). It is NOT
reproduced here -- this uses the Modal `huggingface-token` secret.

USAGE
    modal run modal_gemma3_1b.py --probe 8 --wait   # smoke: format search + 8 paras
    modal run --detach modal_gemma3_1b.py           # full 1200
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
APP_NAME = "diac-gemma3-1b"
RUN_DIR = "/outputs/gemma3_1b_v2"
CACHE_DIR = "/outputs/gemma3_1b_v2/model_cache"

MODEL_ID = "Bisher/gemma-3-1b-pt-10k-diacritization"
BASE_ID = "google/gemma-3-1b-pt"
LABEL = "gemma-3-1b-pt-10k-diacritization"
BENCH_ID = "Misraj/SadeedDiac-25"

BATCH_SIZE = 8
MAX_NEW_CAP = 1024
GEN_LEN_RATIO = 2.2
PROBE_K = 6           # paragraphs per candidate format

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.15.0",
        "peft>=0.20.0",
        "accelerate", "datasets", "sentencepiece", "pyarabic",
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
def run_gemma3(probe: int = 0, chosen_format: str = ""):
    import sys, csv, re, gc, time, importlib.util
    import torch, pandas as pd
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import HfApi

    spec = importlib.util.spec_from_file_location(
        "evalfns", "/root/Evaluation_Functions_Corrected.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["evalfns"] = ev
    spec.loader.exec_module(ev)
    print("[INFO] scorer: Evaluation_Functions_Corrected.py "
          "(notebook's private scorer deliberately NOT used)", flush=True)

    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_tok:
        from huggingface_hub import login
        login(token=hf_tok)

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = f"{RUN_DIR}/hf_home"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ---------------- data ----------------
    print("Loading SadeedDiac-25...", flush=True)
    ds = load_dataset(BENCH_ID, split="train", cache_dir=CACHE_DIR)
    rows = []
    for x in ds:
        gold = x.get("output", "") or ""
        src = (x.get("filename", "") or "").lower()
        if not gold:
            continue
        rows.append({"gold": gold,
                     "inp": ev.strip_diacritics(x.get("input", "") or ""),
                     "domain": "CA (Classical)" if "fadel" in src else "MSA (Modern)"})
    n_total = len(rows)
    print(f"[INFO] {n_total} paragraphs "
          f"(MSA={sum(r['domain'].startswith('MSA') for r in rows)}, "
          f"CA={sum(r['domain'].startswith('CA') for r in rows)})", flush=True)
    if probe:
        # Half from each domain rather than a head slice. The dataset is ordered,
        # so rows[:8] is entirely one track: the smoke test would never exercise
        # the other one, nor produce its row in the results table.
        half = max(1, probe // 2)
        _msa = [r for r in rows if r["domain"].startswith("MSA")][:half]
        _ca = [r for r in rows if r["domain"].startswith("CA")][:probe - len(_msa)]
        rows = _msa + _ca
        print(f"\n*** PROBE MODE: {len(rows)} paragraphs "
              f"({len(_msa)} MSA + {len(_ca)} CA) ***\n", flush=True)

    # ---------------- model ----------------
    is_adapter = False
    try:
        files = {s.rfilename for s in HfApi().model_info(MODEL_ID, files_metadata=False).siblings}
        is_adapter = ("adapter_config.json" in files
                      and not any(f.endswith(".safetensors") and "adapter" not in f
                                  for f in files))
        print(f"[INFO] repo files: {sorted(files)[:12]}", flush=True)
    except Exception as e:
        print(f"[warn] could not inspect repo ({type(e).__name__}: {e}); assuming full model",
              flush=True)
    print(f"[INFO] adapter-only repo: {is_adapter}", flush=True)

    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok_src = BASE_ID if is_adapter else MODEL_ID
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"      # right padding corrupts batched generation

    load_kwargs = dict(trust_remote_code=True, cache_dir=CACHE_DIR,
                       device_map="auto", dtype=DTYPE, attn_implementation="eager")
    if is_adapter:
        print(f"[INFO] loading base {BASE_ID}, attaching adapter {MODEL_ID}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(BASE_ID, **load_kwargs)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, MODEL_ID, cache_dir=CACHE_DIR)
        model = model.merge_and_unload()
    else:
        print(f"[INFO] loading full model {MODEL_ID}", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        except Exception as e:
            print(f"  CausalLM failed ({type(e).__name__}); trying ImageTextToText", flush=True)
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    print(f"[INFO] params {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"on {model.device}", flush=True)

    has_chat = getattr(tok, "chat_template", None) is not None

    # ---------------- prompt formats ----------------
    INSTRUCTION = (
        "أضِف التشكيل (الحركات) الكامل للنص العربي التالي. "
        "لا تُغيّر الكلمات ولا ترتيبها، ولا تحذف أو تُضِف أيّ كلمة، "
        "وأعِد النصّ المُشكّل فقط دون أي شرح أو مقدمة.\n\nالنص:\n"
    )
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
    _HDR_RE = re.compile(r"^\s*النص(\s+المشكّ?ل)?\s*:\s*")

    def clean_output(text, stops=()):
        raw = _THINK_RE.sub("", text)
        # lstrip FIRST: a leading newline would make split("\n")[0] return "" and
        # silently turn a good prediction into a 100%-error one.
        body = raw.lstrip()
        for s in stops:
            if s and s in body:
                body = body.split(s)[0]
        if not body.strip() and raw.strip():
            body = next((l for l in raw.splitlines() if l.strip()), "")
        body = body.strip().strip('"“”').strip()
        return _HDR_RE.sub("", body).strip()

    FORMATS = {
        "plain_bare":        dict(kind="plain", build=lambda t: t + "\n", stops=["\n"]),
        "plain_blankline":   dict(kind="plain", build=lambda t: t + "\n\n", stops=["\n\n"]),
        "plain_ar_labels":   dict(kind="plain",
                                  build=lambda t: f"النص: {t}\nالنص المشكّل: ",
                                  stops=["\n", "النص:"]),
        "plain_io_labels":   dict(kind="plain",
                                  build=lambda t: f"### Input:\n{t}\n\n### Output:\n",
                                  stops=["###", "\n\n"]),
        "plain_instruction": dict(kind="plain", build=lambda t: INSTRUCTION + t + "\n",
                                  stops=["\n\n"]),
        "gemma_turns":       dict(kind="plain",
                                  build=lambda t: f"<start_of_turn>user\n{INSTRUCTION}{t}"
                                                  f"<end_of_turn>\n<start_of_turn>model\n",
                                  stops=["<end_of_turn>"]),
    }
    if has_chat:
        FORMATS["chat_instruction"] = dict(
            kind="chat", build=lambda t: [{"role": "user", "content": INSTRUCTION + t}], stops=[])
        FORMATS["chat_bare"] = dict(
            kind="chat", build=lambda t: [{"role": "user", "content": t}], stops=[])
    print(f"[INFO] candidate formats: {list(FORMATS)}", flush=True)

    def encode(fmt, texts):
        spec_ = FORMATS[fmt]
        if spec_["kind"] == "chat":
            convs = [spec_["build"](t) for t in texts]
            try:
                enc = tok.apply_chat_template(
                    convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", padding=True, enable_thinking=False)
            except TypeError:
                enc = tok.apply_chat_template(
                    convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", padding=True)
        else:
            enc = tok([spec_["build"](t) for t in texts], return_tensors="pt", padding=True)
        return {k: v.to(model.device) for k, v in enc.items()}

    @torch.no_grad()
    def run_batch(fmt, texts):
        enc = encode(fmt, texts)
        in_len = enc["input_ids"].shape[1]
        max_new = min(int(in_len * GEN_LEN_RATIO) + 32, MAX_NEW_CAP)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
        return [clean_output(d, FORMATS[fmt]["stops"]) for d in dec]

    # ---------------- format search ----------------
    if chosen_format:
        best = chosen_format
        print(f"[INFO] format forced to {best!r}, skipping the search", flush=True)
    else:
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["inp"]))
        pick = [order[int(len(order) * f)] for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)][:PROBE_K]
        p_in = [rows[i]["inp"] for i in pick]
        p_gold = [rows[i]["gold"] for i in pick]

        print(f"\n--- prompt-format search on {len(p_in)} paragraphs ---", flush=True)
        scores = {}
        for fmt in FORMATS:
            try:
                preds = run_batch(fmt, p_in)
                m = ev.compute_der_wer(preds, p_gold)
                ratio = (sum(len(p.split()) for p in preds) /
                         max(1, sum(len(g.split()) for g in p_gold)))
                empty = sum(1 for p in preds if not p.strip())
                # DER cannot see insertions, so penalise length drift > 10%
                rank = m["DER_ce"] + 25 * max(0.0, abs(ratio - 1.0) - 0.10)
                scores[fmt] = dict(rank=rank, **m, word_ratio=round(ratio, 2), empty=empty)
                print(f"  {fmt:20s} DER_ce={m['DER_ce']:6.2f} WER_ce={m['WER_ce']:6.2f} "
                      f"ratio={ratio:4.2f} empty={empty}/{len(p_in)} rank={rank:6.2f}",
                      flush=True)
            except Exception as e:
                print(f"  {fmt:20s} FAILED {type(e).__name__}: {e}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

        if not scores:
            raise SystemExit("FATAL: every candidate prompt format failed")
        best = min(scores, key=lambda k: scores[k]["rank"])
        b = scores[best]
        print(f"\n[INFO] best format: {best}  DER_ce={b['DER_ce']}  "
              f"word_ratio={b['word_ratio']}", flush=True)
        if b["DER_ce"] > 50:
            print("!" * 78, flush=True)
            print("Every candidate is above 50 DER. The prompt format is probably still",
                  flush=True)
            print("wrong -- a working format on this task should be well under 20.", flush=True)
            print("!" * 78, flush=True)
        elif not (0.85 <= b["word_ratio"] <= 1.15):
            print("[warn] word count off by >15%: the model is adding or dropping words, "
                  "which inflates DER through the alignment penalty.", flush=True)

        pd.DataFrame([dict(format=k, **v) for k, v in scores.items()]) \
            .sort_values("rank").to_csv(f"{RUN_DIR}/format_search"
                                        f"{'_probe' + str(probe) if probe else ''}.csv",
                                        index=False)
        vol.commit()

    # ---------------- full pass ----------------
    ck = f"{RUN_DIR}/preds_{LABEL}{'_probe' + str(probe) if probe else ''}.csv"
    done = {}
    if os.path.exists(ck):
        with open(ck, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                done[int(r["idx"])] = r["prediction"]
        print(f"[resume] {len(done)}/{len(rows)} already done", flush=True)

    new_file = not os.path.exists(ck)
    fh = open(ck, "a", encoding="utf-8", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(["idx", "prediction"])

    todo = [i for i in range(len(rows)) if i not in done]
    # length-sorted batching: uniform lengths per batch means far less padding waste
    todo.sort(key=lambda i: len(rows[i]["inp"]))
    t0, n_done = time.time(), 0
    for s in range(0, len(todo), BATCH_SIZE):
        idxs = todo[s:s + BATCH_SIZE]
        try:
            preds = run_batch(best, [rows[i]["inp"] for i in idxs])
        except Exception as e:
            print(f"  [warn] batch at {s} failed ({type(e).__name__}: {e}); empty predictions",
                  flush=True)
            preds = [""] * len(idxs)
        for i, p in zip(idxs, preds):
            done[i] = p
            w.writerow([i, p])
        n_done += len(idxs)
        if (s // BATCH_SIZE) % 5 == 0:
            fh.flush(); vol.commit()
            rate = (time.time() - t0) / max(1, n_done)
            print(f"  {len(done)}/{len(rows)}  {rate:.2f}s/para  "
                  f"~{(len(rows)-len(done))*rate/60:.0f} min left", flush=True)
    fh.close(); vol.commit()
    if n_done:
        print(f"  [THROUGHPUT] {(time.time()-t0)/n_done:.2f} s/paragraph "
              f"over {n_done} new paragraphs", flush=True)

    # ---------------- score ----------------
    preds_all = [done.get(i, "") for i in range(len(rows))]
    golds_all = [r["gold"] for r in rows]
    recs = []
    for dom in ["MSA (Modern)", "CA (Classical)"]:
        idx = [i for i, r in enumerate(rows) if r["domain"] == dom]
        if not idx:
            continue
        m = ev.compute_der_wer([preds_all[i] for i in idx], [golds_all[i] for i in idx])
        recs.append({**m, "Model": LABEL, "Domain Track": dom, "n": len(idx)})
    if recs:
        cols = ["DER_ce", "DER_noce", "WER_ce", "WER_noce"]
        mean = {c: round(sum(r[c] for r in recs) / len(recs), 2) for c in cols}
        recs.append({**mean, "Model": LABEL, "Domain Track": "Mean (MSA + CA)",
                     "n": sum(r["n"] for r in recs)})
    overall = ev.compute_der_wer(preds_all, golds_all)
    recs.append({**overall, "Model": LABEL, "Domain Track": "Overall (all rows)",
                 "n": len(rows)})

    full = pd.DataFrame(recs)[["Model", "Domain Track", "n",
                               "DER_ce", "DER_noce", "WER_ce", "WER_noce"]]
    print(f"\n=== {LABEL} — SadeedDiac-25 (format={best}) ===", flush=True)
    print(full.to_markdown(index=False), flush=True)

    out_csv = f"{RUN_DIR}/gemma3_1b_results{'_probe' + str(probe) if probe else ''}.csv"
    full.to_csv(out_csv, index=False)
    vol.commit()
    print(f"\nResults saved to {out_csv}", flush=True)
    return full.to_csv(index=False)


@app.local_entrypoint()
def main(probe: int = 0, chosen_format: str = "", wait: bool = False):
    """`.spawn()` by default -- `.remote()` gets the run cancelled when the
    launching client is killed (see modal_large_models.py)."""
    if wait:
        print(run_gemma3.remote(probe=probe, chosen_format=chosen_format))
        return
    call = run_gemma3.spawn(probe=probe, chosen_format=chosen_format)
    print(f"\nspawned call id : {call.object_id}")
    print(f"app id          : {app.app_id}")
    print(f"    modal app logs {app.app_id}")
