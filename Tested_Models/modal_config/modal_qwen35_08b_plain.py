"""Plain (un-fine-tuned) Qwen/Qwen3.5-0.8B on SadeedDiac-25, on Modal.

This is the ZERO-SHOT BASELINE for the 0.8B model: the stock checkpoint straight
from the Hub, no training of any kind. It exists so the 0.8B full-fine-tune has
a before-picture, the same way the other Tested_Models_v2 entries do.

Prompting, decoding and post-processing are lifted from the 0.8B full-fine-tune
notebook (02_Qwen3.5-0.8B_Full-FineTune_Diacritization.ipynb) so that this
baseline and that fine-tune differ ONLY in the weights. Scoring is the canonical
Evaluation_Functions_Corrected.py.

THREE DETAILS CARRIED OVER FROM THAT NOTEBOOK, each of which silently ruins the
numbers if dropped:

  1. EOS IDS. The Qwen3.5-0.8B repo ships no generation_config.json, and its
     config EOS is <|endoftext|> (248044), NOT the <|im_end|> (248046) that ends
     a chat turn. Left alone, every generation runs to the cap and every tail is
     scored as a deletion. generation_eos_ids() adds both.

  2. LEFT padding AND LEFT truncation. Right padding corrupts batched
     generation. Right truncation is subtler and worse: it cuts the END of the
     rendered chat template -- the "<|im_start|>assistant" part that tells the
     model to answer -- so the model continues the user's text instead of
     diacritising it, and the row reads as a bad prediction rather than a bug.

  3. enable_thinking=False plus <think>-block stripping. Qwen3.5 is a reasoning
     model; without this the reasoning text is scored as if it were the answer.

The prompt is byte-identical to the one used by the 4B LoRA runs, which is what
makes all the Qwen numbers in this project comparable to each other.

USAGE
    modal run modal_qwen35_08b_plain.py --probe 8 --wait   # smoke test
    modal run --detach modal_qwen35_08b_plain.py           # full 1200
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
APP_NAME = "diac-qwen35-08b-plain"
RUN_DIR = "/outputs/qwen35_08b_plain_v2"
CACHE_DIR = "/outputs/qwen35_08b_plain_v2/model_cache"

MODEL_ID = "Qwen/Qwen3.5-0.8B"        # stock weights, NOT fine-tuned
LABEL = "Qwen3.5-0.8B (plain, zero-shot)"
BENCH_ID = "Misraj/SadeedDiac-25"
BENCH_SPLIT = "train"
# Pinned revision, same as the fine-tune notebook: an unpinned benchmark can be
# updated upstream and silently change what "the 1200" means between runs.
BENCH_REVISION = "aa311213e44e4cab6cc3f2848daacd753adc1ce1"

MAX_SEQ_LENGTH = 1024
INFER_BATCH_SIZE = 8
MAX_NEW_CAP = 1024
GEN_LEN_RATIO = 2.2

# Byte-identical to the 4B run's prompt. This is what makes the two benchmark
# numbers comparable.
SYSTEM_PROMPT = (
    "أنت نظام متخصص في التشكيل الآلي للنصوص العربية. "
    "مهمتك إضافة الحركات (التشكيل) الصحيحة إلى النص العربي المُدخل دون تغيير الكلمات أو ترتيبها، "
    "مع مراعاة السياق النحوي والصرفي الكامل للجملة."
)
# Used only if the checkpoint has no chat template (e.g. a -Base variant).
PROMPT_TMPL = SYSTEM_PROMPT + "\n\nالنص:\n{inp}\n\nالنص المشكل:\n"

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
def run_qwen_plain(probe: int = 0):
    import sys, csv, re, time, importlib.util
    import torch, pandas as pd
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

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

    # ---------------- data ----------------
    print(f"Loading {BENCH_ID} @ {BENCH_REVISION[:8]} ...", flush=True)
    try:
        ds = load_dataset(BENCH_ID, split=BENCH_SPLIT, revision=BENCH_REVISION,
                          cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"[warn] pinned revision unavailable ({type(e).__name__}); using default: {e}",
              flush=True)
        ds = load_dataset(BENCH_ID, split=BENCH_SPLIT, cache_dir=CACHE_DIR)

    rows = []
    for x in ds:
        gold = x.get("output", "") or ""
        if not gold:
            continue
        src = (x.get("filename", "") or "").lower()
        rows.append({"gold": gold,
                     "inp": ev.strip_diacritics(x.get("input", "") or ""),
                     "domain": "CA (Classical)" if "fadel" in src else "MSA (Modern)"})
    print(f"[INFO] {len(rows)} paragraphs "
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
    print(f"[INFO] loading PLAIN {MODEL_ID} (no fine-tuning)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, dtype=dtype, device_map="auto")
    model.eval()
    print(f"[INFO] params {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| dtype {dtype} | device {model.device}", flush=True)

    chat_mode = getattr(tokenizer, "chat_template", None) is not None
    supports_system = False
    if chat_mode:
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            supports_system = True
        except Exception:
            print("[INFO] no system role; folding the instruction into the user turn",
                  flush=True)
    else:
        print("[WARN] no chat template; falling back to the plain PROMPT_TMPL", flush=True)
    print(f"[INFO] chat_mode={chat_mode} supports_system={supports_system}", flush=True)

    def build_messages(user_text):
        if supports_system:
            return [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}]
        return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"}]

    def render_prompt(user_text):
        if not chat_mode:
            return PROMPT_TMPL.format(inp=user_text)
        try:
            return tokenizer.apply_chat_template(
                build_messages(user_text), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                build_messages(user_text), tokenize=False, add_generation_prompt=True)

    _THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.DOTALL)
    _ECHOED_HEADER_RE = re.compile(r"^\s*النص المشكل\s*:?\s*", re.MULTILINE)

    def clean_output(text):
        out = _THINK_RE.sub("", text.strip()).strip()
        if not chat_mode:
            # a base model has no turn-end token and generates to the cap forever,
            # usually by inventing a new document -- cut at the first blank line
            out = _ECHOED_HEADER_RE.sub("", out).strip().split("\n\n")[0].strip()
        return out

    def generation_eos_ids():
        """The repo ships no generation_config.json and its config EOS is
        <|endoftext|>, not the <|im_end|> that ends a chat turn. Without both,
        every generation runs to the cap and every tail scores as a deletion."""
        ids = {tokenizer.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid >= 0:
                ids.add(tid)
        return sorted(i for i in ids if i is not None)

    eos_ids = generation_eos_ids()
    print(f"[INFO] eos ids: {eos_ids}", flush=True)

    # ---------------- inference ----------------
    ck = f"{RUN_DIR}/preds_qwen35_08b_plain{'_probe' + str(probe) if probe else ''}.csv"
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

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    todo = [i for i in range(len(rows)) if i not in done]
    todo.sort(key=lambda i: len(rows[i]["inp"]))   # uniform batches, less padding waste
    t0, n_new, truncated = time.time(), 0, 0

    for s in range(0, len(todo), INFER_BATCH_SIZE):
        idxs = todo[s:s + INFER_BATCH_SIZE]
        try:
            prompts = [render_prompt(rows[i]["inp"]) for i in idxs]
            # add_special_tokens=False: the chat template already emitted every
            # special token this prompt needs, as text
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=MAX_SEQ_LENGTH,
                            add_special_tokens=False).to(model.device)
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=min(int(in_len * GEN_LEN_RATIO) + 32, MAX_NEW_CAP),
                    do_sample=False, num_beams=1,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids)
            gen = out[:, in_len:]
            # a row that stopped early is right-padded with pad_token_id, which is
            # in eos_ids -- so "contains no eos id" is exactly "ran to the cap"
            eos_t = torch.tensor(eos_ids, device=gen.device)
            truncated += int((~(gen.unsqueeze(-1) == eos_t).any(-1).any(-1)).sum().item())
            preds = [clean_output(t) for t in tokenizer.batch_decode(gen,
                                                                     skip_special_tokens=True)]
        except Exception as e:
            print(f"  [warn] batch at {s} failed ({type(e).__name__}: {e}); empty predictions",
                  flush=True)
            preds = [""] * len(idxs)

        for i, p in zip(idxs, preds):
            done[i] = p
            w.writerow([i, p])
        n_new += len(idxs)
        if (s // INFER_BATCH_SIZE) % 5 == 0:
            fh.flush(); vol.commit()
            rate = (time.time() - t0) / max(1, n_new)
            print(f"  {len(done)}/{len(rows)}  {rate:.2f}s/para  "
                  f"~{(len(rows)-len(done))*rate/60:.0f} min left", flush=True)

    fh.close(); vol.commit()
    if n_new:
        print(f"  [THROUGHPUT] {(time.time()-t0)/n_new:.2f} s/paragraph over {n_new} new",
              flush=True)
    print(f"[INFO] ran to the generation cap (no EOS): {truncated}/{n_new}", flush=True)

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
    print(f"\n=== {LABEL} — SadeedDiac-25 ===", flush=True)
    print(full.to_markdown(index=False), flush=True)

    out_csv = f"{RUN_DIR}/qwen35_08b_plain_results{'_probe' + str(probe) if probe else ''}.csv"
    full.to_csv(out_csv, index=False)
    vol.commit()
    print(f"\nResults saved to {out_csv}", flush=True)
    return full.to_csv(index=False)


@app.local_entrypoint()
def main(probe: int = 0, wait: bool = False):
    """`.spawn()` by default -- `.remote()` gets the run cancelled when the
    launching client is killed (see modal_large_models.py)."""
    if wait:
        print(run_qwen_plain.remote(probe=probe))
        return
    call = run_qwen_plain.spawn(probe=probe)
    print(f"\nspawned call id : {call.object_id}")
    print(f"app id          : {app.app_id}")
    print(f"    modal app logs {app.app_id}")
