# -*- coding: utf-8 -*-
"""eval_sadeed.py — score a model on every split the Fine-Tuning Models tab wants.

The sheet asks for three result rows per fine-tuning run:

    Train           a sample of Misraj/Sadeed_Tashkeela's train split (data the
                    model was fitted on — shows how well it absorbed the task)
    Test            Misraj/Sadeed_Tashkeela's test split (held out, same domain)
    SadeedDiac-25   the contamination-controlled benchmark, broken into
                    MSA / CA / Mean so it lines up with the zero-shot baselines

Generation and metrics match LLMs_Last_test.py, the script that produced the
recorded baselines, so a fine-tuned number is comparable to the row above it.
`--adapter` is the addition: it loads a LoRA/QLoRA adapter onto the base model.

    # everything, fine-tuned adapter
    python eval_sadeed.py --model google/gemma-4-E4B-it \
        --adapter runs/google__gemma-4-E4B-it/final_adapter \
        --label gemma-4-E4B-it-QLoRA --method QLoRA

    # just the benchmark, base model (re-score a baseline)
    python eval_sadeed.py --model Qwen/Qwen3.5-4B --splits sadeed

Writes into --out-dir, per split:
    <slug>__<split>__preds.csv    resumable generation checkpoint
    <slug>__<split>__pairs.csv    gold,pred for every paragraph
    <slug>__results.csv           all rows, in the Fine-Tuning tab's column order

Metrics come from Models_Functions.py (same directory) so there is exactly one
implementation to audit. The older Mideum_/Small_/Large_models_v1.py scripts
carried an earlier inline copy without numeral normalisation or digit-token
exclusion — baselines produced by those are NOT comparable to these. Re-score
them with the *_models_v2.py scripts in "Models to test" first.
"""

import argparse
import gc
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Models_Functions import compute_der_wer, strip_diacritics  # noqa: E402

BENCH_ID = "Misraj/SadeedDiac-25"
TRAIN_ID = "Misraj/Sadeed_Tashkeela"
MAX_NEW_CAP = 1024

# MUST match finetune_qlora.py. These two constants are the only thing that makes
# the "Test" split genuinely unseen by the fine-tuned model.
HOLDOUT_SEED = 42
HOLDOUT_FRACTION = 0.02

SPLIT_LABELS = {"train": "Train", "test": "Test", "sadeed": "SadeedDiac-25"}

INSTRUCTION = (
    "أضِف التشكيل (الحركات) الكامل للنص العربي التالي. "
    "لا تُغيّر الكلمات ولا ترتيبها، ولا تحذف أو تُضِف أيّ كلمة، "
    "وأعِد النصّ المُشكَّل فقط دون أي شرح أو مقدمة.\n\nالنص:\n"
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF base model id")
    p.add_argument("--adapter", default=None, help="path to a LoRA adapter dir")
    p.add_argument("--label", default=None,
                   help="name written into the Model column (default: --model)")
    p.add_argument("--method", default="",
                   help="LoRA / QLoRA / zero-shot — stamped into the Method column")
    p.add_argument("--splits", default="train,test,sadeed",
                   help="comma list of: train, test, sadeed")
    p.add_argument("--out-dir", default="sadeed_outputs")
    p.add_argument("--cache-dir", default=os.environ.get("HF_CACHE_DIR", "./hf_cache"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--train-samples", type=int, default=500,
                   help="paragraphs drawn from the train split")
    p.add_argument("--test-samples", type=int, default=500,
                   help="paragraphs drawn from the test split")
    p.add_argument("--limit", type=int, default=0,
                   help="quick test: cap the benchmark at N paragraphs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-4bit", action="store_true",
                   help="load the base in 4-bit (match QLoRA training conditions)")
    return p.parse_args()


def build_messages(text):
    return [{"role": "user", "content": INSTRUCTION + text}]


def clean_output(text):
    text = _THINK_RE.sub("", text)
    text = text.strip().strip('"“”').strip()
    text = re.sub(r"^\s*النص(\s+المشكّ?ل)?\s*:\s*", "", text)
    return text.strip()


def _pick_gold_column(df):
    """The column actually carrying diacritics, rather than a trusted name."""
    from pyarabic import araby

    def ratio(s):
        if not isinstance(s, str) or not s:
            return 0.0
        n = sum(1 for c in s if araby.is_haraka(c) or araby.is_shadda(c) or araby.is_tanwin(c))
        return n / len(s)

    scores = {c: df[c].astype(str).head(50).map(ratio).mean() for c in df.columns}
    col = max(scores, key=scores.get)
    return col, scores[col]


# ============================================================
# Data loading — one function per split
# ============================================================
def load_benchmark(cache_dir, limit=0):
    """SadeedDiac-25, with the MSA/CA domain flags."""
    from datasets import load_dataset
    from pyarabic import araby

    ds = load_dataset(BENCH_ID, cache_dir=cache_dir)
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    gold_col, score = _pick_gold_column(df)
    print(f"  [sadeed] split='{split}' rows={len(df)} gold='{gold_col}' ({score:.3f})")

    # Prefer an explicit source-file column — the Fadel-derived paragraphs are the
    # Classical Arabic half. Fall back to the documented 600/600 ordering.
    meta_col = next((c for c in df.columns
                     if c.lower() in ("filename", "file", "source", "origin", "domain")), None)
    if meta_col is not None and df[meta_col].astype(str).str.contains("fadel", case=False).any():
        is_ca = df[meta_col].astype(str).str.contains("fadel", case=False).tolist()
        print(f"  [sadeed] domain split from '{meta_col}': "
              f"CA={sum(is_ca)} MSA={len(is_ca) - sum(is_ca)}")
    else:
        half = len(df) // 2
        is_ca = [i >= half for i in range(len(df))]
        print(f"  [sadeed] [warn] no source column — falling back to first-half MSA / "
              f"second-half CA ({half}/{len(df) - half})")

    gold = [araby.strip_tatweel(g) for g in df[gold_col].astype(str).tolist()]
    inputs = [araby.strip_tashkeel(g) for g in gold]
    if limit:
        gold, inputs, is_ca = gold[:limit], inputs[:limit], is_ca[:limit]
    return gold, inputs, is_ca


def load_tashkeela(cache_dir, which, n, seed):
    """A sample of Misraj/Sadeed_Tashkeela's train or held-out side.

    "Test" MUST come from the same partition finetune_qlora.py held out, or it
    scores paragraphs the model was fitted on and stops meaning anything. Both
    files derive it from HOLDOUT_SEED / HOLDOUT_FRACTION applied to the RAW
    dataset before any filtering — keep the two in sync.

    A published 'test' split, if the dataset ever ships one, wins over the carve.

    No MSA/CA breakdown here — Tashkeela is not domain-tagged, so these two
    splits produce a single row each, which is what the sheet asks for.
    """
    from datasets import load_dataset
    from pyarabic import araby

    ds = load_dataset(TRAIN_ID, cache_dir=cache_dir)

    if which == "test" and "test" in ds:
        sub = ds["test"]
        print(f"  [test] using the dataset's published 'test' split")
    else:
        holdout = ds["train"].train_test_split(test_size=HOLDOUT_FRACTION,
                                               seed=HOLDOUT_SEED)
        sub = holdout["test" if which == "test" else "train"]
        print(f"  [{which}] from the shared holdout partition "
              f"(fraction={HOLDOUT_FRACTION}, seed={HOLDOUT_SEED}) -> {len(sub)} rows")

    if n and n < len(sub):
        sub = sub.shuffle(seed=seed).select(range(n))

    df = sub.to_pandas()
    gold_col, score = _pick_gold_column(df)
    print(f"  [{which}] rows={len(df)} gold='{gold_col}' ({score:.3f})")

    gold = [araby.strip_tatweel(g) for g in df[gold_col].astype(str).tolist()]
    inputs = [araby.strip_tashkeel(g) for g in gold]
    return gold, inputs, [False] * len(gold)


# ============================================================
# Model + generation
# ============================================================
def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                        cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    load_kwargs = dict(trust_remote_code=True, cache_dir=args.cache_dir,
                       device_map="auto", dtype=torch.bfloat16)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except Exception as e:
        print(f"  CausalLM failed ({type(e).__name__}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)

    if args.adapter:
        from peft import PeftModel
        print(f"  attaching adapter {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # fold in — faster generation
    model.eval()
    return model, tok


def generate(model, tok, inputs_texts, args, tag):
    """Greedy generation with a resumable on-disk checkpoint."""
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    ckpt = os.path.join(args.out_dir, tag + "__preds.csv")
    preds = [None] * len(inputs_texts)

    if os.path.exists(ckpt):
        try:
            prev = pd.read_csv(ckpt, header=None, names=["idx", "pred"], keep_default_na=False)
            for _, r in prev.iterrows():
                i = int(r["idx"])
                if 0 <= i < len(preds):
                    preds[i] = str(r["pred"])
            print(f"  resumed {sum(p is not None for p in preds)} cached predictions")
        except Exception as e:
            print(f"  could not read checkpoint ({type(e).__name__}); starting fresh")

    todo = [i for i in range(len(inputs_texts)) if preds[i] is None]
    if not todo:
        print("  all predictions cached")
        return preds

    def save_ckpt():
        done = [(i, preds[i]) for i in range(len(preds)) if preds[i] is not None]
        pd.DataFrame(done).to_csv(ckpt, index=False, header=False)

    order = sorted(todo, key=lambda i: len(inputs_texts[i]))   # length-bucketed batches
    since_save = 0
    for start in tqdm(range(0, len(order), args.batch_size), desc=tag):
        idxs = order[start:start + args.batch_size]
        try:
            convs = [build_messages(inputs_texts[i]) for i in idxs]
            try:
                enc = tok.apply_chat_template(
                    convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", padding=True, enable_thinking=False)
            except TypeError:
                enc = tok.apply_chat_template(
                    convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            in_len = enc["input_ids"].shape[1]
            max_new = min(int(in_len * 2.2) + 32, MAX_NEW_CAP)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     num_beams=1, pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            for j, i in enumerate(idxs):
                preds[i] = clean_output(dec[j])
        except Exception as e:
            print(f"  batch @ {start} failed ({type(e).__name__}: {e}); leaving blank")
            for i in idxs:
                if preds[i] is None:
                    preds[i] = ""
            gc.collect(); torch.cuda.empty_cache()

        since_save += len(idxs)
        if since_save >= args.chunk_size:
            save_ckpt(); since_save = 0

    save_ckpt()
    return preds


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd

    wanted = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SPLIT_LABELS]
    if unknown:
        sys.exit(f"FATAL: unknown split(s) {unknown}; choose from {list(SPLIT_LABELS)}")

    label = args.label or args.model
    slug = (label if args.adapter else args.model).replace("/", "__")
    method = args.method or ("QLoRA" if args.load_4bit and args.adapter
                             else "LoRA" if args.adapter else "zero-shot")

    model, tok = load_model(args)

    rows = []
    for split in wanted:
        print(f"\n{'='*62}\n {SPLIT_LABELS[split]}\n{'='*62}")
        if split == "sadeed":
            golds, inputs, is_ca = load_benchmark(args.cache_dir, args.limit)
        else:
            golds, inputs, is_ca = load_tashkeela(
                args.cache_dir, split,
                args.train_samples if split == "train" else args.test_samples,
                args.seed)

        if not golds:
            print(f"  no data for {split}; skipping")
            continue

        tag = f"{slug}__{split}"
        preds = generate(model, tok, inputs, args, tag)
        preds = [p if p is not None else "" for p in preds]

        pd.DataFrame({"gold": golds, "pred": preds}).to_csv(
            os.path.join(args.out_dir, tag + "__pairs.csv"), index=False, header=False)

        blank = sum(1 for p in preds if not p.strip())
        if blank:
            print(f"  [warn] {blank}/{len(preds)} predictions empty — they score as "
                  f"fully wrong and will inflate the reported error rates")

        def emit(track, idxs):
            m = compute_der_wer([preds[i] for i in idxs], [golds[i] for i in idxs])
            rows.append({**m, "Results": SPLIT_LABELS[split], "Domain Track": track,
                         "Model": label, "Method": method, "n": len(idxs)})
            print(f"  {track:16s} n={len(idxs):5d}  DER_ce={m['DER_ce']:6.2f}  "
                  f"DER_noce={m['DER_noce']:6.2f}  WER_ce={m['WER_ce']:6.2f}  "
                  f"WER_noce={m['WER_noce']:6.2f}")

        if split == "sadeed":
            msa = [i for i in range(len(golds)) if not is_ca[i]]
            ca = [i for i in range(len(golds)) if is_ca[i]]
            if msa:
                emit("MSA (Modern)", msa)
            if ca:
                emit("CA (Classical)", ca)
            # Pooled across all paragraphs, not the average of the two track rows —
            # those only coincide when MSA and CA have equal counts.
            emit("Mean (MSA + CA)", list(range(len(golds))))
        else:
            emit("-", list(range(len(golds))))

        # rewrite after each split so a crash later never loses earlier work
        pd.DataFrame(rows)[["DER_ce", "DER_noce", "WER_ce", "WER_noce",
                            "Results", "Domain Track", "Model", "Method", "n"]].to_csv(
            os.path.join(args.out_dir, slug + "__results.csv"),
            index=False, encoding="utf-8-sig")

    if not rows:
        sys.exit("No results produced.")

    df = pd.DataFrame(rows)[["DER_ce", "DER_noce", "WER_ce", "WER_noce",
                             "Results", "Domain Track", "Model", "Method", "n"]]
    res_path = os.path.join(args.out_dir, slug + "__results.csv")
    df.to_csv(res_path, index=False, encoding="utf-8-sig")
    print(f"\n{'='*62}\nresults -> {res_path}\n{'='*62}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
