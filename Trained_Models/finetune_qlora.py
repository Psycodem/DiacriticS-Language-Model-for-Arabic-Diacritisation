# -*- coding: utf-8 -*-
"""finetune_qlora.py — QLoRA fine-tuning for Arabic diacritisation.

Implements the Option A (LoRA/QLoRA) configuration from Confg_Info.md, but
parameterised so the same script covers every candidate base model:

    python finetune_qlora.py --model Qwen/Qwen3.5-4B
    python finetune_qlora.py --model google/gemma-4-E4B-it

Train dataset : Misraj/Sadeed_Tashkeela  (pre-cleaned, pre-chunked — no re-cleaning)
Eval  dataset : 2% held-out split of the above (the SadeedDiac-25 benchmark is
                scored separately by eval_sadeed.py, never touched during training)

Outputs, all under --out-dir:
    final_adapter/          LoRA adapter (few MB — the thing worth keeping)
    train_log.csv           trainer.state.log_history, for the loss plot
    loss_curve.png          train vs eval loss over steps
    run_config.json         every hyperparameter actually used, for the record

Deviation from Confg_Info.md, deliberate: that doc sketches TRL SFTTrainer args
(dataset_text_field / packing / DataCollatorForCompletionOnlyLM). This script
pre-tokenises and masks the prompt with -100 labels instead, driving a plain
Trainer. Same completion-only loss, no dependency on TRL's shifting API.
"""

import argparse
import json
import os
import sys

# The train/held-out partition of Misraj/Sadeed_Tashkeela. eval_sadeed.py must
# use these exact values to rebuild the same boundary — see the comment at the
# holdout split below.
HOLDOUT_SEED = 42
HOLDOUT_FRACTION = 0.02


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HF base model id, e.g. Qwen/Qwen3.5-4B")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: ./runs/<model-slug>)")
    p.add_argument("--cache-dir", default=os.environ.get("HF_CACHE_DIR", "./hf_cache"))

    # data
    p.add_argument("--train-subset", type=int, default=50000,
                   help="training examples to use; 0 = the full ~1M corpus")
    p.add_argument("--eval-subset", type=int, default=1000)
    p.add_argument("--max-seq-length", type=int, default=1024)

    # schedule — defaults track Confg_Info.md COMMON_TRAINING_ARGS
    p.add_argument("--epochs", type=float, default=6.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--no-4bit", action="store_true",
                   help="plain LoRA on a bf16 base instead of 4-bit QLoRA")
    p.add_argument("--resume", action="store_true",
                   help="resume from the newest checkpoint in --out-dir")
    return p.parse_args()


def slug(model_id):
    return model_id.replace("/", "__")


# ============================================================
# Prompt construction
# ============================================================
# Same Arabic instruction the zero-shot benchmark used (LLMs_Last_test.py), so
# the fine-tuned model is trained on the prompt format it is later scored on.
INSTRUCTION = (
    "أضِف التشكيل (الحركات) الكامل للنص العربي التالي. "
    "لا تُغيّر الكلمات ولا ترتيبها، ولا تحذف أو تُضِف أيّ كلمة، "
    "وأعِد النصّ المُشكَّل فقط دون أي شرح أو مقدمة.\n\nالنص:\n"
)


def build_messages(raw_text):
    """Single user turn — Gemma has no system role, and this format loads
    across every candidate model."""
    return [{"role": "user", "content": INSTRUCTION + raw_text}]


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    out_dir = args.out_dir or os.path.join("runs", slug(args.model))
    adapter_dir = os.path.join(out_dir, "final_adapter")
    for d in (out_dir, args.cache_dir):
        os.makedirs(d, exist_ok=True)

    import torch
    from datasets import load_dataset
    from pyarabic import araby
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    if not torch.cuda.is_available():
        sys.exit("FATAL: no CUDA device visible. QLoRA needs a GPU — "
                 "submit this through the sbatch wrapper, don't run it on a login node.")

    bf16_ok = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"CUDA: {torch.cuda.get_device_name(0)} | bf16={bf16_ok}")

    # ---------- tokenizer ----------
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- training data ----------
    print("Loading Misraj/Sadeed_Tashkeela ...")
    ds = load_dataset("Misraj/Sadeed_Tashkeela", split="train", cache_dir=args.cache_dir)

    # Pick the diacritised column the same way the evaluator does: the one
    # actually carrying diacritics, rather than trusting a hard-coded name.
    def diac_ratio(s):
        if not isinstance(s, str) or not s:
            return 0.0
        n = sum(1 for c in s if araby.is_haraka(c) or araby.is_shadda(c) or araby.is_tanwin(c))
        return n / len(s)

    sample = ds.select(range(min(200, len(ds))))
    scores = {}
    for c in ds.column_names:
        try:
            scores[c] = sum(diac_ratio(v) for v in sample[c]) / len(sample)
        except Exception:
            scores[c] = 0.0
    gold_col = max(scores, key=scores.get)
    print(f"gold column: '{gold_col}' (diacritic ratio {scores[gold_col]:.3f})")
    if scores[gold_col] < 0.05:
        sys.exit(f"FATAL: no column looks diacritised (best={gold_col} "
                 f"@ {scores[gold_col]:.3f}). Inspect the dataset before training.")

    def bare(t):
        """Undiacritised model input: strip tashkeel and tatweel."""
        return " ".join(araby.strip_tatweel(araby.strip_tashkeel(t or "")).split())

    def tokenize(ex):
        diac = ex[gold_col]
        msgs = build_messages(bare(diac))
        prompt_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        answer_ids = tok(diac + tok.eos_token, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[: args.max_seq_length]
        # -100 on the prompt => loss is computed only over the diacritised answer.
        labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_seq_length]
        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels}

    # ---------- the held-out partition ----------
    # Split the RAW dataset first, before any filtering or subsetting, so the
    # partition depends only on (row order, HOLDOUT_SEED, HOLDOUT_FRACTION).
    # eval_sadeed.py recreates it with the same two constants to build its "Test"
    # set, which is the only thing making that set genuinely unseen. Filtering
    # afterwards is safe — it drops rows but never moves one across the boundary.
    #
    # Change these and you must change them in eval_sadeed.py too, or "Test"
    # silently starts scoring paragraphs the model was fitted on.
    holdout = ds.train_test_split(test_size=HOLDOUT_FRACTION, seed=HOLDOUT_SEED)
    train_pool, eval_pool = holdout["train"], holdout["test"]
    print(f"holdout: train_pool={len(train_pool)}  eval_pool={len(eval_pool)} "
          f"(fraction={HOLDOUT_FRACTION}, seed={HOLDOUT_SEED})")

    keep_row = lambda ex: bool(ex[gold_col]) and bool(bare(ex[gold_col]))  # noqa: E731
    train_pool = train_pool.filter(keep_row)
    eval_pool = eval_pool.filter(keep_row)

    # Subset BEFORE tokenising — tokenising 1M rows to then discard 95% is waste.
    if args.train_subset and args.train_subset < len(train_pool):
        train_pool = train_pool.shuffle(seed=args.seed).select(range(args.train_subset))
    if args.eval_subset and args.eval_subset < len(eval_pool):
        eval_pool = eval_pool.shuffle(seed=args.seed).select(range(args.eval_subset))

    tok_kwargs = dict(desc="tokenising", num_proc=max(1, (os.cpu_count() or 2) // 2))
    train_ds = train_pool.map(tokenize, remove_columns=train_pool.column_names, **tok_kwargs)
    eval_ds = eval_pool.map(tokenize, remove_columns=eval_pool.column_names, **tok_kwargs)

    has_target = lambda ex: any(l != -100 for l in ex["labels"])  # noqa: E731
    train_ds = train_ds.filter(has_target)
    eval_ds = eval_ds.filter(has_target)
    print(f"train={len(train_ds)}  eval={len(eval_ds)}")

    # ---------- base model ----------
    load_kwargs = dict(
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    if not args.no_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"Loading base model {args.model} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except Exception as e:
        print(f"  CausalLM failed ({type(e).__name__}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)

    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    # All seven projections, per Confg_Info.md — the old script only adapted
    # q/k/v/o, which leaves the MLP untouched and measurably underfits.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    target_modules = [m for m in target_modules if m in present]
    if not target_modules:
        sys.exit(f"FATAL: none of the expected projection modules exist in {args.model}. "
                 f"Inspect model.named_modules() and set target_modules by hand.")
    print(f"LoRA target modules: {target_modules}")

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    ))
    model.print_trainable_parameters()

    # ---------- train ----------
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        optim="adamw_torch" if args.no_4bit else "paged_adamw_8bit",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "target_modules": target_modules,
                   "gold_col": gold_col, "bf16": bf16_ok}, f, indent=2)

    trainer.train(resume_from_checkpoint=args.resume or None)

    # ---------- save ----------
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"adapter saved -> {adapter_dir}")

    import pandas as pd
    log = pd.DataFrame(trainer.state.log_history)
    log_path = os.path.join(out_dir, "train_log.csv")
    log.to_csv(log_path, index=False)
    print(f"training log saved -> {log_path}")

    # ---------- loss curve ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        if "loss" in log:
            tr = log.dropna(subset=["loss"])
            ax.plot(tr["step"], tr["loss"], label="train loss")
        if "eval_loss" in log:
            ev = log.dropna(subset=["eval_loss"])
            ax.plot(ev["step"], ev["eval_loss"], marker="o", label="eval loss")
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.set_title(f"{args.model} — QLoRA diacritisation")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
        print(f"loss curve saved -> {os.path.join(out_dir, 'loss_curve.png')}")
    except Exception as e:
        print(f"[warn] could not plot loss curve: {type(e).__name__}: {e}")

    print("\nDone. Next: score it on the benchmark with\n"
          f"  python eval_sadeed.py --model {args.model} --adapter {adapter_dir}")


if __name__ == "__main__":
    main()
