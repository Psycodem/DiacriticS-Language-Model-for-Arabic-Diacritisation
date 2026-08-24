"""
Phase 2+3 (Qwen arm, the plan's MAIN test): train the double-pass Qwen3.5-0.8B char-tagger.

Warm-started from the existing 2.58-DER checkpoint (see model_tagging_qwen.py's docstring for
why), so this is closer to fine-tuning a new head onto an already-task-adapted body than training
from scratch -- the ../Train-Tagging/ AraBERT control needed neither a warm start nor a
double-pass input and still took 3,000 steps to converge; expect this arm to need fewer, not
more, given the head start.

Usage:
    python train_tagging_qwen.py --data-dir $SCRATCH/tagging_data_qwen --labels-path labels.json \
        --out $SCRATCH/runs/qwen-tagging
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import LabelVocab  # noqa: E402
from model_tagging_qwen import (QwenCharTagger, assert_fast_path,  # noqa: E402
                                  assert_master_dtype)

MODEL_ID = os.environ.get("MODEL_ID", "/scratch/runs/qwen3.5-0.8b-fullft/final_model")
RANDOM_SEED = 42


class DataCollatorForCharTagging:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_tok = max(len(f["input_ids"]) for f in features)
        max_chr = max(len(f["char_labels"]) for f in features)

        def pad(seq, length, value):
            return list(seq) + [value] * (length - len(seq))

        return {
            "input_ids": torch.tensor([pad(f["input_ids"], max_tok, self.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], max_tok, 0) for f in features]),
            "char_to_token": torch.tensor([pad(f["char_to_token"], max_chr, 0) for f in features]),
            "char_ids": torch.tensor([pad(f["char_ids"], max_chr, 0) for f in features]),
            "char_intra_pos": torch.tensor([pad(f["char_intra_pos"], max_chr, 0) for f in features]),
            "labels": torch.tensor([pad(f["char_labels"], max_chr, -100) for f in features]),
        }


def _length_grouping_kwargs(training_arguments_cls) -> dict:
    fields = {f.name for f in dataclasses.fields(training_arguments_cls)}
    if "train_sampling_strategy" in fields:
        kw = {"train_sampling_strategy": "group_by_length"}
    elif "group_by_length" in fields:
        kw = {"group_by_length": True}
    else:
        return {}
    if "length_column_name" in fields:
        kw["length_column_name"] = "length"
    kw["remove_unused_columns"] = False
    return kw


def _make_length_grouped_trainer(Trainer):
    """See memory/[[transformers5-length-sampler-hang]] and ../Train-Tagging/train_tagging.py's
    identical fix -- materializing the length column is not optional at this corpus size."""
    try:
        from transformers.trainer_pt_utils import LengthGroupedSampler
    except Exception as e:
        print(f"[WARN] could not import LengthGroupedSampler ({type(e).__name__}); "
              f"falling back to stock Trainer")
        return Trainer

    from torch.utils.data import SequentialSampler

    class _TaggingTrainer(Trainer):
        def __init__(self, *args, train_lengths=None, **kwargs):
            self._train_lengths = train_lengths
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, *a, **kw):
            grouped = (getattr(self.args, "train_sampling_strategy", None) == "group_by_length"
                       or getattr(self.args, "group_by_length", False))
            if self._train_lengths is None or not grouped:
                return super()._get_train_sampler(*a, **kw)
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=self._train_lengths,
            )

        def _get_eval_sampler(self, eval_dataset=None, *a, **kw):
            if eval_dataset is None:
                return None
            return SequentialSampler(eval_dataset) if self.args.world_size <= 1 else None

    return _TaggingTrainer


def _make_loss_sanity_callback():
    from transformers import TrainerCallback

    class LossSanityCallback(TrainerCallback):
        def __init__(self, num_labels):
            self.expected = torch.log(torch.tensor(float(num_labels))).item()
            self.checked = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.checked or logs is None or "loss" not in logs:
                return
            self.checked = True
            loss = logs["loss"]
            print(f"[SANITY] step {state.global_step} loss={loss:.4f} "
                  f"(random-head reference ~{self.expected:.4f} -- warm-started body may differ)")
            if loss != loss or loss in (float("inf"), float("-inf")):
                raise RuntimeError(f"first training loss is not finite ({loss}) -- stop.")
            if loss < 0.05:
                raise RuntimeError(
                    f"loss={loss:.4f} at the first logged step is suspiciously close to 0 -- "
                    f"the label mask is very likely inverted."
                )

    return LossSanityCallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels-path", default="labels.json")
    ap.add_argument("--out", required=True)
    # max_steps=0 means "full epoch": num_train_epochs drives the cosine schedule instead, same
    # convention as Train-Qwen-0.8B-Full-FT/train_full_ft_qwen35_08b.py's NUM_TRAIN_EPOCHS/
    # MAX_STEPS=-1 pair. A positive value here still overrides it, for smoke tests / rungs.
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "0")))
    ap.add_argument("--num-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")))
    # 16 x 4 = effective batch 64, IDENTICAL to the 2,000-step run -- same optimization geometry,
    # so eval_loss stays comparable. Only the PER-DEVICE batch changed (8->16): the 2,000-step
    # run used just 33.2/80GB of the H100 (measured), leaving real headroom. Doubling the
    # per-device batch does more work per kernel launch without changing what the optimizer sees
    # per step.
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "16")))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LEARNING_RATE", "2e-5")))
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    # ~20 evals across a full epoch (N_EVALS=20 convention from the generative script) rather
    # than the 2,000-step run's every-100-steps (which would be ~800 evals over a full epoch --
    # each eval costs ~11-17s, so that would burn 2.5-4h on eval alone).
    ap.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "800")))
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")))
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(RANDOM_SEED)
    assert_fast_path()  # THE gate: fail before the GPU burns money on the slow chunked-scan path

    vocab = LabelVocab.load(args.labels_path)
    train_ds = load_from_disk(os.path.join(args.data_dir, "train"))
    dev_ds = load_from_disk(os.path.join(args.data_dir, "dev"))

    if args.smoke_test:
        args.max_steps = 60
        train_ds = train_ds.shuffle(seed=RANDOM_SEED).select(range(min(12000, len(train_ds))))

    print(f"[INFO] train={len(train_ds):,} dev={len(dev_ds):,} labels={len(vocab)}")

    t0 = time.time()
    train_lengths = list(train_ds["length"])
    print(f"[INFO] materialized {len(train_lengths):,} lengths in {time.time() - t0:.1f}s")

    model = QwenCharTagger(MODEL_ID, num_labels=len(vocab))
    assert_master_dtype(model)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    collator = DataCollatorForCharTagging(pad_token_id=tokenizer.pad_token_id)

    grouping_kwargs = _length_grouping_kwargs(TrainingArguments)
    # max_steps=0 -> disabled (-1), num_train_epochs drives the schedule instead. HF Trainer
    # requires max_steps to be -1, not 0, to fall back to num_train_epochs.
    effective_max_steps = args.max_steps if args.max_steps else -1
    training_args = TrainingArguments(
        output_dir=args.out,
        max_steps=effective_max_steps,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=args.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=False,  # off, same call as the generative 0.8B run at this size
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        seed=RANDOM_SEED,
        report_to=[],
        # Kept because prefetching is strictly correct, NOT because it fixed a 3x slowdown --
        # there was never a 3x slowdown. The "4.5 -> 4.3 -> 3.5 s/step" ladder that motivated
        # these was measured on 60-step smoke tests, where a fixed ~190s startup (Triton JIT of
        # the linear-attention kernels) is 75% of the wall clock and swamps every real effect.
        # Solving total = F + steps*r across the 60-step smoke and the 2,000-step run at
        # identical geometry gives r = 1.074 s/step, vs the generative full-FT's 1.046 s/step
        # excluding eval: this tagger runs at the SAME speed as the run it is compared against
        # (59.6 vs 60.3 samples/s). Judge any future change from a run of >=500 steps.
        # Default dataloader_num_workers=0 would still run DataCollatorForCharTagging's six
        # padded-array list comprehensions synchronously between steps; prefetch_factor
        # requires num_workers > 0.
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=(4 if args.num_workers > 0 else None),
        dataloader_pin_memory=True,
        **grouping_kwargs,
    )

    TrainerCls = _make_length_grouped_trainer(Trainer)
    LossSanityCallback = _make_loss_sanity_callback()

    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        train_lengths=train_lengths,
        callbacks=[LossSanityCallback(len(vocab))],
    )

    t_start = time.time()
    trainer.train()
    elapsed = time.time() - t_start

    # global_step, not training_args.max_steps -- the latter is -1 in full-epoch mode.
    total_steps = trainer.state.global_step
    s_per_step = elapsed / max(total_steps, 1)
    print(f"[THROUGHPUT] {elapsed:.0f}s total  {total_steps} steps  {s_per_step:.3f}s/step")
    # Entering trainer.train() costs a fixed ~190s (Triton JIT of the Gated-DeltaNet /
    # causal-conv1d kernels, worker spawn, allocator warmup) that does not scale with steps.
    # Print what it does to THIS run's number so a short run is never budgeted from naively.
    _FIXED_STARTUP_S = 190.0
    if total_steps:
        _adj = max(elapsed - _FIXED_STARTUP_S, 0.0) / total_steps
        print(f"[THROUGHPUT] ~{100 * min(_FIXED_STARTUP_S, elapsed) / max(elapsed, 1):.0f}% of "
              f"that is fixed startup -> steady state ~{_adj:.3f}s/step. "
              f"Budget long runs from THIS number, not the line above.")

    best_ckpt = trainer.state.best_model_checkpoint
    print(f"[INFO] best_model_checkpoint={best_ckpt}")
    assert best_ckpt is not None or not training_args.load_best_model_at_end, (
        "load_best_model_at_end=True but best_model_checkpoint is None"
    )

    final_dir = os.path.join(args.out, "final_model")
    trainer.model.save_bundle(final_dir, vocab, tokenizer)
    print(f"[OK] saved final bundle to {final_dir}")

    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_id": MODEL_ID, "num_labels": len(vocab),
            "n_train": len(train_ds), "n_dev": len(dev_ds),
            "max_steps_arg": args.max_steps, "num_epochs_arg": args.num_epochs,
            "total_steps": total_steps,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr, "warmup_ratio": args.warmup_ratio, "seed": RANDOM_SEED,
            "best_model_checkpoint": best_ckpt,
            "elapsed_seconds": elapsed, "s_per_step": s_per_step,
        }, f, indent=2)


if __name__ == "__main__":
    main()
