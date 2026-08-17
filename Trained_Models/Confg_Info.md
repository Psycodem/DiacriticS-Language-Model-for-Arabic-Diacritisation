# Common Configuration — LoRA/QLoRA vs. Full Fine-Tuning (Qwen3.5-9B, Arabic Diacritization)

## Common configuration — shared by both methods

```python
import os
import torch

MODEL_ID = "Qwen/Qwen3.5-9B"

# NOTE: cache_dir — pass this to every load_dataset()/from_pretrained() call so repeated runs
# reuse downloaded weights/data instead of re-downloading each time.
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")
OUTPUT_DIR = "./qwen3.5-9b-diacritization"
EVAL_DIR = "./eval_outputs"

for d in (CACHE_DIR, OUTPUT_DIR, EVAL_DIR):
    os.makedirs(d, exist_ok=True)

# NOTE: no data preprocessing needed — Misraj/Sadeed_Tashkeela ships already cleaned, chunked
# (50-60 words), and filtered. Just wrap input/output into chat `messages` — do not re-clean/
# re-chunk/re-filter it for either training method.

RANDOM_SEED = 42
MAX_SEQ_LENGTH = 1024
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"  # verify against tokenizer.chat_template

COMMON_TRAINING_ARGS = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_eval_batch_size=2,
    lr_scheduler_type="cosine",
    warmup_steps=30,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",   # log_history is read manually for the loss plot — see below
)
```

## Option A — LoRA / QLoRA configuration

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# QLoRA-specific: 4-bit quantized frozen base model
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

LORA_TRAINING_ARGS = dict(
    **COMMON_TRAINING_ARGS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,   # effective batch size ~32
    learning_rate=2e-4,               # LoRA tolerates a higher LR since only adapters train
    optim="paged_adamw_8bit",         # memory-efficient optimizer, pairs well with 4-bit base
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,   # incompatible with completion-only loss masking (DataCollatorForCompletionOnlyLM)
)

# NOTE: save the adapter, not the full model — a few MB regardless of base model size.
# Reload later with: PeftModel.from_pretrained(base_model, ADAPTER_DIR)
LORA_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
```

## Option B — Full (normal) fine-tuning configuration

```python
FULL_FT_TRAINING_ARGS = dict(
    **COMMON_TRAINING_ARGS,
    per_device_train_batch_size=1,     # much lower — every parameter's optimizer state is materialized
    gradient_accumulation_steps=32,    # compensate for the smaller per-device batch
    learning_rate=2e-5,                # full FT needs a much lower LR to avoid catastrophic forgetting
    optim="adamw_torch",               # no quantized-optimizer trick available/needed here
    bf16=True,
    gradient_checkpointing=True,       # still essential — full FT is far more memory-hungry
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
    # deepspeed="ds_zero3_config.json",  # NOTE: at 9B params, full FT typically needs
                                          # DeepSpeed ZeRO-3 / FSDP sharding across multiple GPUs —
                                          # QLoRA does not need this.
)

# NOTE: save the full model here (not an "adapter") — every weight was updated.
FULL_FT_MODEL_DIR = os.path.join(OUTPUT_DIR, "full_model")
# model.save_pretrained(FULL_FT_MODEL_DIR); tokenizer.save_pretrained(FULL_FT_MODEL_DIR)
```

## Shared: evaluation CSV + loss plot (same for both methods)

```python
# NOTE: csv file — after inference, save predictions per split (train-sample/test/benchmark) as
# CSV with input/reference/prediction columns, e.g.:
#   df.to_csv(os.path.join(EVAL_DIR, f"{split_name}_predictions.csv"), index=False, encoding="utf-8-sig")

# NOTE: plot the loss — read trainer.state.log_history after training (works identically for
# both LoRA and full FT, since both use the same Trainer/SFTTrainer loop) and plot train vs.
# eval loss over steps.
```

## Key differences at a glance

| | LoRA / QLoRA | Full fine-tuning |
|---|---|---|
| Base model | 4-bit quantized, frozen | Full precision (bf16), all weights trainable |
| Trainable params | ~0.1–1% (adapters only) | 100% |
| `per_device_train_batch_size` | Higher (2+) | Much lower (1), often needs sharding |
| `learning_rate` | Higher (~2e-4) | Lower (~2e-5) |
| `optim` | `paged_adamw_8bit` | `adamw_torch` (or DeepSpeed-fused variants) |
| Multi-GPU sharding | Usually not needed | Typically required at 9B (DeepSpeed ZeRO-3 / FSDP) |
| What you save | Small adapter (few MB) | Full model (~18GB+ in bf16) |
| Data preprocessing | None needed either way — dataset is pre-cleaned |
