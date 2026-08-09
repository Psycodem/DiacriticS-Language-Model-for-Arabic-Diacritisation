
sft_config = SFTConfig(
    output_dir="./qwen3.5-9b-qlora-diacritization",
    num_train_epochs=6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,   # effective batch size ~32
    per_device_eval_batch_size=2,
    learning_rate=2e-4,               # typical QLoRA LR (base model frozen, only adapters train)
    lr_scheduler_type="cosine",
    warmup_steps=30,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    max_seq_length=1024,
    dataset_text_field="text",
    packing=False,                    # packing is incompatible with completion-only masking
    report_to="none",
)
