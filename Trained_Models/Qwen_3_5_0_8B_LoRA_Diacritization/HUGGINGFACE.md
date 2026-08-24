# Published adapters

**Nothing from this run is on the Hugging Face Hub yet.** Unlike the sibling
[`Gemma_4_LoRA_Diacritization/`](../Gemma_4_LoRA_Diacritization/) and
[`Qwen_3_5_4B_LoRA_Diacritization/`](../Qwen_3_5_4B_LoRA_Diacritization/) folders, there is no
`Psycodem/...` repo for this model — do not cite one.

The `*/adapter/` folders here hold the adapter **metadata only** (`adapter_config.json`,
`run_config.json`, tokenizer and chat template), matching the sibling folders' convention. The
weights themselves — `adapter_model.safetensors`, ~40 MB per checkpoint at r=16 — live on the
Modal Volume `diacritics-scratch`:

```
runs/qwen3.5-0.8b-lora-r16-eb96-contract-v1/final_model/       # step 10,862
runs/qwen3.5-0.8b-lora-r16-eb96-contract-v1/checkpoint-1200/   # best eval_loss
```

Retrieve them with:

```bash
modal profile activate saad702713
modal volume get diacritics-scratch \
  runs/qwen3.5-0.8b-lora-r16-eb96-contract-v1/final_model ./final_model
```

Base model: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B).

To publish, upload both checkpoints as subfolders of one repo (`final`, `best`) so the layout
matches the 4B repo's `10pct`/`30pct`/`50pct` convention, then rewrite this file to match the
siblings'.
