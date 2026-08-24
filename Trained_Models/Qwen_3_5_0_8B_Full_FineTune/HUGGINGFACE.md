# Published models

**Nothing from this run is on the Hugging Face Hub.** There is no `Psycodem/...` repo for this
model — do not cite one. The two links below are the *4B LoRA adapters*, published separately;
they are not this model:

- [Psycodem/gemma-4-e4b-lora-diacritization](https://huggingface.co/Psycodem/gemma-4-e4b-lora-diacritization)
- [Psycodem/qwen3.5-4b-lora-diacritization](https://huggingface.co/Psycodem/qwen3.5-4b-lora-diacritization)

This is a **full fine-tune, not LoRA**, so there is no `adapter/` folder mirroring the sibling
layout — the artifact is a complete ~1.6 GB bf16 checkpoint, well past GitHub's 100 MB file limit.
It lives on the Modal Volume `diacritics-scratch`:

```
runs/qwen3.5-0.8b-fullft/final_model/         # end of training
runs/qwen3.5-0.8b-fullft/checkpoint-4070/     # the reported checkpoint
```

```bash
modal volume get diacritics-scratch runs/qwen3.5-0.8b-fullft/final_model ./final_model
```

A local copy of the step-4,070 checkpoint also sits in the untracked
`Train-Qwen-0.8B-Full-FT/Qwen3.5-0.8B-Arabic-Diacritization-Full-FT-Best-Step-4070/`.

This checkpoint is also the **warm-start body** for
[`../Qwen_3_5_0_8B_Tagging/`](../Qwen_3_5_0_8B_Tagging/), so it must not be deleted or moved
without breaking that run's provenance.

Base model: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B).
