# Published models

**Nothing from this run is on the Hugging Face Hub.** Unlike
[`Gemma_4_LoRA_Diacritization/`](../Gemma_4_LoRA_Diacritization/) and
[`Qwen_3_5_4B_LoRA_Diacritization/`](../Qwen_3_5_4B_LoRA_Diacritization/), there is no
`Psycodem/...` repo for this model — do not cite one.

This is **not a LoRA adapter**, so there is no `adapter/` folder to mirror the sibling layout: the
tagger replaces the generative LM head with a 16-class per-character classification head and
fine-tunes the whole body, so the artifact is a full ~1.5 GB checkpoint rather than a ~40 MB
adapter. It is far too large for the source tree and lives on the Modal Volume
`diacritics-scratch`:

```
runs/qwen-tagging-fullepoch/final_model/        # the reported model
runs/qwen-tagging-fullepoch/checkpoint-16000/   # best eval_loss (0.0746), step 16,000 of 16,258
runs/arabert-tagging/labels.json                # the FROZEN 16-class label vocabulary
```

```bash
modal volume get diacritics-scratch runs/qwen-tagging-fullepoch/final_model ./final_model
```

The label vocabulary is shared with the AraBERT control and is **reused, never re-derived** —
regenerating it would silently invalidate both runs' checkpoints.

Base model: this warm-starts from the **full fine-tuned Qwen3.5-0.8B**
([`../Qwen_3_5_0_8B_Full_FineTune/`](../Qwen_3_5_0_8B_Full_FineTune/), macro DER_noce 2.58), not
from stock `Qwen/Qwen3.5-0.8B`. That is deliberate: it measures a tagging head against a
generative head on a *shared body*, not tagging from scratch.
