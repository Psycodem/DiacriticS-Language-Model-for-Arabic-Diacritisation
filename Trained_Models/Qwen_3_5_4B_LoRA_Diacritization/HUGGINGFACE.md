# Published adapters

The three adapters in this folder are on the Hugging Face Hub as one repo, with
each training fraction as a subfolder:

**[Psycodem/qwen3.5-4b-lora-diacritization](https://huggingface.co/Psycodem/qwen3.5-4b-lora-diacritization)**

| Subfolder | Local folder |
|---|---|
| `10pct` | [`10pct/adapter/`](10pct/adapter/) |
| `30pct` | [`30pct/adapter/`](30pct/adapter/) |
| `50pct` | [`50pct/adapter/`](50pct/adapter/) |

```python
from peft import PeftModel
model = PeftModel.from_pretrained(model, "Psycodem/qwen3.5-4b-lora-diacritization", subfolder="50pct")
```

The local copies under each `*/adapter/` folder are byte-identical to what was
uploaded. Prefer the Hub for anything that needs to be reproducible by someone
else — it avoids cloning ~800 MB of weights with the source tree.

Base model: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B).
