# -*- coding: utf-8 -*-
"""medium_models_v2.py — re-score the medium-tier models.

v1 (Mideum_models_v1.py) scored with the oldest inline metric copy — no numeral
normalisation, no citation stripping, no digit-token exclusion.

    python medium_models_v2.py

Two repo ids are corrected here. v1 listed the Gemma diacritiser as bare
"gemma-3-1b-pt-10k-diacritization" with no owner prefix, which cannot resolve on
the Hub — the sheet gives it as Bisher/…, and that is what this uses. It is also
loaded through `pipeline("text-generation", …)`, matching How_to_use.py and
instruct_models_v1.py rather than the raw AutoModelForCausalLM path v1 used.

That makes this model the one case where generation changes as well as scoring,
so treat its delta as "now runs at all" rather than as a clean metric-only
comparison. It has no recorded numbers in the sheet, which is consistent with
never having loaded.
"""

from benchmark_runner import run_registry

REGISTRY = {
    "Qwen3.5-0.8B": {
        "repo": "Qwen/Qwen3.5-0.8B",
        "kind": "causal_chat",
        "single_line": True,      # v1 kept only the first output line for instruct models
    },
    "gemma-3-1b-pt-10k-diacritization": {
        "repo": "Bisher/gemma-3-1b-pt-10k-diacritization",
        "kind": "pipeline_chat",
        "trust_remote_code": True,
    },
}

if __name__ == "__main__":
    run_registry(REGISTRY, "medium_models_v2_results.csv")
