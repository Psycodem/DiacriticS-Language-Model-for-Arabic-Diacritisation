# -*- coding: utf-8 -*-
"""small_models_v2.py — re-score the small task-specific models.

v1 (Small_models_v1.py) had the ce=/noce= split but not numeral normalisation,
citation stripping, or digit-token exclusion, so its numbers sit on a different
scale from the LLMs_Last_test.py rows in the results sheet.

    python small_models_v2.py

Tashkeel-350M-v2 keeps its own Arabic prompt from v1 ("قم بتشكيل هذا النص") —
the point here is to change the scoring, not the generation.
"""

from benchmark_runner import run_registry

REGISTRY = {
    "Fine-Tashkeel": {
        "repo": "basharalrfooh/Fine-Tashkeel",
        "kind": "seq2seq",
    },
    "Tashkeel-350M-v2": {
        "repo": "Etherll/Tashkeel-350M-v2",
        "kind": "causal_chat",
        "prompt_style": "tashkeel",
    },
}

if __name__ == "__main__":
    run_registry(REGISTRY, "small_models_v2_results.csv")
