# -*- coding: utf-8 -*-
"""seq2seq_models_v2.py — re-score the seq2seq diacritisers with the canonical metrics.

Replaces the scoring in arabic_diacritization_benchmark_with_case_ending_results_samples.ipynb,
which used an inline DER/WER copy without numeral normalisation, citation
stripping, or digit-token exclusion.

    python seq2seq_models_v2.py

Heads-up on runtime: Glonor-ByT5 is byte-level and was the slowest model in the
original sweep (5h15m for 600 paragraphs). Predictions checkpoint every 50
paragraphs, so re-running resumes rather than starting over.
"""

from benchmark_runner import run_registry

REGISTRY = {
    "Flan-T5-Tashkeel-Small": {
        "repo": "Abdou/arabic-tashkeel-flan-t5-small",
        "kind": "seq2seq",
    },
    "Glonor-ByT5-Arabic": {
        "repo": "glonor/byt5-arabic-diacritization",
        "kind": "seq2seq",
    },
}

if __name__ == "__main__":
    run_registry(REGISTRY, "seq2seq_models_v2_results.csv")
