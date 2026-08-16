"""Gold-set evaluation harness for the resolution engine.

- ``gold_set.csv``  — 105 hand-adjudicated owner-contact records (the benchmark).
- ``scorer``        — pairwise precision/recall/F1 + FP/FN pair lists.
- ``build_gold``    — regenerate the gold seed from the adjudication rules.
- ``run_eval``      — fit + threshold sweep + score on a dense slice.
- ``run_loop``      — pass 1 vs the corporate-co-owner feedback merge.
- ``run_full``      — full-population run + precision guard + portfolios export.
"""
