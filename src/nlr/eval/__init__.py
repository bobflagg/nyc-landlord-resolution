"""Gold-set evaluation harness for the resolution engine.

- ``gold_set.csv``  — 105 hand-adjudicated owner-contact records (the benchmark).
- ``scorer``        — pairwise precision/recall/F1 + FP/FN pair lists.
- ``build_gold``    — regenerate the gold seed from the adjudication rules.
- ``run_eval``      — fit + threshold sweep + score on a dense slice.
- ``run_loop``      — pass 1 vs the corporate-co-owner feedback merge.
- ``run_full``      — full-population run + precision guard + portfolios export.
- ``frame_gold.jsonl`` — a blind, preregistered, stratified pair-level gold fixture (frozen by
  ``bor.eval.export_gold``); carries each pair's gold label, stratum, WoW baseline and frame size.
- ``frame_impact``  — false-split impact of the splink edges vs Who Owns What (per-stratum
  precision + population magnitude). Pure read of the fixture — no DB, no splink re-run.
- ``run_frame_impact`` — print the frame-impact table + headline (reproduces the README's
  "Measured against Who Owns What" numbers).
"""
