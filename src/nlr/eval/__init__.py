"""Gold-set evaluation harness for the resolution engine.

- ``gold_set.csv``  — 105 hand-adjudicated owner-contact records (the benchmark).
- ``scorer``        — pairwise precision/recall/F1 + FP/FN pair lists.
- ``build_gold``    — regenerate the gold seed from the adjudication rules.
- ``run_eval``      — fit + threshold sweep + score on a dense slice.
- ``run_loop``      — pass 1 vs the shared-company feedback merge.
- ``run_full``      — full-population run + precision guard + portfolios export.
- ``frame_gold.jsonl`` — a blind, stratified pair-level gold fixture (frame frozen before labeling;
  exported by ``bor.eval.export_gold``); one human annotator's labels, plus each pair's stratum,
  WoW baseline and frame size.
- ``frame_impact``  — link precision of the splink edges, and of the links that join separate Who
  Owns What portfolios (per-stratum precision + population magnitude). Precision only; recall is
  not estimated. Pure read of the fixture — no DB, no splink re-run.
- ``run_frame_impact`` — print the per-stratum table + the model-stratum headline (reproduces the
  README's "Measured against Who Owns What" figures).
"""
