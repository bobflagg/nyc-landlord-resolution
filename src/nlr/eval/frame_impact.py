"""False-split impact of CONNECTED_BY_SPLINK, scored against a blind, preregistered frame.

Where the 105-record ``gold_set.csv`` measures *internal* pairwise precision, this measures the
thing that matters for the Who Owns What integration: **when splink adds a "same owner" edge, how
often is it right, and how many real false-splits does it fix that WoW leaves fragmented?**

The input is ``frame_gold.jsonl`` — one adjudicated pair per row, frozen by ``bor.eval.export_gold``
from a blinded, stratified sample of the candidate frame and hand-labeled in the owner-review app.
Each row already carries its gold ``label`` (SAME/DIFFERENT/INDETERMINATE), its ``stratum``, the
WoW baseline decision for the pair, and the ``frame_size`` of its stratum. This module is a pure
function of that fixture — **no database, no splink re-run** — so the headline numbers reproduce
anywhere the repo does (Mode A, the audit of the shipped edges; see the design doc for Mode B).

    uv run python -m nlr.eval.run_frame_impact

Strata: ``S2_model`` = splink's own merges (this engine's contribution — the precision headline);
``S4_hard_neg`` = unlinked same-surname pairs (the residual — false splits nothing recovered).
INDETERMINATE is first-class: it lowers coverage, never counts as right or wrong.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

SPLINK_SIGNAL = "splink-fellegi-sunter"      # S2_model — nlr's edges
RESIDUAL_STRATUM = "S4_hard_neg"             # unlinked same-surname — the miss population


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Point estimate + 95% Wilson interval for k/n. (0,0,0) when n==0."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load_frame_gold(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _counts(rows: list[dict]) -> dict:
    same = sum(1 for r in rows if r["label"] == "SAME")
    diff = sum(1 for r in rows if r["label"] == "DIFFERENT")
    indet = len(rows) - same - diff
    return {"n": len(rows), "same": same, "diff": diff, "indet": indet}


def score_stratum(rows: list[dict]) -> dict:
    """Per-stratum report. For a SAME-target stratum (splink merges) precision = SAME/(SAME+DIFF);
    for the residual stratum the same arithmetic reads as the share of unlinked pairs truly SAME."""
    c = _counts(rows)
    frame_size = rows[0]["frame_size"] if rows else 0
    decided = c["same"] + c["diff"]
    p_strict = wilson(c["same"], decided)
    p_incl = wilson(c["same"], c["n"])
    coverage = round(decided / c["n"], 3) if c["n"] else 0.0
    # net-new vs WoW: a correct merge (gold SAME) the WoW baseline leaves split
    same_rows = [r for r in rows if r["label"] == "SAME"]
    net_new = sum(1 for r in same_rows if r.get("wow") == "DIFFERENT")
    wow_already = sum(1 for r in same_rows if r.get("wow") == "SAME")
    rate = c["same"] / c["n"] if c["n"] else 0.0
    nn_rate = net_new / c["n"] if c["n"] else 0.0
    return {
        **c, "frame_size": frame_size, "coverage": coverage,
        "precision_strict": p_strict, "precision_inclusive": p_incl,
        "net_new_fix": net_new, "wow_already_same": wow_already,
        "est_correct_merges": round(frame_size * p_strict[0]),
        "est_net_new_fixes": round(frame_size * nn_rate),
        "est_same_population": round(frame_size * rate),
        "false_merges": [r for r in rows if r["label"] == "DIFFERENT"],   # FP, for inspection
    }


def score_frame(gold_rows: list[dict]) -> dict:
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in gold_rows:
        by_stratum[r["stratum"]].append(r)
    return {st: score_stratum(rows) for st, rows in sorted(by_stratum.items())}
