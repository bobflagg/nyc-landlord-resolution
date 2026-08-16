"""Pairwise precision/recall scoring of a linkage result against a labeled gold set.

Ground truth is a table of records, each hand-adjudicated to a ``gold_entity_id``
(any two records sharing an id are a true match). Scoring is restricted to pairs
where *both* records are in the gold set — the standard for a partial gold set —
and reported precision-first, because a false merge (naming the wrong landlord)
is the costly error.

The gold set is keyed on the extraction's deterministic ``unique_id``, so a fresh
linkage run produces the same ids and joins cleanly.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd


def load_gold(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"unique_id": str, "gold_entity_id": str})


def score(clusters_df: pd.DataFrame, gold_df: pd.DataFrame):
    """Return ``(metrics, false_merges, missed_merges)`` for the labeled records.

    ``metrics`` is a dict with pairwise tp/fp/fn, precision, recall, f1.
    ``false_merges`` / ``missed_merges`` are lists of (record_a, record_b) dicts
    for inspection (the FP and FN pairs).
    """
    j = gold_df.merge(clusters_df[["unique_id", "cluster_id"]], on="unique_id", how="inner")
    recs = j.to_dict("records")

    tp = fp = fn = tn = 0
    false_merges, missed = [], []
    for a, b in combinations(recs, 2):
        gold_same = a["gold_entity_id"] == b["gold_entity_id"]
        pred_same = a["cluster_id"] == b["cluster_id"]
        if pred_same and gold_same:
            tp += 1
        elif pred_same and not gold_same:
            fp += 1
            false_merges.append((a, b))
        elif not pred_same and gold_same:
            fn += 1
            missed.append((a, b))
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = dict(labeled=len(j), pairs=tp + fp + fn + tn, tp=tp, fp=fp, fn=fn,
                   precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3))
    return metrics, false_merges, missed


def sweep(preds, nodes, gold_df, thresholds=(0.80, 0.88, 0.92, 0.95, 0.98),
          name_freq=None) -> pd.DataFrame:
    """Cluster (name-gated) at each threshold and score against the gold set. Returns a
    table (one row per threshold) so the F1-maximizing operating point is visible.
    ``nodes`` is the record frame (supplies the singleton universe); ``name_freq``
    enables the common-name veto so the sweep matches the production model."""
    from nlr import splink_source as ss

    rows = []
    for t in thresholds:
        clusters = ss.cluster_gated(preds, nodes, threshold=t, name_freq=name_freq)
        m, _, _ = score(clusters, gold_df)
        rows.append({"threshold": t, **{k: m[k] for k in
                     ("precision", "recall", "f1", "tp", "fp", "fn")}})
    return pd.DataFrame(rows)
