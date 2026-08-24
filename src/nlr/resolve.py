"""owner_index — the public resolution API.

``owner_index(conn)`` resolves HPD owner contacts into precision-first entities and returns
a plain lookup: ``(normalized owner name, bbl) -> owner_id``. All the hard, validated work
lives in :mod:`nlr.splink_source`; this just runs it and returns the mapping, so consumers
(a portfolio-clustering pipeline, an analysis) never touch Splink internals.
"""
import hashlib
import re
from collections import Counter

import pandas as pd

from . import splink_source as ss

# Clustering threshold. The stratified slice (below) is dense in same-name pairs, so the model
# scores same-name/same-address pairs confidently; 0.999 holds gold precision at ~0.996 (0
# different-surname clusters). Calibrated against the 105-record gold set.
DEFAULT_THRESHOLD = 0.999

# A thin singleton base (~1% of single-occurrence names) mixed into the training slice so u/λ
# stay calibrated to the population rather than to the dense same-name groups alone.
_BASE_HASH_PCT = 1
_gkey = lambda ln, fi: int(hashlib.md5(f"{ln}|{fi}".encode()).hexdigest()[:8], 16)


def normalize_name(s):
    """Canonical owner-name form used in the index key. Consumers MUST normalize their own
    names with this same function before looking up ``(name, bbl)``."""
    return re.sub(r"\s+", " ", s).strip().upper() if isinstance(s, str) else s


def _stratified_train(full: pd.DataFrame) -> pd.DataFrame:
    """Principled training slice: every identity in a multi-member ``(last_name,
    first_initial)`` group — every record with a potential same-name peer, across ALL
    operators — plus a thin singleton base for u/λ calibration.

    Replaces the old ~10% random hash sample. It is representative and reproducible: two
    independent draws resolve the population near-identically (validated at same-owner-pair
    Jaccard ~0.99, vs ~0.5 for an arbitrary random slice), so the resolution reflects the data,
    not which records happened to be sampled. Computed in pandas over the extracted frame (not
    SQL), so the Postgres and DuckDB paths now produce identical results — there is no longer a
    hash-function divergence between backends."""
    key = list(zip(full["last_name"].fillna(""), full["first_initial"].fillna("")))
    gsize = Counter(key)
    keep = [(gsize[k] >= 2) or (_gkey(*k) % 100 < _BASE_HASH_PCT) for k in key]
    return full[pd.Series(keep, index=full.index)].reset_index(drop=True)


def owner_index(conn, *, threshold: float = DEFAULT_THRESHOLD, feedback: bool = True) -> dict:
    """``{(normalized name, bbl): owner_id}`` for every person owner-contact in the HPD data.

    ``owner_id`` is opaque and stable WITHIN one call (EM is stochastic → not stable across
    runs; fine for one-shot use, don't persist it). Person owners only; orgs/LLCs are
    excluded (they pass through unresolved). Runs the full population once (~1-2 min).
    """
    # 1. full population of owner-contact identities (one row per (name, address), bbls agg'd)
    full = ss.extract(conn, "TRUE")
    full = full[full.contact_kind == "person"].drop_duplicates("unique_id").reset_index(drop=True)

    # 2. principled stratified training slice (stable prior) + full-population guards
    train = _stratified_train(full)
    degrees = ss.address_degrees(conn)   # aggregator-address masking (real full-pop degrees)
    name_freq = ss.name_freq(conn)       # (surname, first-initial) rarity → common-name veto

    # 3. resolve: train on the slice, predict over the full pop, cluster with both vetoes
    #    (name-anchored blocking + first-name veto + common-name veto).
    linker, preds = ss.fit_predict_full(train, full, addr_degrees=degrees)
    clusters = ss.cluster_gated(preds, full, threshold, name_freq=name_freq)

    # 3b. feedback loop: bridge same-owner offices that share a private corporate co-owner —
    #     the cross-office consolidation name/address can't reach (Croman via Centennial
    #     Properties, Rashad via The Andrews Organization). Guarded by corp-degree cap +
    #     name-rarity + first-name veto; validated precision-safe against the owner-level gold
    #     (P 1.0, 0 cross-surname) for a ~+30pt recall lift. Pass feedback=False to skip it.
    if feedback:
        corp_df = ss.corp_owners_for(conn)     # (bbl, corp) over all buildings
        corp_deg = ss.corp_degrees(conn)       # corp -> distinct-landlord degree (aggregator cap)
        clusters = ss.feedback_merge(full, clusters, corp_df, corp_deg, name_freq=name_freq)

    # 4. explode each entity to its (name, bbl) pairs → owner_id
    m = (clusters[["unique_id", "cluster_id"]]
         .merge(full[["unique_id", "name_full", "bbls"]], on="unique_id")
         .explode("bbls"))
    return {(normalize_name(name), bbl): cid
            for name, bbl, cid in zip(m["name_full"], m["bbls"], m["cluster_id"])
            if isinstance(bbl, str)}
