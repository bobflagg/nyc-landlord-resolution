"""owner_index — the public resolution API.

``owner_index(conn)`` resolves HPD owner contacts into precision-first entities and returns
a plain lookup: ``(normalized owner name, bbl) -> owner_id``. All the hard, validated work
lives in :mod:`nlr.splink_source`; this just runs it and returns the mapping, so consumers
(a portfolio-clustering pipeline, an analysis) never touch Splink internals.
"""
import re

from . import splink_source as ss

# ~10% deterministic dense training slice — a RECALL tuning lever. fit_predict_full trains
# m/u/λ here (low-λ trap fix) and predicts over the full population; hash-based (not random())
# so a run reproduces. Slice size sets λ *inversely* — a smaller slice merges MORE. ~10% is
# a good operating point: the big fragmented operators consolidate (Croman -> one portfolio)
# and precision is unaffected (0 different-surname clusters at every size tested). It is NOT
# tuned for maximum consolidation, and it's fiddly: a *targeted* small slice (the eval's
# gold-operator slice) merges more, but a plain small random slice under-samples specific
# operators and fragments them, and a cross-office-name enrichment did not help (it enlarges
# the slice, lowering λ). Precision never depends on this knob — only how much consolidation.
#
# DuckDB note: `splink_source._duckdb_sql` maps `hashtext` -> DuckDB's `hash`, so the
# no-Postgres path samples a *different* ~10% (different algorithm, not a different rate).
# Precision is identical and most operators consolidate identically; a borderline operator
# can land a fragment or two differently (e.g. Croman 1 cluster on Postgres, ~3 near-fragments
# on DuckDB). Same recall lever, backend-dependent — use the Postgres path for exact parity.
_TRAIN_SAMPLE = ("abs(hashtext(coalesce(c.firstname,'')||coalesce(c.lastname,'')"
                 "||coalesce(c.businesshousenumber,''))) % 1000 < 100")


def normalize_name(s):
    """Canonical owner-name form used in the index key. Consumers MUST normalize their own
    names with this same function before looking up ``(name, bbl)``."""
    return re.sub(r"\s+", " ", s).strip().upper() if isinstance(s, str) else s


def owner_index(conn, *, threshold: float = 0.95) -> dict:
    """``{(normalized name, bbl): owner_id}`` for every person owner-contact in the HPD data.

    ``owner_id`` is opaque and stable WITHIN one call (EM is stochastic → not stable across
    runs; fine for one-shot use, don't persist it). Person owners only; orgs/LLCs are
    excluded (they pass through unresolved). Runs the full population once (~1-2 min).
    """
    # 1. full population of owner-contact identities (one row per (name, address), bbls agg'd)
    full = ss.extract(conn, "TRUE")
    full = full[full.contact_kind == "person"].drop_duplicates("unique_id").reset_index(drop=True)

    # 2. dense training slice (stable prior) + full-population guards
    train = ss.extract(conn, _TRAIN_SAMPLE)
    train = train[train.contact_kind == "person"].drop_duplicates("unique_id").reset_index(drop=True)
    degrees = ss.address_degrees(conn)   # aggregator-address masking (real full-pop degrees)
    name_freq = ss.name_freq(conn)       # (surname, first-initial) rarity → common-name veto

    # 3. resolve: train on the slice, predict over the full pop, cluster with both vetoes
    #    (name-anchored blocking + first-name veto + common-name veto). No feedback loop —
    #    at full scale term-frequency subsumes it.
    linker, preds = ss.fit_predict_full(train, full, addr_degrees=degrees)
    clusters = ss.cluster_gated(preds, full, threshold, name_freq=name_freq)

    # 4. explode each entity to its (name, bbl) pairs → owner_id
    m = (clusters[["unique_id", "cluster_id"]]
         .merge(full[["unique_id", "name_full", "bbls"]], on="unique_id")
         .explode("bbls"))
    return {(normalize_name(name), bbl): cid
            for name, bbl, cid in zip(m["name_full"], m["bbls"], m["cluster_id"])
            if isinstance(bbl, str)}
