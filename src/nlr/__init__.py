"""nyc-landlord-resolution (``nlr``)

Precision-first probabilistic resolution of NYC landlord/owner identities from public HPD
records — the record-linkage engine behind WatchlineNYC's portfolio de-fragmentation,
extracted to run on the HPD tables alone (no knowledge graph, no geocoder).

Public API:
    owner_index(conn)   -> {(normalized name, bbl): owner_id}   # resolve owners
    normalize_name(s)   -> canonical name form for the index key
    splink_source       # the resolution model (extract / fit_predict_full / cluster_gated)
"""
from . import splink_source
from .resolve import normalize_name, owner_index

__all__ = ["owner_index", "normalize_name", "splink_source"]
