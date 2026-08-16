"""
SplinkSource — probabilistic record linkage for the portfolio pipeline (Stage 2).

Replaces WoW's rule-based ``landlords_with_connections`` matching with Splink
(Fellegi-Sunter). It extracts *raw* HPD owner-contact records, normalizes them
lightly, and links them into resolved landlord entities using fuzzy comparisons
+ (in the full run) term-frequency weighting -- which recovers the geocode/typo
splits the address-gated rules cannot (e.g. Croman's "4 WEST 51 STREET", a
dropped "42"; Rashad's "666 BROADAWAY").

DB-agnostic: ``extract()`` takes any psycopg2 connection -- the raw HPD tables
(``hpd_contacts`` / ``hpd_registrations``) exist in both the WoW and JustFix
databases. Linkage runs on DuckDB in-process.

Design notes:
  * Grain = one row per distinct (kind, name, raw business address) identity,
    with ``bbls`` aggregated -- so term frequency reflects name rarity, not
    portfolio size, and the fragments WoW split are preserved for Splink to link.
  * Source = raw ``hpd_contacts`` (not ``wow_landlords``): the raw components
    still carry the shared house/street/zip that Geosupport had blanked.
  * ``DISTINCT ON (bbl)`` most-recent registration kills the multi-building
    fan-out; a registration covering several BBLs spreads to each.
  * Heavy address standardization (Geosupport/libpostal) is a later step that
    *adds* columns; Splink compares raw AND standardized, so a geocode failure
    never blanks the comparison.
"""
from __future__ import annotations

import pandas as pd

# The owner roles WoW keeps; agents/managers are excluded (shared across
# unrelated landlords -> false links via registered-agent addresses).
OWNER_ROLES = "{HeadOfficer,IndividualOwner,CorporateOwner,JointOwner}"

# `__WHERE__` is replaced (not %-parameterized) so callers can inject a slice
# predicate over the `c.` contact alias, e.g. "random() < 0.004" or a name list.
EXTRACT_SQL = r"""
WITH chosen_reg AS (                         -- most-recent registration per BBL (no fan-out)
  SELECT DISTINCT ON (btrim(r.bbl)) btrim(r.bbl) AS bbl, r.registrationid
  FROM hpd_registrations r
  WHERE r.bbl IS NOT NULL
  ORDER BY btrim(r.bbl), r.registrationenddate DESC NULLS LAST, r.lastregistrationdate DESC NULLS LAST
),
obs AS (                                     -- one row per (bbl, qualifying owner contact)
  SELECT cr.bbl, c.type AS role,
    NULLIF(upper(btrim(c.firstname)),'')            AS first_name,
    NULLIF(upper(btrim(c.lastname)),'')             AS last_name,
    NULLIF(upper(btrim(c.corporationname)),'')      AS corp_name,
    NULLIF(upper(btrim(c.businesshousenumber)),'')  AS biz_house,
    NULLIF(upper(btrim(c.businessstreetname)),'')   AS biz_street,
    NULLIF(upper(btrim(c.businessapartment)),'')    AS biz_apt,
    NULLIF(upper(btrim(c.businesscity)),'')         AS biz_city,
    NULLIF(upper(btrim(c.businessstate)),'')        AS biz_state,
    NULLIF(btrim(c.businesszip),'')                 AS biz_zip
  FROM chosen_reg cr
  JOIN hpd_contacts c ON c.registrationid = cr.registrationid
  WHERE c.type = ANY('{HeadOfficer,IndividualOwner,CorporateOwner,JointOwner}')
    AND (c.firstname IS NOT NULL OR c.lastname IS NOT NULL OR c.corporationname IS NOT NULL)
    AND (c.businesshousenumber IS NOT NULL OR c.businessstreetname IS NOT NULL)
    AND length(concat(c.businesshousenumber, c.businessstreetname)) > 2
    AND (__WHERE__)
),
rec AS (
  SELECT
    CASE WHEN first_name IS NOT NULL OR last_name IS NOT NULL THEN 'person' ELSE 'org' END AS contact_kind,
    COALESCE(NULLIF(trim(concat_ws(' ', first_name, last_name)),''), corp_name)             AS name_full,
    first_name, last_name,
    CASE WHEN first_name IS NOT NULL OR last_name IS NOT NULL THEN corp_name END            AS assoc_corp,
    trim(both ' ,' from regexp_replace(
      COALESCE(NULLIF(trim(concat_ws(' ', first_name, last_name)),''), corp_name, ''),
      ',?\s*(LLC|L\.L\.C\.|INC\.?|CORP\.?|LTD\.?|L\.P\.|LLP|PLLC|P\.C\.|LP|PC)\s*$','','i')) AS name_normalized,
    biz_house, biz_street, biz_apt,
    NULLIF(regexp_replace(coalesce(biz_apt,''),'\D','','g'),'')                             AS biz_apt_num,
    biz_city, biz_state, biz_zip, bbl, role
  FROM obs
)
SELECT
  md5(contact_kind||'|'||coalesce(name_full,'')||'|'||
      concat_ws('|', biz_house, biz_street, biz_apt, biz_zip, biz_city, biz_state)) AS unique_id,
  contact_kind, name_full, first_name, last_name, name_normalized,
  biz_house, biz_street, biz_apt, biz_apt_num, biz_city, biz_state, biz_zip,
  regexp_replace(biz_street,
    '\s+(STREET|ST|AVENUE|AVE|PLACE|PL|BOULEVARD|BLVD|ROAD|RD|LANE|LN|DRIVE|DR|COURT|CT|PARKWAY|PKWY|TERRACE|TER)$',
    '') AS biz_street_norm,
  left(first_name, 1)                                    AS first_initial,
  count(DISTINCT bbl)                                     AS n_bbls,
  array_agg(DISTINCT bbl ORDER BY bbl)                   AS bbls,
  array_remove(array_agg(DISTINCT assoc_corp), NULL)     AS assoc_corps
FROM rec
GROUP BY contact_kind, name_full, first_name, last_name, name_normalized,
         biz_house, biz_street, biz_apt, biz_apt_num, biz_city, biz_state, biz_zip
"""

# A (house, street) shared by more than this many distinct landlords is treated as
# an aggregator/registered-agent address and its address evidence is dropped. Tuned
# for full-population degrees (WoW's MAX_ADDR_DEGREE is 50 over its node set).
AGGREGATOR_DEGREE = 25

# Columns Splink compares (scalars only; bbls/assoc_corps ride along as payload).
COMPARISON_COLS = ["unique_id", "contact_kind", "name_full", "first_name", "last_name",
                   "name_normalized", "biz_house", "biz_street", "biz_street_norm",
                   "first_initial", "biz_apt_num", "biz_zip"]


def extract(conn, where: str = "TRUE") -> pd.DataFrame:
    """Run the extraction, returning one row per owner-contact identity.

    ``where`` is a SQL predicate over the ``c`` contact alias used to slice the
    population (e.g. ``"random() < 0.004"``). It is trusted internal SQL, not
    user input.
    """
    sql = EXTRACT_SQL.replace("__WHERE__", where)
    return pd.read_sql(sql, conn)   # no params -> LIKE '%' in `where` stays literal


# Distinct landlords per (house, normalized street) across the WHOLE owner-contact
# population — the aggregator signal. Computed unfiltered (no slice), so masking is
# production-correct rather than depending on what happens to be in a sample. The
# street-suffix regex MUST match biz_street_norm in EXTRACT_SQL so the join lines up.
ADDR_DEGREE_SQL = r"""
SELECT h AS biz_house, s AS biz_street_norm, count(DISTINCT who) AS addr_degree
FROM (
  SELECT DISTINCT
    upper(btrim(c.firstname)) || '|' || upper(btrim(c.lastname))          AS who,
    NULLIF(upper(btrim(c.businesshousenumber)),'')                        AS h,
    regexp_replace(NULLIF(upper(btrim(c.businessstreetname)),''),
      '\s+(STREET|ST|AVENUE|AVE|PLACE|PL|BOULEVARD|BLVD|ROAD|RD|LANE|LN|DRIVE|DR|COURT|CT|PARKWAY|PKWY|TERRACE|TER)$',
      '')                                                                 AS s
  FROM hpd_contacts c
  WHERE c.type = ANY('{HeadOfficer,IndividualOwner,CorporateOwner,JointOwner}')
    AND (c.firstname IS NOT NULL OR c.lastname IS NOT NULL)
    AND (c.businesshousenumber IS NOT NULL OR c.businessstreetname IS NOT NULL)
) t
WHERE h IS NOT NULL AND s IS NOT NULL
GROUP BY h, s
"""


def address_degrees(conn) -> pd.DataFrame:
    """Full-population (house, biz_street_norm) -> distinct-landlord count. Pass to
    ``fit(addr_degrees=...)`` so the aggregator mask uses real degrees, not sample."""
    return pd.read_sql(ADDR_DEGREE_SQL, conn)


# Distinct landlords per corporation name — the corp analog of address degree. A
# corp used by many unrelated landlords is a registered agent / big manager, a weak
# cross-office bridge; a private umbrella (Croman's Centennial) is used by few.
CORP_DEGREE_SQL = r"""
SELECT corp, count(DISTINCT who) AS degree
FROM (
  SELECT DISTINCT upper(btrim(c.corporationname)) AS corp,
                  upper(btrim(c.firstname)) || '|' || upper(btrim(c.lastname)) AS who
  FROM hpd_contacts c
  WHERE c.corporationname IS NOT NULL AND length(btrim(c.corporationname)) > 3
    AND (c.firstname IS NOT NULL OR c.lastname IS NOT NULL)
) t
GROUP BY corp
"""


def corp_degrees(conn) -> pd.DataFrame:
    """Full-population corp -> distinct-landlord count, to cap aggregator corps in
    the feedback merge."""
    return pd.read_sql(CORP_DEGREE_SQL, conn)


# Distinct landlord identities per (last name, first initial) — name rarity. A rare
# name (Croman/Castellano/Kadden ~5-11) across offices is almost surely one person;
# a common one (Smith 25, Chen 232) might be several, so the corp-bridge — a shared
# third-party manager — must NOT merge them.
NAME_FREQ_SQL = r"""
SELECT last AS last_name, init AS first_initial, count(*) AS identities
FROM (
  SELECT DISTINCT upper(btrim(lastname)) AS last, left(upper(btrim(firstname)),1) AS init,
                  upper(btrim(firstname)) AS f, upper(btrim(businesshousenumber)) AS h,
                  upper(btrim(businessstreetname)) AS s
  FROM hpd_contacts
  WHERE lastname IS NOT NULL AND firstname IS NOT NULL
    AND type = ANY('{HeadOfficer,IndividualOwner,CorporateOwner,JointOwner}')
) t
GROUP BY last, init
"""


def name_freq(conn) -> pd.DataFrame:
    """Full-population (last name, first initial) -> distinct-identity count, to gate
    the feedback merge to rare names."""
    return pd.read_sql(NAME_FREQ_SQL, conn)


# Address columns blanked when an address is an aggregator (see _mask_aggregators).
_ADDR_COLS = ["biz_house", "biz_street", "biz_street_norm", "biz_zip", "biz_apt_num"]


def _mask_aggregators(df: pd.DataFrame, addr_degrees: pd.DataFrame | None,
                      aggregator_degree: int) -> pd.DataFrame:
    """Blank the address evidence of aggregator/registered-agent addresses.

    An address shared by many *distinct* landlords is an office building, not a
    private office; dropping its address columns leaves those records able to link
    on name only — same-name pairs still merge, different-name neighbours no longer
    do. ``addr_degrees`` (from :func:`address_degrees`) gives real full-population
    degrees; without it, degree is computed within ``df`` (sample-only)."""
    d = df.copy()
    if addr_degrees is not None:
        d = d.merge(addr_degrees, on=["biz_house", "biz_street_norm"], how="left")
        d["_deg"] = d["addr_degree"].fillna(0)
        d = d.drop(columns=["addr_degree"])
    else:
        d["_deg"] = d.groupby(["biz_house", "biz_street_norm"])["name_normalized"].transform("nunique")
    d.loc[d["_deg"] > aggregator_degree, _ADDR_COLS] = None
    return d.drop(columns=["_deg"])


def _settings():
    """The Splink model settings — comparisons + blocking. Shared by every fit path
    so the slice-trained model transfers cleanly onto the full population."""
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on

    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            # Term-frequency on names: a rare "CROMAN"/"RASHAD" match counts far more
            # than a common one — the lever WoW's rules lack. TF is computed over the
            # predicted population, so predict on the full set (or a representative
            # sample), not a target-heavy slice, or every target name looks "common".
            cl.JaroWinklerAtThresholds("last_name", [0.92, 0.85]).configure(term_frequency_adjustments=True),
            cl.JaroWinklerAtThresholds("first_name", [0.92, 0.80]).configure(term_frequency_adjustments=True),
            # Component address (house Levenshtein for "4"/"424", street fuzzy for
            # "BROADWAY"/"BROADAWAY", zip exact + TF). Aggregator down-weighting is a
            # later refinement — the composite-address attempt broke merging.
            cl.LevenshteinAtThresholds("biz_house", [1, 2]),
            # Compare the *suffix-normalized* street ("WEST 21ST STREET" == "WEST 21ST")
            # so a tight fuzzy threshold still catches the common suffix gap without
            # loosening enough to fuse unrelated streets.
            cl.JaroWinklerAtThresholds("biz_street_norm", [0.92, 0.85]),
            cl.ExactMatch("biz_zip").configure(term_frequency_adjustments=True),
        ],
        # NAME-ANCHORED blocking — one rule: same last name + first initial. Landlord
        # resolution keys on the person, so a surname match is required to even *score*
        # a pair. The first-initial gate additionally stops different-first-name
        # namesakes at one address (ANDREA vs FILIPPO, DIVYA vs JAMAL); STEVE/STEVEN,
        # ZACH/ZACHARY share an initial and still merge.
        #
        # An address-only rule (block_on biz_house, biz_street_norm, first_initial) was
        # here too, but the full-population run proved it a precision hole: it scores
        # DIFFERENT-surname office-mates (JOEL BRAVER + JACOB GUTMAN at one address, same
        # initial), and a strong shared-address match overpowers the surname mismatch —
        # 2,044 clusters fused different people at population scale. Removing it zeroed
        # those out at ZERO cost to recall (Croman/Rashad/Valiotis consolidate via the
        # name rule + term-frequency; the 87-record gold P/R was unchanged). The eval's
        # rare-surname/masked-address gold could not see this failure mode. Trade-off:
        # surname-*typo* pairs are no longer candidates (the fuzzy last_name levels go
        # dead for blocking) — recover later with a typo-safe block if it proves to
        # matter, NOT by restoring the address-only rule. Never `biz_zip` alone (208M).
        blocking_rules_to_generate_predictions=[
            block_on("last_name", "first_initial"),
        ],
        retain_intermediate_calculation_columns=True,
    )


def _train_linker(d: pd.DataFrame, seed: int):
    """Construct a Linker over already-masked ``d`` and run the training passes."""
    from splink import DuckDBAPI, Linker, block_on

    linker = Linker(d[COMPARISON_COLS], _settings(), db_api=DuckDBAPI())
    # Same name + same address is ~certainly a match — a good anchor for training.
    linker.training.estimate_probability_two_random_records_match(
        [block_on("first_name", "last_name", "biz_house", "biz_street_norm")], recall=0.7)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=seed)
    # Two EM passes triangulate m: block on the name to learn the address weights,
    # block on the address to learn the name weights.
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("first_name", "last_name"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("biz_house", "biz_street_norm"))
    return linker


def fit(df: pd.DataFrame, *, addr_degrees: pd.DataFrame | None = None,
        aggregator_degree: int = AGGREGATOR_DEGREE, seed: int = 42):
    """Build + train the Splink model and return ``(linker, predictions)``.

    Predictions are made at a permissive threshold, so the caller can cluster at
    any higher threshold (e.g. to sweep against a gold set) without re-predicting.
    Pass ``addr_degrees`` (from :func:`address_degrees`) to mask on real full-
    population degrees; otherwise degree is computed within ``df`` (sample-only).
    """
    d = _mask_aggregators(df, addr_degrees, aggregator_degree)
    linker = _train_linker(d, seed)
    preds = linker.inference.predict(threshold_match_probability=0.5)
    return linker, preds


def fit_predict_full(train_df: pd.DataFrame, full_df: pd.DataFrame, *,
                     addr_degrees: pd.DataFrame | None = None,
                     aggregator_degree: int = AGGREGATOR_DEGREE, seed: int = 42):
    """Train on a dense slice, then predict over the full population.

    On a huge, mostly-unique population Splink's estimated "probability two random
    records match" (λ) collapses toward zero and the model *under-merges even
    identical records* (the low-λ trap). The fix is to learn m/u/λ on a dense
    representative slice (~2-3k records, where λ is well-conditioned) and transfer
    the trained model onto the full set for prediction — term-frequency adjustments
    still recompute over the full population, so name rarity is population-correct.

    Both frames are masked identically. Returns ``(linker_full, predictions)``;
    cluster/​build_portfolios/​feedback_merge consume these exactly as in the eval
    path. ``full_df`` should already be deduped on ``unique_id`` (identities), so a
    record in both the slice and the full set is not double-counted at predict.
    """
    import os
    import tempfile

    from splink import DuckDBAPI, Linker

    dtrain = _mask_aggregators(train_df, addr_degrees, aggregator_degree)
    train_linker = _train_linker(dtrain, seed)

    dfull = _mask_aggregators(full_df, addr_degrees, aggregator_degree)
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.json")
        train_linker.misc.save_model_to_json(model_path)
        # Load the trained m/u/λ onto a linker bound to the full population.
        linker_full = Linker(dfull[COMPARISON_COLS], model_path, db_api=DuckDBAPI())
        preds = linker_full.inference.predict(threshold_match_probability=0.5)
    return linker_full, preds


def cluster(linker, preds, threshold: float = 0.95) -> pd.DataFrame:
    """Cluster predictions into entities at ``threshold`` via Splink's own connected-
    components (ungated); adds ``cluster_id``. Prefer :func:`cluster_gated` — it adds
    the first-name veto below and needs no linker handle."""
    c = linker.clustering.cluster_pairwise_predictions_at_threshold(
        preds, threshold_match_probability=threshold)
    return c.as_pandas_dataframe()


def _addr_key(house, street_norm) -> str:
    """A normalized address token for the common-name veto: digits-only house +
    suffix-normalized street. Folds formatting variants (``140-06`` == ``14006``,
    ``56 BAY 34TH`` == ``56 BAY34``) but keeps genuinely different buildings apart."""
    import re
    h = re.sub(r"\D", "", house) if isinstance(house, str) else ""
    s = street_norm if isinstance(street_norm, str) else ""
    return h + "|" + s


def cluster_gated(preds, nodes, threshold: float = 0.95, *,
                  gamma_col: str = "gamma_first_name",
                  name_freq: pd.DataFrame | None = None,
                  name_cap: int = 15) -> pd.DataFrame:
    """Cluster with two precision vetoes, then connected-components.

    **First-name veto** (always on). Splink's pairwise score lets a strong exact-address
    match overpower a first-name disagreement, so two same-surname records at one office
    with *different* first names (JACOB vs JOSEF GUTMAN — different people) merge. Drop
    any edge whose first names genuinely disagree — Splink's own ``gamma_first_name == 0``
    level (both present, Jaro-Winkler below the lowest threshold). Typo/variant/prefix
    pairs sit at gamma ≥ 1 (STEVE/STEVEN, ZACH/ZACHARY, EFSTATHIOS/EFSTAHIOS) and a null
    first name is its own level, so both still merge. Trade-off: genuine nicknames that
    aren't near-matches (ABE/ABRAHAM) won't bridge — acceptable, precision-first.

    **Common-name veto** (on when ``name_freq`` is passed). An exact common full name
    repeated at a *different* address (two unrelated JIN CHENs) merges on name alone —
    the coincidence term-frequency under-penalizes. Drop an edge when the name is common
    (``> name_cap`` distinct identities for its (last_name, first_initial), from
    :func:`name_freq`) AND the two normalized addresses differ (:func:`_addr_key`). Rare
    names stay below the cap, so cross-office merges (Croman ~8) are untouched; same-
    address formatting variants share an address key, so those correct merges survive.
    This mirrors the loop's name-rarity gate, applied to the base linkage.

    ``nodes`` is the record frame (or a unique_id iterable) so every node — including
    singletons Splink would emit — gets a ``cluster_id``. Returns the node columns
    (the :data:`COMPARISON_COLS` present) plus ``cluster_id`` — a drop-in for
    :func:`cluster`'s output for scoring / :func:`build_portfolios` / :func:`feedback_merge`.
    """
    p = preds.as_pandas_dataframe() if hasattr(preds, "as_pandas_dataframe") else preds
    if gamma_col not in p.columns:
        raise ValueError(f"{gamma_col!r} not in predictions — fit() sets "
                         "retain_intermediate_calculation_columns=True so it is present")
    node_df = nodes if hasattr(nodes, "columns") else None
    ids = (node_df["unique_id"] if node_df is not None else pd.Series(list(nodes))).tolist()

    edges = p[(p["match_probability"] >= threshold) & (p[gamma_col] != 0)]

    if name_freq is not None:
        if node_df is None:
            raise ValueError("common-name veto needs `nodes` as a record frame")
        rar = {(r.last_name, r.first_initial): r.identities for r in name_freq.itertuples()}
        node_rar = {u: rar.get((ln, fi), 0) for u, ln, fi
                    in zip(node_df["unique_id"], node_df["last_name"], node_df["first_initial"])}
        akey = {u: _addr_key(h, s) for u, h, s
                in zip(node_df["unique_id"], node_df["biz_house"], node_df["biz_street_norm"])}
        # keep an edge unless the name is common AND the addresses genuinely differ
        keep = [(node_rar.get(a, 0) <= name_cap) or (akey.get(a) == akey.get(b))
                for a, b in zip(edges["unique_id_l"], edges["unique_id_r"])]
        edges = edges[pd.Series(keep, index=edges.index).values]

    parent = {u: u for u in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a, b in zip(edges["unique_id_l"], edges["unique_id_r"]):
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    cmap = {u: find(u) for u in ids}
    if node_df is not None:
        out = node_df[[c for c in COMPARISON_COLS if c in node_df.columns]].copy()
    else:
        out = pd.DataFrame({"unique_id": ids})
    out["cluster_id"] = out["unique_id"].map(cmap)
    return out.reset_index(drop=True)


def link(df: pd.DataFrame, *, threshold: float = 0.95, seed: int = 42):
    """Convenience: fit + cluster at one threshold. Returns ``(clusters_df, linker)``."""
    linker, preds = fit(df, seed=seed)
    return cluster(linker, preds, threshold), linker


def build_portfolios(df: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Splink clusters into resolved-entity portfolios — the entity ->
    portfolio stage. At the name+address level the entity *is* the portfolio: its
    building set is the union of its member records' ``bbls`` (this is what recovers
    Croman's / Rashad's WoW-split buildings into one portfolio). One row per entity:
    ``entity_id, name, n_records, n_bbls, bbls``."""
    m = clusters[["unique_id", "cluster_id"]].merge(
        df[["unique_id", "name_full", "bbls", "n_bbls"]], on="unique_id")
    rows = []
    for cid, g in m.groupby("cluster_id"):
        bbls = sorted({b for lst in g["bbls"] for b in (lst or [])})
        name = g.sort_values("n_bbls", ascending=False)["name_full"].iloc[0]
        rows.append({"entity_id": cid, "name": name, "n_records": len(g),
                     "n_bbls": len(bbls), "bbls": bbls})
    return (pd.DataFrame(rows).sort_values("n_bbls", ascending=False)
              .reset_index(drop=True))


def corp_owners_for(conn, bbls=None) -> pd.DataFrame:
    """Corporate co-owner names on buildings — the cross-office bridge. Returns
    (bbl, corp). E.g. CENTENNIAL PROPERTIES NY spans all of Croman's offices, so his
    740 Broadway and 424 W 51 entities share it. Pass ``bbls`` to scope to a set of
    buildings (an eval slice); pass ``None`` for every building (the full run — far
    cheaper than shipping a 170k-element array as a query parameter)."""
    base = ("SELECT DISTINCT btrim(r.bbl) AS bbl, upper(btrim(c.corporationname)) AS corp "
            "FROM hpd_contacts c JOIN hpd_registrations r ON r.registrationid = c.registrationid "
            "WHERE c.corporationname IS NOT NULL AND length(btrim(c.corporationname)) > 3")
    if bbls is None:
        return pd.read_sql(base + " AND r.bbl IS NOT NULL", conn)
    return pd.read_sql(base + " AND btrim(r.bbl) = ANY(%(bbls)s)", conn,
                       params={"bbls": list(bbls)})


def feedback_merge(df: pd.DataFrame, clusters: pd.DataFrame, corp_df: pd.DataFrame,
                   corp_deg: pd.DataFrame | None = None, corp_cap: int = 25,
                   name_freq: pd.DataFrame | None = None, name_cap: int = 15) -> pd.DataFrame:
    """One feedback-loop pass: merge pass-1 entities that are name-compatible
    (same last name + first initial) AND share a corporate co-owner across their
    buildings. This bridges the cross-office splits name+address alone can't
    (Croman's 740 Broadway <-> 424 W 51, via Centennial Properties). Two guards:

      * corp_cap — corps shared by more than this many distinct landlords
        (registered agents / big managers) are excluded (aggregator-corp cap).
      * name_cap — only names with at most this many distinct identities are
        bridged. A shared corp is often just a third-party MANAGER; for a rare name
        (Croman ~8) the pair is still surely one person, but for a common one
        (Smith 25, Chen 232) it isn't — so common names are never corp-bridged.

    Returns the clusters frame with a re-mapped ``cluster_id``."""
    from collections import defaultdict

    if corp_deg is not None:
        keep = set(corp_deg.loc[corp_deg["degree"] <= corp_cap, "corp"])
        corp_df = corp_df[corp_df["corp"].isin(keep)]
    rare = None
    if name_freq is not None:
        rare = set(map(tuple, name_freq.loc[name_freq["identities"] <= name_cap,
                                            ["last_name", "first_initial"]].to_numpy()))
    corp_by_bbl = corp_df.groupby("bbl")["corp"].apply(set).to_dict()
    m = clusters[["unique_id", "cluster_id", "last_name", "first_initial"]].merge(
        df[["unique_id", "bbls"]], on="unique_id")

    ent = {}
    for cid, g in m.groupby("cluster_id"):
        bbls = {b for lst in g["bbls"] for b in (lst or [])}
        corps = set().union(*(corp_by_bbl.get(b, set()) for b in bbls)) if bbls else set()
        last = g["last_name"].mode()
        init = g["first_initial"].mode()
        ent[cid] = {"corps": corps,
                    "last": last.iloc[0] if len(last) else None,
                    "init": init.iloc[0] if len(init) else None}

    parent = {cid: cid for cid in ent}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    # Block by (last name, first initial); only same-name entities can merge.
    groups = defaultdict(list)
    for cid, e in ent.items():
        if e["last"] and e["corps"]:
            groups[(e["last"], e["init"])].append(cid)
    for key, ids in groups.items():
        if rare is not None and key not in rare:
            continue                         # common name — never corp-bridge
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ent[ids[i]]["corps"] & ent[ids[j]]["corps"]:
                    parent[find(ids[i])] = find(ids[j])

    remap = {cid: find(cid) for cid in ent}
    out = clusters.copy()
    out["cluster_id"] = out["cluster_id"].map(lambda c: remap.get(c, c))
    return out
