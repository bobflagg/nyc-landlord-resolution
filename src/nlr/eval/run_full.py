"""Full-population run: train on a dense slice, resolve every owner-contact identity.

This is the validation-gate run (EXTRACTION.md #1/#2): does the linkage that scores
P~1.0 / R~0.95 on the eval slice still hold — and still consolidate Croman/Rashad —
when it resolves the *whole* HPD owner-contact population, not a 3k sample?

Structure:
  * extract the full person population (one row per identity, ~145k);
  * train the Splink model on the same dense slice the eval uses (stable λ), then
    predict/cluster over the full population (``fit_predict_full``);
  * score the gold subset (present in the full extract) to confirm quality holds;
  * run the feedback loop over the full population and report the portfolio
    distribution + the Croman/Rashad consolidation at scale;
  * export resolved portfolios to parquet — the hand-off surface for materializing
    ``Portfolio`` nodes / the standalone repo (EXTRACTION.md "integration boundary").

  uv run python -m nlr.eval.run_full
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("splink").setLevel(logging.ERROR)
import os
import time
from pathlib import Path

import pandas as pd

from nlr.db import pg_conn
from nlr import splink_source as ss
from nlr.eval import scorer
from nlr.eval.build_gold import TARGET_WHERE, NBR_WHERE

GOLD = Path(scorer.__file__).parent / "gold_set.csv"
EXPORT = os.environ.get("PORTFOLIOS_OUT",
                        str(Path(scorer.__file__).parent.parent / "portfolios_full.parquet"))
THRESHOLD = 0.95   # coded default in ss.cluster(); the eval sweep is flat 0.80–0.98

# Deterministic ~1.2% training slice (hash, not random()) so the full run — the
# export/materialization baseline — reproduces byte-for-byte apart from EM stochasticity.
DET_SAMPLE = ("abs(hashtext(coalesce(c.firstname,'')||coalesce(c.lastname,'')"
              "||coalesce(c.businesshousenumber,''))) % 1000 < 12")


def main():
    t0 = time.time()
    conn = pg_conn()

    # Full population — every owner-contact identity (person). Dedup on the
    # deterministic unique_id so a multi-BBL registration isn't double-counted.
    full = ss.extract(conn, "TRUE")
    full = full[full.contact_kind == "person"].drop_duplicates("unique_id").reset_index(drop=True)

    # Dense training slice (same composition the eval validated): targets + their
    # aggregator-address neighbours + a random ~1.2% sample. Trains m/u/λ where λ is
    # well-conditioned; the full population is what we then predict on.
    train = pd.concat([ss.extract(conn, TARGET_WHERE),
                       ss.extract(conn, NBR_WHERE),
                       ss.extract(conn, DET_SAMPLE)], ignore_index=True)
    train = train[train.contact_kind == "person"].drop_duplicates("unique_id").reset_index(drop=True)

    degs = ss.address_degrees(conn)     # full-population aggregator degrees
    corp_deg = ss.corp_degrees(conn)    # cap aggregator corps in the loop
    nf = ss.name_freq(conn)             # name rarity gate for the loop
    print(f"full persons: {len(full):,}   train slice: {len(train):,}   "
          f"addr-degrees: {len(degs):,}   [{time.time()-t0:.1f}s]", flush=True)

    # Train on the slice, predict over the full population.
    linker, preds = ss.fit_predict_full(train, full, addr_degrees=degs)
    print(f"trained + predicted full population   [{time.time()-t0:.1f}s]", flush=True)

    gold = scorer.load_gold(GOLD)
    # Confirm the operating point still holds at population scale.
    print("\nthreshold sweep (gold subset, scored on FULL-population clusters):")
    print(f"{'thr':>6}{'prec':>7}{'recall':>8}{'f1':>7}{'tp':>5}{'fp':>4}{'fn':>5}")
    for t in (0.90, 0.95, 0.98):
        cl_t = ss.cluster_gated(preds, full, threshold=t, name_freq=nf)
        m, _, _ = scorer.score(cl_t, gold)
        print(f"{t:>6}{m['precision']:>7}{m['recall']:>8}{m['f1']:>7}{m['tp']:>5}{m['fp']:>4}{m['fn']:>5}")

    pass1 = ss.cluster_gated(preds, full, threshold=THRESHOLD, name_freq=nf)
    print(f"\nclustered @ {THRESHOLD}   [{time.time()-t0:.1f}s]", flush=True)

    # Population-scale precision guard — the 87-record gold can't see this. Every
    # predicted pair shares a first initial, so a cluster with >1 distinct surname is
    # a different-person fusion (JOEL BRAVER + JACOB GUTMAN at one office). Name-
    # anchored blocking should keep this at 0; a non-zero count means the precision
    # hole is back (someone re-added an address-only blocking rule?).
    freq = {(r.last_name, r.first_initial): r.identities for r in nf.itertuples()}
    pg = pass1.assign(
        _akey=[ss._addr_key(h, s) for h, s in zip(pass1.biz_house, pass1.biz_street_norm)],
        _rar=[freq.get((ln, fi), 0) for ln, fi in zip(pass1.last_name, pass1.first_initial)])
    gp1 = pg.groupby("cluster_id")
    diff_surname = int((gp1["last_name"].nunique() > 1).sum())
    diff_first = int((gp1["first_name"].nunique() > 1).sum())
    # common name (> name_cap identities) spanning >1 normalized address = the leak
    common_leak = int(((gp1["_akey"].nunique() > 1) & (gp1["_rar"].max() > 15)).sum())
    sz = gp1.size()
    print(f"precision guard: {diff_surname:,} clusters with >1 surname (target 0), "
          f"{common_leak:,} common-name cross-address (target 0), {diff_first:,} with "
          f">1 first name, of {len(sz):,} clusters ({int((sz > 1).sum()):,} multi-record)")

    # Feedback loop over the full population: corp co-owners on ALL buildings.
    corp = ss.corp_owners_for(conn, None)
    conn.close()
    pass2 = ss.feedback_merge(full, pass1, corp, corp_deg=corp_deg, name_freq=nf)
    print(f"feedback loop done   corp rows {len(corp):,}   [{time.time()-t0:.1f}s]", flush=True)

    p1 = ss.build_portfolios(full, pass1)
    p2 = ss.build_portfolios(full, pass2)

    m1, _, _ = scorer.score(pass1, gold)
    m2, fp2, _ = scorer.score(pass2, gold)
    print(f"\ngold subset ({m1['labeled']} labeled records):")
    print(f"{'':<8}{'precision':>10}{'recall':>8}{'f1':>7}{'tp':>5}{'fp':>4}{'fn':>5}")
    for lbl, m in [("pass 1", m1), ("pass 2", m2)]:
        print(f"{lbl:<8}{m['precision']:>10}{m['recall']:>8}{m['f1']:>7}{m['tp']:>5}{m['fp']:>4}{m['fn']:>5}")

    def dist(p, lbl):
        multi = p[p.n_records > 1]
        print(f"\n{lbl}: {len(p):,} portfolios  "
              f"({len(multi):,} multi-record, {len(p)-len(multi):,} singletons)  "
              f"largest {int(p.n_bbls.max()):,} bbls / {int(p.n_records.max()):,} records")
    dist(p1, "pass 1")
    dist(p2, "pass 2")

    # The headline targets: total buildings and how many clusters they land in.
    def target(cl, ln):
        c = cl.merge(full[["unique_id", "n_bbls"]], on="unique_id")
        c = c[c.last_name == ln]
        g = c.groupby("cluster_id")["n_bbls"].sum().sort_values(ascending=False)
        return list(g.values)
    for ln in ("CROMAN", "RASHAD", "CASTELLANO"):
        print(f"{ln:<11} bbl-count per cluster  pass1: {target(pass1, ln)}  "
              f"pass2: {target(pass2, ln)}")

    p2.to_parquet(EXPORT)
    print(f"\nexported {len(p2):,} resolved portfolios -> {EXPORT}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
