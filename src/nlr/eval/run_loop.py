"""Feedback loop, one iteration: Splink pass 1 (name+address) -> derive a
cross-office feature (shared corporate co-owner) -> merge name-compatible
entities that share one. Scores pass 1 vs pass 2 against the gold set so the
cross-office recall lift is measured.

  uv run python -m nlr.eval.run_loop
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("splink").setLevel(logging.ERROR)
from pathlib import Path

import pandas as pd

from nlr.db import pg_conn
from nlr import splink_source as ss
from nlr.eval import scorer
from nlr.eval.build_gold import TARGET_WHERE, NBR_WHERE, OFFICE_WHERE

GOLD = Path(scorer.__file__).parent / "gold_set.csv"


def main():
    # Common surnames added so the gate has common-name bridge candidates to suppress.
    COMMON = ("upper(btrim(c.lastname)) = ANY("
              "ARRAY['SMITH','CHEN','LEE','COHEN','GONZALEZ','WONG','KIM','PATEL','NGUYEN'])")
    conn = pg_conn()
    df = pd.concat([ss.extract(conn, TARGET_WHERE),
                    ss.extract(conn, NBR_WHERE),
                    ss.extract(conn, OFFICE_WHERE),
                    ss.extract(conn, COMMON),
                    ss.extract(conn, "random() < 0.012")], ignore_index=True)
    degs = ss.address_degrees(conn)
    df = df[df.contact_kind == "person"].drop_duplicates("unique_id").reset_index(drop=True)
    gold = scorer.load_gold(GOLD)

    nf = ss.name_freq(conn)                    # name rarity: gates the corp-bridge AND
                                               # the base linkage common-name veto
    linker, preds = ss.fit(df, addr_degrees=degs)
    pass1 = ss.cluster_gated(preds, df, threshold=0.9, name_freq=nf)

    # Cross-office feature: corporate co-owners on the entities' buildings.
    all_bbls = {b for lst in df["bbls"] for b in (lst or [])}
    corp = ss.corp_owners_for(conn, all_bbls)
    corp_deg = ss.corp_degrees(conn)          # cap aggregator corps (registered agents)
    conn.close()

    pass2 = ss.feedback_merge(df, pass1, corp, corp_deg=corp_deg, name_freq=nf)
    ungated = ss.feedback_merge(df, pass1, corp, corp_deg=corp_deg)   # no name gate

    # How many common-name blocks the gate stops the loop from bridging. Cluster
    # frames already carry last_name/first_initial, so group directly.
    base = pass1.groupby(["last_name", "first_initial"])["cluster_id"].nunique()
    def collapsed(cl):
        n = cl.groupby(["last_name", "first_initial"])["cluster_id"].nunique()
        return {k for k in n.index if n[k] < base.get(k, n[k])}
    freq = dict(zip(zip(nf.last_name, nf.first_initial), nf.identities))
    suppressed = {k for k in collapsed(ungated) - collapsed(pass2) if freq.get(k, 0) > 15}

    m1, _, _ = scorer.score(pass1, gold)
    m2, fp2, fn2 = scorer.score(pass2, gold)
    print(f"records {len(df)}   gold {len(gold)}   corp rows {len(corp)}")
    print(f"common-name bridges suppressed by the gate: {len(suppressed)} "
          f"(e.g. {sorted(suppressed)[:4]})\n")
    print(f"{'':<10}{'precision':>10}{'recall':>8}{'f1':>7}{'tp':>5}{'fp':>4}{'fn':>5}")
    for lbl, m in [("pass 1", m1), ("pass 2", m2)]:
        print(f"{lbl:<10}{m['precision']:>10}{m['recall']:>8}{m['f1']:>7}{m['tp']:>5}{m['fp']:>4}{m['fn']:>5}")

    # Croman before/after (his 740 Broadway should fold into the main entity)
    def croman_clusters(cl):
        c = cl.merge(df[["unique_id", "n_bbls"]], on="unique_id")
        c = c[c.last_name == "CROMAN"]
        return c.groupby("cluster_id")["n_bbls"].sum().sort_values(ascending=False)
    print("\nCROMAN clusters  pass1:", list(croman_clusters(pass1).values),
          " pass2:", list(croman_clusters(pass2).values))
    print(f"pass-2 false merges ({len(fp2)}):")
    for a, b in fp2[:6]:
        print(f"  {a['name_full']:<16} @ {a['biz_street']:<18} <> {b['name_full']:<16} @ {b['biz_street']}")


if __name__ == "__main__":
    main()
