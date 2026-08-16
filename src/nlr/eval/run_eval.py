"""Fit the linkage on a representative slice, sweep clustering thresholds, and
score each against the gold set — so the F1-maximizing operating point is picked
by data, not by eyeballing four landlords."""
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
    conn = pg_conn()
    # A denser population than the full sample: Splink's "probability two random
    # records match" collapses toward zero on a huge mostly-unique set, which starves
    # the model. ~2k gives a stable prior while still carrying real name-rarity for TF.
    df = pd.concat([ss.extract(conn, TARGET_WHERE),
                    ss.extract(conn, NBR_WHERE),
                    ss.extract(conn, OFFICE_WHERE),
                    ss.extract(conn, "random() < 0.012")], ignore_index=True)
    degs = ss.address_degrees(conn)          # full-population aggregator degrees
    nf = ss.name_freq(conn)                   # name rarity for the common-name veto
    conn.close()
    df = df[df.contact_kind == "person"].drop_duplicates("unique_id").reset_index(drop=True)
    gold = scorer.load_gold(GOLD)
    print(f"records: {len(df)}   gold labels: {len(gold)}   addr-degree rows: {len(degs)}")

    linker, preds = ss.fit(df, addr_degrees=degs)
    table = scorer.sweep(preds, df, gold, thresholds=(0.80, 0.88, 0.92, 0.95, 0.98), name_freq=nf)
    print("\n" + table.to_string(index=False))

    best_t = float(table.loc[table.f1.idxmax(), "threshold"])
    clusters = ss.cluster_gated(preds, df, best_t, name_freq=nf)
    m, fmerge, missed = scorer.score(clusters, gold)
    print(f"\nbest F1 @ threshold {best_t}:  {m}")

    def show(pairs, label):
        print(f"\n{label} ({len(pairs)}):")
        for a, b in pairs[:6]:
            print(f"  {a['name_full']:<16} @ {a['biz_house']} {a['biz_street']:<20}"
                  f"  <>  {b['name_full']:<16} @ {b['biz_house']} {b['biz_street']}")
    show(fmerge, "false merges (precision errors)")
    show(missed, "missed merges (recall errors)")


if __name__ == "__main__":
    main()
