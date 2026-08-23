"""Seed the entity-resolution gold set from the hard cases.

Extracts the target landlords (all their address/name variants) plus a capped set
of *aggregator-address decoys* — different landlords who share a target's office
address — and adjudicates each record to a ``gold_entity_id``. Run once to write
``gold_set.csv``; thereafter the CSV is the hand-maintained artifact (edit rows to
flip the debatable calls; don't regenerate over your edits).

The hard cases it encodes:
  * Croman's typo strandees ("4 WEST 51", "424 WEST 22") -> one entity   [must-merge]
  * Rashad's "666 BROADAWAY"/"BROADWQAY" typos            -> one entity   [must-merge]
  * STEVE/STEVEN, ZACH/ZACHARY variants                   -> one entity   [must-merge]
  * DIVYA vs JAMAL Rashad (same address, diff person)     -> two entities [must-NOT-merge]
  * different-first-name Castellanos (namesakes)          -> distinct     [must-NOT-merge]
  * aggregator-address neighbours                         -> distinct     [must-NOT-merge]
  * Scott Castellano's two offices / Kadden's offices     -> one entity   [DEBATABLE — flip if you disagree]
  * SHARED-OFFICE co-mates at one small address           -> distinct     [must-NOT-merge]
      Different landlords who file from the same non-aggregator office (223 & 18
      Spencer St, Williamsburg — ~19 and ~7 distinct surnames each). This is the
      population failure mode the rare-surname/masked-address gold above could NOT
      see: the old address-block fused same-initial office-mates (JOEL BRAVER +
      JOEL LEIFER; JACOB GUTMAN + JOSEF GUTMAN). Name-anchored linkage must keep
      them apart. GUTMAN JACOB vs JOSEF is a same-surname/different-person case —
      the common-surname discrimination axis, with evidence (two distinct people).

NOT yet in the gold (needs deed/ACRIS evidence to adjudicate, don't guess): hyper-
common full names repeated across unrelated addresses (MOHAMMAD ISLAM, YU LI) that
term-frequency under-penalizes. See CLAUDE.md "Not done".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nlr.db import pg_conn
from nlr import splink_source as ss

OUT = Path(__file__).parent / "gold_set.csv"
MAX_DECOYS = 20

TARGET_WHERE = "upper(btrim(c.lastname)) = ANY(ARRAY['CROMAN','RASHAD','CASTELLANO','KADDEN'])"
# Landlords sharing a target office address -> aggregator decoys (must-not-merge).
NBR_WHERE = """(
     (btrim(c.businesshousenumber)='254' AND upper(btrim(c.businessstreetname)) LIKE 'WEST 31%')
  OR (btrim(c.businesshousenumber)='575' AND upper(btrim(c.businessstreetname)) LIKE 'FIFTH AVE%')
  OR (btrim(c.businesshousenumber)='22'  AND upper(btrim(c.businessstreetname)) LIKE 'WEST 21%')
)"""

TARGET_NAMES = {"CROMAN", "RASHAD", "CASTELLANO", "KADDEN"}

# Shared-office co-mates: genuinely different landlords filing from one small
# (non-aggregator) office. Curated to one identity per distinct person (no typo-twins
# like GUTTMAN), so every within-set pair is a clean must-NOT-merge. GUTMAN keeps
# both JACOB and JOSEF — same surname, different people (the common-surname axis).
OFFICE_WHERE = """(
     (btrim(c.businesshousenumber)='223' AND upper(btrim(c.businessstreetname)) LIKE 'SPENCER%'
        AND upper(btrim(c.lastname)) = ANY(ARRAY['BRAVER','FREUND','GREENFIELD','GROSS','LEBOWITZ','LEIFER','GUTMAN']))
  OR (btrim(c.businesshousenumber)='18'  AND upper(btrim(c.businessstreetname)) LIKE 'SPENCER%'
        AND upper(btrim(c.lastname)) = ANY(ARRAY['JACOBOWITZ','KOHN','LANDAU','LEVY','SPITZER','KUBITSHUK']))
)"""


def office_key(ln: str, house: str, street: str) -> str:
    """A canonical office token that folds typo/apt variants of the SAME street
    address together but keeps genuinely different offices apart (conservative:
    different office = different entity)."""
    s, h = (street or ""), (house or "")
    if ln == "CROMAN":
        if "51" in s: return "w51"            # folds "424 WEST 51" + "4 WEST 51" (typo)
        if "22" in s and "WEST" in s: return "w22"
        if h.startswith("632"): return "632bway"
        if h.startswith("740"): return "740bway"
        if "BROADWAY" in s: return "424bway"
    if ln == "CASTELLANO":
        if "21" in s: return "w21"            # folds "22 WEST 21" + "22 22ND WEST 21"
        if "31" in s: return "w31"
    if ln == "RASHAD":
        if h.startswith("666") or "BROAD" in s: return "666bway"  # folds BROADAWAY/BROADWQAY/666 666
    if ln == "KADDEN":
        if h.startswith("520"): return "520mad"
        if h.startswith("575"): return "575fifth"
    return (h[:4] + "_" + s.split(" ")[0][:6]).strip("_")   # everything else: literal address


def adjudicate(r) -> tuple[str, str]:
    ln = (r.last_name or "").strip()
    fn = (r.first_name or "").strip()
    house = (r.biz_house or "").strip()
    street = (r.biz_street or "")
    # Shared-office co-mates (disjoint address space from the targets): each distinct
    # person is a distinct entity, so all within-office pairs are must-NOT-merge.
    if house in ("223", "18") and street.startswith("SPENCER"):
        key = f"{house}_{ln}_{fn}".lower().replace(" ", "_")
        return f"e_office_{key}", f"office-mate @ {house} spencer (must-NOT-merge)"
    ok = office_key(ln, r.biz_house, r.biz_street)
    # OWNER-LEVEL adjudication: the four target families are labeled per OWNER, not per
    # office. Same last name + same first name across offices = one operator — supported by
    # ACRIS multi-parcel co-ownership + a shared private corp (evidence pulled when flipped).
    # First names stay distinct, so genuinely different people (ANDREA vs SCOTT CASTELLANO,
    # DIVYA vs JAMAL RASHAD) remain separate. Was office-separate for Kadden/Rashad/Castellano
    # (the earlier "[DEBATABLE]" choice); flipped with evidence. `office_key`/`ok` now unused.
    if ln == "CROMAN":
        return "e_croman", "croman (one operator — all offices merged)"
    if ln == "KADDEN":
        # Zach/Zachary Kadden — one person; all offices merged.
        return "e_kadden", "kadden (one operator — offices merged)"
    if ln == "RASHAD":
        who = "divya" if fn.startswith("DIVYA") else ("jamal" if fn.startswith("JAMAL") else fn.lower())
        return f"e_rashad_{who}", f"rashad_{who} (one operator — offices merged)"
    if ln == "CASTELLANO":
        if fn == "SCOTT":
            # Choice NY Management (private, 8 landlords) bridges his 22 W 21 and 254 W 31
            # offices across 40 buildings — one operator (verified independently).
            return "e_castellano_scott", "castellano_scott (one operator — corp-bridged offices)"
        return f"e_castellano_{fn.lower()}", f"castellano_{fn.lower()} (one operator — offices merged)"
    # Decoys keyed on the person, so a decoy's own duplicate records are one entity.
    return f"e_nbr_{fn}_{ln}", "aggregator_neighbour (must-NOT-merge)"


def main() -> None:
    conn = pg_conn()
    df_t = ss.extract(conn, TARGET_WHERE)
    df_n = ss.extract(conn, NBR_WHERE)
    df_o = ss.extract(conn, OFFICE_WHERE)
    conn.close()

    df_t = df_t[df_t.contact_kind == "person"]
    df_o = df_o[df_o.contact_kind == "person"].drop_duplicates("unique_id")
    # Decoys = neighbours whose last name isn't a target; cap to keep the set small.
    decoys = df_n[(df_n.contact_kind == "person") & (~df_n.last_name.isin(TARGET_NAMES))]
    decoys = decoys.drop_duplicates("unique_id").head(MAX_DECOYS)

    df = (pd.concat([df_t, decoys, df_o], ignore_index=True)
            .drop_duplicates("unique_id").reset_index(drop=True))

    labeled = []
    for r in df.itertuples():
        eid, tag = adjudicate(r)
        labeled.append(dict(
            unique_id=r.unique_id, gold_entity_id=eid, case_tag=tag,
            name_full=r.name_full, biz_house=r.biz_house, biz_street=r.biz_street,
            biz_zip=r.biz_zip, n_bbls=int(r.n_bbls)))
    gold = pd.DataFrame(labeled).sort_values(["gold_entity_id", "name_full"])
    gold.to_csv(OUT, index=False)

    print(f"wrote {len(gold)} labeled records -> {OUT}")
    sizes = gold.groupby("gold_entity_id").size().sort_values(ascending=False)
    print(f"{gold.gold_entity_id.nunique()} gold entities "
          f"({(sizes > 1).sum()} multi-record, {(sizes == 1).sum()} singletons)")
    print(sizes.head(8).to_string())


if __name__ == "__main__":
    main()
