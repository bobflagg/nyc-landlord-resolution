"""Score the precision of CONNECTED_BY_SPLINK's links against the frozen frame gold and print the
per-stratum table + the model-stratum headline. Pure read of ``frame_gold.jsonl`` — no DB, no
splink re-run — so it reproduces the README's "Measured against Who Owns What" figures anywhere.
Precision only; recall is not estimated."""
import json
from pathlib import Path

from nlr.eval import frame_impact as fi

GOLD = Path(fi.__file__).parent / "frame_gold.jsonl"
MANIFEST = Path(fi.__file__).parent / "frame_gold.manifest.json"


def _pct(w):
    return f"{w[0]*100:4.1f}% [{w[1]*100:.1f},{w[2]*100:.1f}]"


def main():
    rows = fi.load_frame_gold(GOLD)
    prov = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    reports = fi.score_frame(rows)

    gann = ", ".join(prov.get("gold_annotators", [])) or "?"
    print(f"frame={prov.get('frame','?')}  gold={prov.get('gold_rule','?')} ({gann})  "
          f"exported={prov.get('exported_at','?')}  pairs={len(rows)}\n")

    hdr = f"{'stratum':<28}{'n':>4}{'SAME':>6}{'DIFF':>6}{'INDET':>6}  {'precision (strict)':<22}{'cover':>7}{'frame':>8}"
    print(hdr); print("-" * len(hdr))
    for st, m in reports.items():
        print(f"{st:<28}{m['n']:>4}{m['same']:>6}{m['diff']:>6}{m['indet']:>6}  "
              f"{_pct(m['precision_strict']):<22}{m['coverage']*100:>6.1f}%{m['frame_size']:>8}")

    s2 = reports.get("S2_model")
    s4 = reports.get(fi.RESIDUAL_STRATUM)
    print("\n— HEADLINE (precision only; recall is not estimated) —")
    if s2:
        print(f"CONNECTED_BY_SPLINK link precision (S2_model, n={s2['same']+s2['diff']}): "
              f"{_pct(s2['precision_strict'])}  — {s2['same']}/{s2['same']+s2['diff']} links correct, "
              f"{s2['diff']} wrong.")
        print(f"  links that join separate WoW portfolios (net-new): {s2['net_new_fix']}/{s2['net_new_decided']} correct  "
              f"{_pct(s2['net_new_precision'])}  (WoW already groups {s2['wow_already_same']} of the correct links).")
        print(f"  extrapolated to the {s2['frame_size']}-pair model frame: "
              f"~{s2['est_correct_merges']:,} correct links; "
              f"~{s2['est_net_new_fixes']:,} of them join separate WoW portfolios.")
    if s4:
        resid = fi.wilson(s4["same"], s4["same"] + s4["diff"])
        print(f"Exploratory, not a headline (S4_hard_neg, n={s4['same']+s4['diff']}): "
              f"{resid[0]*100:.1f}% of unlinked same-surname pairs were judged the same owner.")

    if s2 and s2["false_merges"]:
        print(f"\nwrong links in S2 (gold DIFFERENT among splink's links) — {len(s2['false_merges'])}:")
        for r in s2["false_merges"]:
            print(f"  {r['pair_id']}  {r.get('a_name','?'):<22} <> {r.get('b_name','?')}")


if __name__ == "__main__":
    main()
