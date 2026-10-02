"""Score CONNECTED_BY_SPLINK's false-split impact against the frozen frame gold and print the
per-stratum table + the two headline numbers. Pure read of ``frame_gold.jsonl`` — no DB, no
splink re-run — so it reproduces the README's "Measured against Who Owns What" figures anywhere."""
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
    print("\n— HEADLINE —")
    if s2:
        print(f"CONNECTED_BY_SPLINK precision (S2_model, n={s2['same']+s2['diff']}): "
              f"{_pct(s2['precision_strict'])}  — {s2['same']}/{s2['same']+s2['diff']} merges correct, "
              f"{s2['diff']} false merges.")
        print(f"  net-new false-split fixes (gold SAME & WoW split): {s2['net_new_fix']}/{s2['n']}  "
              f"(WoW already merged {s2['wow_already_same']}).")
        print(f"  extrapolated to the {s2['frame_size']}-pair model frame: "
              f"~{s2['est_correct_merges']:,} correct merges; "
              f"~{s2['est_net_new_fixes']:,} net-new WoW false-splits recovered.")
    if s4:
        resid = fi.wilson(s4["same"], s4["same"] + s4["diff"])
        print(f"Residual missed splits (S4_hard_neg, n={s4['same']+s4['diff']}): "
              f"{resid[0]*100:.1f}% of unlinked same-surname pairs are truly the same owner "
              f"-> ~{round(s4['frame_size']*resid[0]):,} of {s4['frame_size']} unrecovered.")

    if s2 and s2["false_merges"]:
        print(f"\nfalse merges in S2 (gold DIFFERENT among splink's merges) — {len(s2['false_merges'])}:")
        for r in s2["false_merges"]:
            print(f"  {r['pair_id']}  {r.get('a_name','?'):<22} <> {r.get('b_name','?')}")


if __name__ == "__main__":
    main()
