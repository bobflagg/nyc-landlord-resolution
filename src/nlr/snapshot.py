"""Generate a local Parquet snapshot of the HPD tables — the no-Postgres path.

Reads ``hpd_contacts`` / ``hpd_registrations`` from the configured Postgres
(:func:`nlr.db.pg_conn`) and writes two Parquet files that a DuckDB connection
(:func:`nlr.db.duckdb_conn`) serves as those same tables. Run it **once**; then the
resolution, the eval, and ``owner_index`` run entirely offline against the snapshot —
no database, no geocoder.

    uv run python -m nlr.snapshot [OUTDIR]          # default OUTDIR: ./data

Then, offline:

    NLR_SNAPSHOT=./data uv run python -m nlr.eval.run_eval
    # or, in code:
    from nlr.db import duckdb_conn
    from nlr import owner_index
    idx = owner_index(duckdb_conn("data/hpd_contacts.parquet",
                                  "data/hpd_registrations.parquet"))

Only the columns the resolution SQL reads are exported — that keeps the snapshot small
(~15 MB + ~2 MB) and doubles as the exact schema the planned Stage-2 public-CSV loader
must materialize. Parquet is written through DuckDB, so no pyarrow dependency is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The minimal columns the extraction SQL touches (see nlr.splink_source). This is the
# contract for any backend that wants to serve hpd_contacts / hpd_registrations.
CONTACT_COLS = ["registrationid", "type", "firstname", "lastname", "corporationname",
                "businesshousenumber", "businessstreetname", "businessapartment",
                "businesscity", "businessstate", "businesszip"]
REG_COLS = ["registrationid", "bbl", "registrationenddate", "lastregistrationdate"]

TABLES = {"hpd_contacts": CONTACT_COLS, "hpd_registrations": REG_COLS}


def export(outdir: str | Path = "data") -> Path:
    """Export the HPD tables from Postgres to ``outdir`` as Parquet. Returns ``outdir``."""
    import duckdb
    import pandas as pd

    from .db import pg_conn

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pg = pg_conn()
    d = duckdb.connect()
    try:
        for table, cols in TABLES.items():
            print(f"  exporting {table} ...", flush=True)
            df = pd.read_sql(f"SELECT {', '.join(cols)} FROM {table}", pg)
            d.register("t", df)
            dest = out / f"{table}.parquet"
            d.execute(f"COPY t TO '{dest}' (FORMAT PARQUET)")
            d.unregister("t")
            print(f"    wrote {dest}  ({dest.stat().st_size // 1_000_000} MB, {len(df):,} rows)")
    finally:
        pg.close()
        d.close()
    print(f"snapshot ready in {out}/  —  run offline with  NLR_SNAPSHOT={out}")
    return out


def main() -> None:
    export(sys.argv[1] if len(sys.argv) > 1 else "data")


if __name__ == "__main__":
    main()
