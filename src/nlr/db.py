"""Database connection shim — the one seam to the data. Two backends:

* **Postgres** (:func:`pg_conn`) — point at any Postgres holding the raw HPD tables
  (``hpd_contacts`` + ``hpd_registrations``): load them with WoW's loader, a dump, or the
  NYC HPD open-data CSVs. Configure via a ``.env`` file or the environment:
  ``PGHOST  PGPORT  PGDATABASE  PGUSER  PGPASSWORD``.

* **DuckDB snapshot** (:func:`duckdb_conn`) — the no-Postgres path. Generate a local
  Parquet snapshot once with ``python -m nlr.snapshot`` (reads the Postgres above), then
  run the resolution/eval entirely offline over it. The resolution SQL is dialect-dispatched
  in :mod:`nlr.splink_source`, so the same ``owner_index`` / ``extract`` calls run unchanged.

:func:`default_conn` selects between them via the ``NLR_SNAPSHOT`` env var.

The remaining milestone ("north star") is a DuckDB loader straight over the public HPD CSVs —
``pip install`` → point at the open data, skip Postgres entirely; see the README.
"""
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def pg_conn():
    """A psycopg2 connection to the HPD Postgres, configured from the environment."""
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        keepalives=1,
    )


def duckdb_conn(hpd_contacts, hpd_registrations):
    """An in-process DuckDB connection exposing ``hpd_contacts`` / ``hpd_registrations``
    as views over local files — the no-Postgres path. Each argument is a path (or glob)
    to a ``.parquet`` or ``.csv`` file with the same columns as the Postgres tables.

    The resolution engine's SQL is dialect-dispatched (see ``splink_source``), so the same
    ``extract`` / ``owner_index`` calls run against this connection unchanged.
    """
    import duckdb

    con = duckdb.connect()
    reader = lambda p: (f"read_csv_auto('{p}')" if str(p).lower().endswith(".csv")
                        else f"read_parquet('{p}')")
    con.execute(f"CREATE VIEW hpd_contacts AS SELECT * FROM {reader(hpd_contacts)}")
    con.execute(f"CREATE VIEW hpd_registrations AS SELECT * FROM {reader(hpd_registrations)}")
    return con


def default_conn():
    """The connection the eval/CLI use: a DuckDB snapshot when ``NLR_SNAPSHOT`` points at a
    directory holding ``hpd_contacts.parquet`` / ``hpd_registrations.parquet`` (generate it
    with ``python -m nlr.snapshot``), otherwise a Postgres connection. Lets the same code run
    with or without a database."""
    snap = os.environ.get("NLR_SNAPSHOT")
    if snap:
        from pathlib import Path
        d = Path(snap)
        return duckdb_conn(d / "hpd_contacts.parquet", d / "hpd_registrations.parquet")
    return pg_conn()
