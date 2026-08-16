"""Database connection shim — the one seam to the data.

Point this at any Postgres holding the raw HPD tables (``hpd_contacts`` +
``hpd_registrations``): load them with WoW's loader, a dump, or the NYC HPD open-data
CSVs. Configure via a ``.env`` file or the environment:

    PGHOST  PGPORT  PGDATABASE  PGUSER  PGPASSWORD

(A DuckDB-over-CSV loader — ``pip install`` → point at the public CSVs, no Postgres — is
the planned "north star"; see the README. This Postgres path is v1.)
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
