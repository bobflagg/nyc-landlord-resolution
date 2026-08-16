# nyc-landlord-resolution

**Precision-first probabilistic resolution of NYC landlord/owner identities** from public
HPD records. Given the raw owner contacts on HPD registrations, it decides *who is who* —
which records describe the same operator — so a single owner's buildings stop scattering
across separate "portfolios."

Runs on the two HPD tables alone (`hpd_contacts`, `hpd_registrations`). **No knowledge
graph, no geocoder.**

## The problem

NYC landlord registrations fragment one operator across many records — typo'd names and
business addresses, multiple offices, per-building shell LLCs. Matching that gates on exact
or near-exact fields inherits the fragmentation: **Steven Croman shows as ~6 separate
operators** because a business-address typo (`4 WEST 51` vs `424 WEST 51`) or a second
office defeats the match. This library resolves the owner *probabilistically* (Splink /
Fellegi-Sunter), so Croman's ~12 contact identities across 6 offices resolve to **one entity
of 127 buildings** — while never fusing two different people.

## Precision-first, by design

The resolution is tuned so it **never merges two different owners**, even at the cost of a
little recall:

- **Name-anchored blocking** — a surname match is required to even score a pair.
- **First-name veto** — JACOB and JOSEF at one office are different people, not merged.
- **Common-name veto** — two unrelated JIN CHENs at different addresses never merge.
- **Aggregator-address masking** — a registered-agent office shared by many landlords is
  down-weighted, so office-mates aren't fused.

Validated against a **105-record hand-adjudicated gold set** (`nlr/eval/gold_set.csv`):
pairwise **P ≈ 0.996**, **zero namesake fusions**, and the fragmented operators
consolidate.

## Use it

```python
from nlr import owner_index
from nlr.db import pg_conn

with pg_conn() as conn:
    owners = owner_index(conn)          # {(NORMALIZED_NAME, bbl): owner_id}
```

`owner_index` runs the full-population resolution once (~1–2 min) and returns a lookup: two
contacts with the same `owner_id` are the same operator. That's the whole API — identity
resolution, nothing else. What you *do* with it (count buildings per owner, group a graph
into portfolios, …) is up to you. Normalize your own names with `nlr.normalize_name` before
looking up `(name, bbl)`.

## Evaluate

```bash
uv sync
cp .env.example .env          # set PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD
uv run python -m nlr.eval.run_eval     # fit + threshold sweep + score vs the gold set
uv run python -m nlr.eval.run_full     # full-population run + precision guard + export
```

## Where it comes from — and the Who Owns What integration

This is the record-linkage engine built for **[WatchlineNYC](https://github.com/bobflagg/WatchlineNYC)**,
extracted to stand on its own. It pairs naturally with JustFix's
**[Who Owns What](https://github.com/JustFixNYC/who-owns-what)** (WoW): WoW already models
portfolios as a graph of landlord nodes linked by name/address matches, then clusters them
(WCC + Louvain). Its one blind spot is an owner's *own* typo'd/multi-office fragments — which
is exactly what this engine resolves.

The integration is deliberately tiny and **does not change WoW's clustering at all** — it
adds one high-confidence edge type:

```python
# resolve owners, then add a "same owner" edge between the graph nodes that resolve together
owners = owner_index(conn)
for owner, nodes in group_nodes_by_owner(graph, owners).items():
    add_clique(graph, nodes, type="splink", weight=10.0)   # WCC/Louvain do the rest
```

Because edges only *add*, connected components only *merge*, never split: WoW's existing
address-nexus portfolios (a shell operation sharing one managing office) are preserved, while
an owner's scattered offices collapse into one. In a live run this consolidated ~7,476
fragmented portfolios with zero namesake fusions.

## Status / roadmap

- **v1 (here):** the resolution engine + gold-set benchmark, over a Postgres holding the HPD
  tables.
- **Next:** a **DuckDB-native** path — point at the public HPD open-data CSVs, `pip install`
  → run, no database at all — and a bundled sample so it runs on clone.

## License

MIT.
