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

### Run without Postgres

Snapshot the two HPD tables once, then run the resolution and the eval entirely offline
over DuckDB — no database in the loop:

```bash
uv run python -m nlr.snapshot data        # Postgres -> data/*.parquet (~17 MB, run once)
NLR_SNAPSHOT=data uv run python -m nlr.eval.run_eval   # same eval, DuckDB backend
```

```python
from nlr import owner_index
from nlr.db import duckdb_conn

owners = owner_index(duckdb_conn("data/hpd_contacts.parquet",
                                 "data/hpd_registrations.parquet"))
```

The Postgres SQL is dialect-dispatched onto DuckDB, so results match — the gold benchmark
reproduces **P ≈ 0.996** on either backend. `owner_index`'s training slice is a *stratified*
sample (every identity with a same-name peer), computed in pandas over the extracted records —
identically on both backends — so there is no cross-backend divergence (the earlier
hash-sample caveat is gone).

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

## Measured against Who Owns What

The 105-record gold set above measures *internal* pairwise precision. A second, independent
evaluation measures the thing that matters for the integration: **when this engine adds a
"same owner" edge, how often is it right, and how many real false-splits does it fix that WoW
leaves fragmented?**

The test is a **preregistered, blinded, hand-adjudicated** stratified sample of the
model-linkage frame — the 11,742 candidate pairs splink scores but WoW's name/address
clustering does not already merge. 150 pairs were drawn, labeled blind by a human adjudicator
(SAME / DIFFERENT / INDETERMINATE) against the primary record, and only then unblinded and
scored against each pair's WoW decision.

- **Precision: 96.7%** (145 / 150 correct merges; Wilson 95% CI 92–99%). Five false merges.
- **64 of the 150** recover a split that WoW leaves fragmented (the two sides sit in separate
  WoW portfolios) — net-new false-split fixes, not merges WoW already had.
- Extrapolating the sample rate to the full frame: **≈5,000 WoW false-splits recovered** at
  ~97% precision, across ~11,400 correct merges.
- **Residual:** on a parallel sample of *unlinked* same-surname pairs, ~11% are in fact the
  same owner — splink's precision-first vetoes still miss roughly **780** recoverable splits.
  That is the cost of never fusing two different people.

The headline is robust to who labels it: precision and the false-split count are **unchanged**
when gold is restricted to a single human annotator (a second LLM reader agrees at κ = 0.89
and is excluded from gold), so the result does not rest on any model's judgment.

Reproduce it from the committed gold fixture — no database, no splink re-run:

```bash
uv run python -m nlr.eval.run_frame_impact      # per-stratum table + the headline above
```

> The gold is a frozen fixture (`nlr/eval/frame_gold.jsonl`): a blind, preregistered sample
> adjudicated in the owner-review app and exported by `bor.eval.export_gold` against the
> September 2026 HPD/ACRIS dump — provenance in `frame_gold.manifest.json`, full protocol and
> population math in [docs/false-split-impact-metric.md](docs/false-split-impact-metric.md). The
> figures above cover the model (splink) stratum — this engine's contribution. Deed-based linkage
> and the end-to-end head-to-head vs WoW belong to the broader Watchline system, not this library.

## Status / roadmap

- **Resolution engine + gold-set benchmark** over a Postgres holding the HPD tables.
- **DuckDB snapshot path** (done) — `python -m nlr.snapshot` then run offline over Parquet,
  no database. Same engine, same gold numbers.
- **Next — DuckDB-native over the public CSVs:** a loader that reads the HPD open-data CSVs
  directly (column-maps and constructs BBL), so `pip install` → point at the open data → run,
  with no Postgres ever needed to seed the snapshot.

## License

MIT.
