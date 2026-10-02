# NYC Landlord Resolution

**Precision-first probabilistic resolution of NYC landlord/owner identities** from public
HPD records. Given the raw owner contacts on HPD registrations, it decides *who is who* —
which records describe the same operator — so a single owner's buildings stop scattering
across separate "portfolios."

Runs on the two HPD tables alone (`hpd_contacts`, `hpd_registrations`). **No knowledge
graph, no geocoder.**

## Why accurate landlord portfolios matter

Knowing which buildings belong to the same landlord is the foundation of nearly every
data-driven housing investigation in New York. Journalists tracing a negligent operator
across the five boroughs, tenant organizers looking for others stuck with the same landlord,
watchdog agencies trying to target the worst actors, and researchers measuring patterns of
neglect all depend on it. Group a landlord's buildings correctly and a pattern — harassment,
deferred repairs, speculative flipping — becomes visible across the whole portfolio; get the
grouping wrong and accountability slips through the cracks: the operator with a hundred
buildings looks like a dozen small landlords, or an innocent namesake is saddled with someone
else's violations. Accurate portfolio identification is what turns scattered public records
into a tool for accountability.

## Who Owns What — the gold standard, and its two blind spots

JustFix's **[Who Owns What](https://github.com/JustFixNYC/who-owns-what)** (WoW) is the gold
standard for this task. It models the city's landlords as a graph of registration contacts
linked by shared names and business addresses, then clusters that graph (WCC + Louvain) into
portfolios — the backbone of countless tenant tools and news investigations. Because its
links are name/address matches, it is strong and conservative, but it has two characteristic
failure modes:

- **False splits** — one landlord fractured into several portfolios. A business-address typo,
  a second office, or a per-building shell LLC defeats the match, and an owner's own buildings
  scatter across unrelated groups.
- **False merges** — distinct landlords fused into one. A registered-agent office, a
  management company, or a law firm's address is shared by many unrelated owners, and the
  address link lumps them together.

The two errors pull in opposite directions, and fixing one naively worsens the other: loosen
the matching to recover splits and you create merges; tighten it to avoid merges and you
entrench the splits.

## Fixing the first: false splits

`nyc-landlord-resolution` targets the false-split failure mode. Instead of matching on exact
or near-exact fields, it resolves each owner *probabilistically* — Splink's Fellegi-Sunter
model scores every candidate pair on name and address agreement, learning from the data how
much a rare-surname match or a one-character address difference is worth. **Steven Croman**,
who surfaces as ~12 contact identities across 6 offices (one office a `424 WEST 51` vs
`4 WEST 51` typo away from another), resolves to a single entity of **127 buildings** — while
two unrelated JIN CHENs at different addresses never merge.

> **[→ See it on the map](https://bobflagg.github.io/nyc-landlord-resolution/maps/croman.html)** —
> Croman's 127 buildings as one resolved owner, toggled against the 6 separate portfolios Who
> Owns What splits him into.

The engine is tuned to **never merge two different owners**, even at the cost of a little
recall:

- **Name-anchored blocking** — a surname match is required to even score a pair.
- **First-name veto** — JACOB and JOSEF at one office are different people, not merged.
- **Common-name veto** — two unrelated JIN CHENs at different addresses never merge.
- **Aggregator-address masking** — a registered-agent office shared by many landlords is
  down-weighted, so office-mates aren't fused.

It drops into WoW **without changing its clustering at all** — it adds one high-confidence
edge type:

```python
# resolve owners, then add a "same owner" edge between the graph nodes that resolve together
owners = owner_index(conn)
for owner, nodes in group_nodes_by_owner(graph, owners).items():
    add_clique(graph, nodes, type="splink", weight=10.0)   # WCC/Louvain do the rest
```

Because edges only *add*, connected components only *merge*, never split: WoW's existing
address-nexus portfolios (a shell operation sharing one managing office) are preserved, while
an owner's scattered offices collapse into one. In a live run this consolidated **~7,476
fragmented portfolios** with zero namesake fusions. (This is the record-linkage engine built
for **[WatchlineNYC](https://github.com/bobflagg/WatchlineNYC)**, extracted to stand on its own.)

## How it measures up

Validated against a **105-record hand-adjudicated gold set** (`nlr/eval/gold_set.csv`) —
owner-contact records each labeled by hand to a true owner, scored pairwise and precision-first
because a false merge (naming the wrong landlord) is the costly error: pairwise
**precision ≈ 0.996**, **zero namesake fusions**, and the fragmented operators consolidate. The
benchmark reproduces identically on either backend — Postgres or the offline DuckDB snapshot.

That 105-record set measures *internal* precision; a second, blinded evaluation measures the
fix **against WoW directly** — how often a splink merge is right and how many real false-splits
it recovers. See [Measured against Who Owns What](#measured-against-who-owns-what) below.

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

## Measured against Who Owns What

The 105-record gold set above measures *internal* pairwise precision. A second, independent
evaluation measures the thing that matters for the integration: **when this engine adds a
"same owner" edge, how often is it right, and how many real false-splits does it fix that WoW
leaves fragmented?**

![Head-to-head on the model frame: on the 67 sampled splink merges where splink and JustFix
disagree, splink is right 64 times and JustFix 3 — McNemar p < 0.001.](docs/measured-vs-wow.svg)

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

## A follow-up — fixing the second: false merges

The other failure mode — false merges, where WoW fuses distinct landlords who merely share an
office — is the subject of a follow-up project,
**[beneficial-ownership-resolution](https://github.com/bobflagg/beneficial-ownership-resolution)**
("Watchline"). Where this engine asks *"are these the same name?"*, that one asks *"are these
the same beneficial owner?"* — pulling in ACRIS deeds, NYS DOS filings, and LLM-adjudicated
review to tell apart owners who only coincide at a registered-agent address, a managing agent,
or a law office.

On the same blind, preregistered 512-pair frame, its resolver beat WoW on **236 of the 259
pairs where the two disagree** (McNemar p < 0.001; annotator κ = 0.89), and correctly **split
~98%** of the pairs WoW groups only through a shared aggregator address. Between the two
projects, both of WoW's blind spots are covered — false splits here, false merges there.

## Status / roadmap

- **Resolution engine + gold-set benchmark** over a Postgres holding the HPD tables.
- **DuckDB snapshot path** (done) — `python -m nlr.snapshot` then run offline over Parquet,
  no database. Same engine, same gold numbers.
- **Next — DuckDB-native over the public CSVs:** a loader that reads the HPD open-data CSVs
  directly (column-maps and constructs BBL), so `pip install` → point at the open data → run,
  with no Postgres ever needed to seed the snapshot.

## License

MIT.
