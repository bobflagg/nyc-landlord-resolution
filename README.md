# NYC Landlord Resolution

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

## Who Owns What

JustFix's **[Who Owns What](https://github.com/JustFixNYC/who-owns-what)** (WoW) is the gold
standard for this task. It models the city's landlords as a graph of registration contacts
linked by shared names and business addresses, then clusters that graph (WCC + Louvain) into
portfolios — the backbone of countless tenant tools and news investigations. Read for what it
is, a WoW portfolio is a *registration network*: buildings tied through the people and offices
they register with. Where that tie is an owner's or a manager's office, it is an **operational
network** — the buildings run through the same hands. Two things can go wrong around it, and they
differ in kind:

- **Noise in the signal (false splits)** — one landlord fractured into several portfolios. A
  business-address typo, a second office, or a per-building shell LLC defeats the match, and an
  owner's own buildings scatter across unrelated groups. The question is still *who operates
  this?* — the measurement is just noisy. **This repo fixes that.**
- **A signal asked a second question** — a shared business address says two buildings register
  from the same place, and what that implies depends on the place. An owner's own office points to
  one owner-operator; a management office, to shared operation but not ownership; and a shared
  mailbox or service provider — an office suite many small landlords file from, a registered
  agent, a law firm — to little about either. The portfolio accurately records who registers
  where; read as an ownership claim, it attributes buildings to a party who doesn't own them.

**Don't ask one signal two questions.** The first problem is cleaned up by better record linkage.
The second isn't a matching problem at all — it needs a different question, answered by different
evidence (see [the follow-up](#ownership-gets-its-own-question) below).

## Fixing false splits

This repo targets the false-split failure mode. Instead of matching on exact
or near-exact fields, it resolves each owner *probabilistically* — Splink's Fellegi-Sunter
model scores every candidate pair on name and address agreement, learning from the data how
much a rare-surname match or a one-character address difference is worth. **Domenico
Antonelli**, a Queens landlord whose 10 buildings are registered from a handful of offices, is
split by WoW into **9 separate portfolios**. A single office, `150-115 POWELLS COVE BOULEVARD`,
is also typed `150-115 POWELLS COW B`, a hyphen is dropped from `146-48`, and the rest are
different offices, so each spawns its own portfolio. Scoring name and address together reunites
the variants (9 portfolios become 5 groups). A **second pass** then joins those groups when
their buildings share a company named on the registrations (here his managing agent) and the
name is rare, giving **one entity**. That second pass rests on management evidence rather than
a name or an address, and it is the least reliable stage in the labeled sample. Two unrelated
JIN CHENs at different addresses are not merged on name alone. The trace, step by step, with the
sampled breakdown of that pass, is in **[What-joins-Antonelli.ipynb](What-joins-Antonelli.ipynb)**.

> **[→ See it on the map](https://bobflagg.github.io/nyc-landlord-resolution/maps/antonelli.html)** —
> Antonelli's 10 buildings as one resolved owner, toggled against the 9 separate portfolios Who
> Owns What splits him into.

The engine is tuned to **avoid merging two different owners**, even at the cost of some recall:

- **Name-anchored blocking** — a surname match is required to even score a pair.
- **First-name veto** — JACOB and JOSEF at one office are different people, not merged.
- **Common-name veto** — a common name at different addresses is not merged on name alone
  (two unrelated JIN CHENs).
- **Aggregator-address masking** — a registered-agent office shared by many landlords is
  down-weighted, so office-mates aren't fused.
- **Shared-company pass, with guards** — only rare names, compatible first names, and a cap on
  how many contacts are listed under the shared company. A company can be an owner or a managing
  agent, and the pass does not distinguish them.

It drops into WoW **without changing its clustering algorithm** — it adds one high-confidence
edge type:

```python
# resolve owners, then add a "same owner" edge between the graph nodes that resolve together
owners = owner_index(conn)
for owner, nodes in group_nodes_by_owner(graph, owners).items():
    add_clique(graph, nodes, type="splink", weight=10.0)   # WCC/Louvain do the rest
```

Because edges only *add*, connected components can only *merge*; WoW's existing address-network
portfolios (a shell operation sharing one managing office) stay together, and WoW's own WCC and
Louvain steps run unchanged on the larger graph. An owner's scattered offices collapse into
one. In a live run this consolidated **~7,476 fragmented portfolios**; how often the added links
are right is measured
[below](#measured-against-who-owns-what). (This is the record-linkage engine built
for **[WatchlineNYC](https://github.com/bobflagg/WatchlineNYC)**, extracted to stand on its own.)

## How it measures up

Developed and tuned against a **105-record hand-adjudicated gold set** (`nlr/eval/gold_set.csv`) —
owner-contact records each labeled by hand to a true owner, scored pairwise and precision-first
because a false merge (naming the wrong landlord) is the costly error: pairwise
**precision ≈ 0.996** on that set, and the fragmented operators consolidate. The clustering
threshold was calibrated against this set, so 0.996 is an optimistic development-set figure, not an
out-of-sample result. The benchmark reproduces identically on either backend — Postgres or the
offline DuckDB snapshot.

Because that set was used for tuning, a second, blinded evaluation measures the
fix **against WoW directly** — how often a splink merge is right and how many real false-splits
it recovers. See [Measured against Who Owns What](#measured-against-who-owns-what) below.

## How it works

A step-by-step walkthrough of the resolution algorithm — following the arc of the official
Splink tutorial — lives in **[How-it-works.ipynb](How-it-works.ipynb)**. It traces one landlord
from raw HPD contacts through every stage: data prep and the name-anchored blocking rule, the
exploratory look at name rarity and aggregator addresses, the Fellegi-Sunter model fit and its
match-weight waterfall, the two precision vetoes and the shared-company feedback pass (a company
named on the registrations, owner or managing agent) that reunites an owner's scattered offices,
and finally the full-population `owner_index` scored against the gold set. It is the readable
companion to the code — the *why* behind each precision decision.
**[What-joins-Antonelli.ipynb](What-joins-Antonelli.ipynb)** is a second walkthrough that follows
one landlord's records through scoring, the base groups and the shared-company pass, with
waterfall charts of the pairs that decide the outcome.

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

The Postgres SQL is dialect-dispatched onto DuckDB, so results match — the development-set benchmark
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
  That is the cost of the precision-first vetoes.

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

## Ownership gets its own question

The second problem — a shared address read as shared ownership — is the subject of a follow-up
project,
**[NYC Beneficial Owner Resolution](https://github.com/bobflagg/nyc-beneficial-owner-resolution)**
("Watchline"). Where this engine asks *"are these the same name?"*, that one asks *"are these
the same beneficial owner?"* — resolving ownership from ownership evidence only (ACRIS deeds and
resolved registration identities) rather than from a shared address, and keeping an
operational-network layer, with aggregator hubs masked, as its own separate layer.

It is evaluated on precision only, on a stratified sample of candidate pairs frozen before labeling
and judged against primary records without being told which system or stratum produced each pair,
by one human annotator and an LLM second reader, with disagreements resolved by an LLM-conducted
review. Roughly 97% of links from a held multi-parcel deed and roughly 97% of resolved-identity
links were judged the same owner. Links through a deed later split into separate LLCs are the weak
spot, at roughly 70%, and roughly 98% of pairs sharing a masked aggregator address were correctly
kept apart. Recall is not estimated, and there is no head-to-head tally against WoW, because a WoW
portfolio is a co-registration network, not an ownership classifier. Between the two projects, the
operational signal gets cleaned up here and ownership gets its own answer there.

## Status / roadmap

- **Resolution engine + gold-set benchmark** over a Postgres holding the HPD tables.
- **DuckDB snapshot path** (done) — `python -m nlr.snapshot` then run offline over Parquet,
  no database. Same engine, same gold numbers.
- **Next — DuckDB-native over the public CSVs:** a loader that reads the HPD open-data CSVs
  directly (column-maps and constructs BBL), so `pip install` → point at the open data → run,
  with no Postgres ever needed to seed the snapshot.

## License

MIT.
