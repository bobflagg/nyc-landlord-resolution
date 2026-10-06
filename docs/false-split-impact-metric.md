# Link precision of CONNECTED_BY_SPLINK — metric design

(Formerly "False-split impact of CONNECTED_BY_SPLINK"; the file name is kept so existing links
keep working.)

**Status:** implemented. `nlr/eval/frame_impact.py` and `nlr/eval/run_frame_impact.py` score the
committed fixture `nlr/eval/frame_gold.jsonl`, which `bor.eval.export_gold` produces. **Scope:
precision only.** Recall is not estimated.

## The question

When `CONNECTED_BY_SPLINK` (nlr's `splink-fellegi-sunter` edges) links two landlord nodes, **how
often is the link right** (same owner), and how many of the correct links join pairs that Who Owns
What keeps in separate portfolios — the links a patch would add? The unit is an adjudicated
**pair** (A, B); each side is a landlord node (owner name + bbls); the gold label is
`SAME | DIFFERENT | INDETERMINATE`.

A WoW portfolio records who registers where, so it is a co-registration network and not an
ownership classifier. This metric therefore does **not** score WoW's portfolios, and it does not
tally nlr against WoW. The patch only adds edges, so the precision of the edges it adds is the
quantity to measure. A correct link between nodes in separate WoW portfolios reunites an owner's
fragmented registrations (a "net-new" link in this document).

The gold is a stratified sample drawn from a **frame frozen before labeling** (2026-09-20, seed 42):
the 512-pair frame produced by `bor.eval.sample` and judged in the `owner-review` app against
primary records (HPD registrations, ACRIS), without being told the model's score, the stratum, or
WoW's grouping. It is not described as preregistered: after the first results were seen, two
analyses that appeared in earlier drafts were dropped, a head-to-head tally against WoW and a
residual-recall estimate from the unlinked stratum (S4). Both ask a different question from link
precision.

## Data flow

```
owner-review (blind pairwise annotation)
   └─ scripts/export_annotations.py > annotations.jsonl      # latest per (pair, annotator)
bor.eval  (holds the private frame + blinding key)
   └─ export_gold step  (this doc, §1)  →  eval_out/frame_gold.jsonl   # gold ⋈ frame metadata
nlr.eval  (this doc, §2)
   └─ frame_gold.jsonl (committed fixture)  →  per-stratum precision + population magnitude
```

nlr stays HPD-only: it never builds the frame or the gold — it consumes one gold fixture, exactly
as it already consumes `eval/gold_set.csv`. The reweighted *population* estimate stays in
`bor.eval` (it owns the strata weights and the deed/aggregator strata).

---

## 1. The bor-side export step → `frame_gold.jsonl`

**Where:** `bor/eval/export_gold.py`, writing `eval_out/frame_gold.jsonl`. Read-only over a frozen
frame + the handed-back annotations.

**Inputs**
- `annotations.jsonl` — handed back from owner-review (`scripts/export_annotations.py`). Rows:
  `{pair_id, annotator_id, label ∈ {SAME,DIFFERENT,INDETERMINATE}, evidence_tiers, c2_checks,
  severe, rationale, duration_ms, ts}`. It is already **latest per (pair, annotator)**.
- `eval_out/review_queue.jsonl` — `{pair_id, a:{ref,name,bbls}, b:{ref,name,bbls}}` (blinded).
- `eval_out/blinding_key.jsonl` — `{pair_id, stratum, signal, watchline, wow, a_nodeid, b_nodeid,
  a_bbls, b_bbls}`.
- `eval_out/frame_manifest.json` — `{frame, built_at, builder_commit, seed, strata:{S:{frame,n}}}`.

**Steps**
1. **Select the gold annotator(s).** Keep only human annotators; **exclude** any `annotator_id`
   starting `agent-` (an LLM reader is an evaluated subject, never gold). The committed fixture is
   **one human annotator's labels** (`bob`; `gold_rule: single-human`, no disagreements to
   resolve). With ≥2 humans, resolve by a stated rule (unanimous, or an adjudication pass) fixed
   before the labels are seen, and record disagreements; do **not** silently majority-vote without
   recording agreement.
2. **Keep only adjudicated pairs** — a pair with no gold annotation is simply absent.
3. **Join on `pair_id`:** review_queue (names/bbls) + blinding_key (`stratum`, `signal`, `wow`,
   nodeids) + the gold `label`.
4. **Attach `frame_size`** per pair from `frame_manifest.strata[stratum].frame` — needed for
   extrapolation.
5. **Emit** one JSONL row per adjudicated pair (schema in §1.1) and a companion
   `frame_gold.manifest.json` (§1.2) for provenance.

**Signal values** (so nlr can select its own stratum): `splink-fellegi-sunter` (S2_model),
`acris-deed` (S1a), `acris-deed-linked-successor` (S1b), `aggregator-mask` (S3), `none` (S4).

### 1.1 `frame_gold.jsonl` row schema

```json
{
  "pair_id": "P0417",
  "stratum": "S2_model",
  "signal": "splink-fellegi-sunter",
  "label": "SAME",
  "a_name": "…", "a_bbls": ["…"],
  "b_name": "…", "b_bbls": ["…"],
  "frame_size": 11742,
  "wow": "DIFFERENT",               // WoW baseline: SAME if both sides share a wow.wow_portfolios portfolio
  "a_nodeid": 8254, "b_nodeid": 56767,
  "annotator_id": "bob"
}
```
`a_name/a_bbls/b_name/b_bbls` are what let nlr map each side to its owner entity; the nodeids are
carried for cross-checking against bor but are not required by nlr.

### 1.2 `frame_gold.manifest.json` (provenance, for reproducibility)

`{frame, built_at, builder_commit (frame), bor_commit, owner_review_commit, exported_at,
gold_annotators:[…], gold_rule, n_pairs, n_disagreements, wow_source, per_stratum:{S:{frame_size,
n_adjudicated}}}`. Freeze this with the number; it pins the snapshot the precision was measured
against.

---

## 2. The `nlr.eval` metric

**Where:** `nlr/eval/frame_impact.py` + CLI `nlr/eval/run_frame_impact.py`. Mirrors the existing
`scorer` (pairwise, precision-first) but at the **pair** level.

### 2.1 Scorer

```
score_frame(gold_pairs) -> {stratum -> report}
```
- `gold_pairs`: parsed `frame_gold.jsonl` rows.
- **Mode A — audit (implemented, no re-run).** Trust `signal`: `S2_model` pairs *are* the splink
  links by construction, so their gold precision audits the exact edges nlr shipped when the frame
  was frozen. A pure function of the gold labels; no database, no splink re-run.
- **Mode B — regression (not implemented).** Re-run nlr's resolution now, map each side
  `(name, bbl)` → nlr owner-id, and set `merged[pair] = owner_id(a) == owner_id(b)`, to test
  *current* nlr and report **drift**. `What-joins-Antonelli.ipynb` does a version of this mapping
  to classify sampled links by how they were linked.

### 2.2 Formulas (per stratum, INDETERMINATE first-class)

On the set of interest (S2_model):
- `precision_strict  = SAME / (SAME + DIFFERENT)`      (exclude INDETERMINATE)
- `precision_inclusive = SAME / (SAME + DIFFERENT + INDETERMINATE)`
- `coverage = 1 − INDETERMINATE / n`
- **Wilson 95% CI** on `precision_strict` with `n = SAME + DIFFERENT`.
- **Population magnitude:** `est_correct_links = precision × frame_size`; and the net-new share
  `net_new / n × frame_size`, where `net_new` = gold SAME and `wow == DIFFERENT`. Both are
  extrapolations from the sample, not measurements.

### 2.3 The headline numbers

- **S2_model precision.** *"When the model links two records, it is right P% of the time (CI …)."*
  Population: about `P × 11,742` correct links.
- **Precision of the net-new links.** Among sampled S2 pairs with `wow == DIFFERENT` (the two sides
  in separate WoW portfolios), the share judged SAME. The script prints it directly: 64 of 67
  (95.5%, Wilson 88–99%), computed from the fixture's `wow` field.

Report a per-stratum table: `n_adjudicated, SAME/DIFF/INDET, precision_strict (+CI),
precision_inclusive, coverage, est_population`, plus a dump of the wrong links (gold = DIFFERENT
among the links) for inspection.

The CLI also prints one line for the S4 stratum (the share of unlinked same-surname pairs judged the
same owner), labeled exploratory. That number depends on how the S4 frame was built, and it is not a
headline; the script no longer extrapolates it to a population count. Recall is not estimated here or anywhere in this evaluation, and the absence of a
link does not imply that two records are different owners.

### 2.4 Report headline (example)

> **CONNECTED_BY_SPLINK link precision** (frame `accuracy-eval-4-strata`, S2_model, n=150):
> precision 96.7% (Wilson 95% CI 92–99%), coverage 100%. Of the sampled links, 67 join pairs that WoW
> keeps in separate portfolios, and 64 of those were correct. Extrapolated to the 11,742-pair model
> frame: about 11,400 correct links, about 5,000 of which join separate WoW portfolios. Recall is not
> estimated.

---

## 3. Placement: nlr vs bor

- **nlr** reports **per-stratum precision** for the model stratum and the simple
  `rate × frame_size` magnitude — self-contained, HPD-scoped, one gold fixture in.
- **bor.eval** owns the **cross-stratum reweighted population** estimate for the deed, resolved-
  identity and aggregator strata (precision of the links bor asserts, and of pairs it keeps apart),
  using its `sample`/`score`/`divergence` machinery. nlr's number is "precision of my edges."
- Neither project estimates recall.

## 4. Gotchas (bake into docstrings + report footnotes)

- **Precision only.** The frame is *signal-stratified* (S2 = exactly nlr's links), so precision is a
  clean read. Recall would have to be reweighted across strata and is not estimated.
- **Blind gold** (owner-review neutral dossier) → uncontaminated by nlr's output. The *sampling* used
  nlr's edges, which is why precision (not recall) is the clean read.
- **Gold provenance.** The committed fixture is one human annotator's labels. A sensitivity
  analysis adds an LLM second reader and an LLM-conducted resolution of the disagreements
  (not published); it leaves the S2 figures unchanged. A blind second human reading and human
  arbitration are planned, and the figures are **provisional until then**. Do not describe the gold
  as blind human adjudication or report an inter-annotator agreement figure until a second human has
  labeled it.
- **`wow` baseline.** Populated at export from `wow.wow_portfolios` (a shared portfolio ⇒ SAME).
  S2 precision measures *link* precision; "net-new" means `wow == DIFFERENT`.
- **The split by stage is exploratory.** Classifying the net-new links as base-model (name and
  address) or shared-company second pass was done after the errors were known, on small samples
  (49 and 18 pairs). Treat it as a lead; confirm it on a fresh sample. It is computed in
  `What-joins-Antonelli.ipynb`, not by `run_frame_impact`.
- **Data vintage / drift.** The frame is a frozen snapshot; a re-run (Mode B) may disagree on a few
  pairs if the model or threshold moved. Report the drift. Mode A avoids drift but only audits the
  frozen edges.
- **Mapping coverage.** ~99.9% of nodes map to a splink entity; a pair side that doesn't map is
  excluded — count and report it. (All 67 net-new pairs mapped in the notebook.)
- **Exclude the LLM reader.** Only human annotators are gold; `agent-*` annotations are the
  evaluated subject and must never enter `frame_gold.jsonl`.

## 5. Open items

1. A blind second human reading and human arbitration of the disagreements, then regenerate the
   fixture, figures and manifest and update the README, landing page and chart.
2. A fresh labeled sample to test the exploratory stage split, and a decision on how the shared-
   company pass should ship (dropped, or its own edge type).
3. Mode B (drift against the current model), if the model or threshold changes.
