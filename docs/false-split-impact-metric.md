# False-split impact of CONNECTED_BY_SPLINK — metric design

**Status:** design only (no code yet). A metric to add to `nlr.eval`.

## The question

When `CONNECTED_BY_SPLINK` (nlr's `splink-fellegi-sunter` edges) merges two landlord nodes, is
that merge a **correct false-split fix** — and how many such fixes does it make? The unit is an
adjudicated **pair** (A, B); each side is a landlord node (owner name + bbls); the gold label is
`SAME | DIFFERENT | INDETERMINATE`.

nlr's job is *precision-first* resolution; this metric is how we quantify its effect on the
false-split problem against a **blind, preregistered, stratified** gold — the 512-pair adjudication
frame produced by `bor.eval.sample` and adjudicated in the `owner-review` app.

## Data flow

```
owner-review (blind pairwise annotation)
   └─ scripts/export_annotations.py > annotations.jsonl      # latest per (pair, annotator)
bor.eval  (holds the private frame + blinding key)
   └─ export_gold step  (this doc, §1)  →  eval_out/frame_gold.jsonl   # gold ⋈ frame metadata
nlr.eval  (this doc, §2)
   └─ frame_gold.jsonl (imported fixture)  →  per-stratum precision + population magnitude
```

nlr stays HPD-only: it never builds the frame or the gold — it consumes one gold fixture, exactly
as it already consumes `eval/gold_set.csv` today. The reweighted *population* estimate stays in
`bor.eval` (it owns the strata weights, the deed/aggregator strata, `divergence`/`gate`).

---

## 1. The bor-side export step → `frame_gold.jsonl`

**Where:** `bor/eval/export_gold.py` (new), writing `eval_out/frame_gold.jsonl`. Read-only over a
frozen frame + the handed-back annotations.

**Inputs**
- `annotations.jsonl` — handed back from owner-review (`scripts/export_annotations.py`). Rows:
  `{pair_id, annotator_id, label ∈ {SAME,DIFFERENT,INDETERMINATE}, evidence_tiers, c2_checks,
  severe, rationale, duration_ms, ts}`. It is already **latest per (pair, annotator)**.
- `eval_out/review_queue.jsonl` — `{pair_id, a:{ref,name,bbls}, b:{ref,name,bbls}}` (blinded).
- `eval_out/blinding_key.jsonl` — `{pair_id, stratum, signal, watchline, wow, a_nodeid, b_nodeid,
  a_bbls, b_bbls}`.
- `eval_out/frame_manifest.json` — `{frame, built_at, builder_commit, seed, strata:{S:{frame,n}}}`.

**Steps**
1. **Select the gold annotator(s).** Keep only human annotators; **exclude** any
   `annotator_id` starting `agent-` (the adjudicator agent is an evaluated subject, never gold).
   With one human (`bob-flagg`) the gold label is that row's `label`. With ≥2 humans, resolve by a
   stated rule (unanimous, or an adjudication pass) and record disagreements; do **not** silently
   majority-vote without recording κ.
2. **Keep only adjudicated pairs** — a pair with no gold annotation is simply absent (the metric is
   scored on the adjudicated subset; annotation is ongoing).
3. **Join on `pair_id`:** review_queue (names/bbls) + blinding_key (`stratum`, `signal`, `wow`,
   nodeids) + the gold `label`.
4. **Attach `frame_size`** per pair from `frame_manifest.strata[stratum].frame` (the population size
   of that stratum) — needed for extrapolation.
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
  "wow": null,                       // WoW baseline grouping if/when bor populates it (see §4)
  "a_nodeid": 8254, "b_nodeid": 56767,
  "annotator_id": "bob-flagg", "ts": "2026-09-…"
}
```
`a_name/a_bbls/b_name/b_bbls` are what let nlr map each side to its owner entity; the nodeids are
carried for cross-checking against bor but are not required by nlr.

### 1.2 `frame_gold.manifest.json` (provenance, for reproducibility)

`{frame, built_at, builder_commit (frame), bor_commit, owner_review_commit, exported_at,
gold_annotators:[…], per_stratum:{S:{frame_size, n_adjudicated}}}`. Freeze this with the number;
it pins the snapshot the precision was measured against.

---

## 2. The `nlr.eval` metric

**Where:** `nlr/eval/frame_impact.py` + CLI `nlr/eval/run_frame_impact.py`. Mirrors the existing
`scorer` (pairwise, precision-first, partial gold) but at the **pair** level.

### 2.1 Scorer

```
score_frame(gold_pairs, *, merged=None) -> {stratum -> report}
```
- `gold_pairs`: parsed `frame_gold.jsonl` rows.
- `merged`: optional `{pair_id -> bool}` — did nlr merge this pair? Two modes:
  - **Mode A — audit (default, no re-run).** Trust `signal`: `S2_model` pairs *are* the splink
    merges by construction, so their gold precision audits the exact edges nlr shipped when the
    frame was frozen. Pure function of the gold labels — the cleanest "impact of the shipped edges."
  - **Mode B — regression (optional).** Re-run nlr's resolution now; map each side `(name, bbl)` →
    nlr owner-id via the extraction's deterministic `unique_id` (the same explode-join
    `bor.splink_bridge` uses to map lwc nodes → splink entities); set
    `merged[pair] = owner_id(a) == owner_id(b)`. Tests *current* nlr and reports **drift** (pairs the
    frame calls splink-linked that today's model no longer merges, and vice versa).

### 2.2 Formulas (per stratum, INDETERMINATE first-class)

On the set of interest (S2_model, or in Mode B the pairs nlr merges):
- `precision_strict  = SAME / (SAME + DIFFERENT)`      (exclude INDETERMINATE)
- `precision_inclusive = SAME / (SAME + DIFFERENT + INDETERMINATE)`
- `coverage = 1 − INDETERMINATE / n`
- **Wilson 95% CI** on `precision_strict` with `n = SAME + DIFFERENT`.
- **Population magnitude:** `est_true_fixes = (SAME/n_adj) × frame_size` (+CI from the proportion).

### 2.3 The two headline numbers

- **S2_model precision = false-split-fix quality.** *"When splink merges, it's right P% (CI …)."*
  Population: `P_S2 × 11,742` ≈ correct portfolio merges.
- **S4_hard_neg residual = false splits still missed.** The **gold-SAME rate** on `signal=none`
  pairs (unlinked, same-surname) = merges nothing recovered. Population: `goldSAME(S4) × 7,083`.

Report a per-stratum table: `n_adjudicated, SAME/DIFF/INDET, precision_strict (+CI),
precision_inclusive, coverage, est_population`, plus a dump of the false-merge pairs (gold=DIFFERENT
among merged) for inspection — reuse the FP/FN idiom from `scorer.score`.

### 2.4 Report headline (example)

> **CONNECTED_BY_SPLINK false-split impact** (frame `accuracy-eval-4-strata`, S2_model, n=NN of 150):
> precision NN.N% (Wilson 95% CI …), coverage NN%. Extrapolated to the 11,742-pair model frame:
> ≈X,XXX correct portfolio merges. Residual (S4, n=NN): NN% of unlinked same-surname pairs are truly
> the same owner → ≈Y,YYY false splits still unrecovered.

---

## 3. Placement: nlr vs bor

- **nlr** reports **per-stratum precision** (S2 fix-quality, S4 residual) and the simple
  `rate × frame_size` magnitude — self-contained, HPD-scoped, one gold fixture in.
- **bor.eval** owns the **cross-stratum reweighted population** estimate and a true **recall** (which
  needs all same-owner strata combined with deed/aggregator — bor's `sample`/`score`/`divergence`
  machinery). nlr's number is "precision of my edges"; bor's is "population false-split reduction."

## 4. Gotchas (bake into docstrings + report footnotes)

- **Precision is clean; recall is not directly readable from the frame.** The frame is
  *signal-stratified* (S2 = exactly nlr's merges), so precision is honest but recall must be
  reweighted (S2 recovered-SAME vs S4 missed-SAME across frame sizes) — hence recall lives in bor.
- **Blind gold** (owner-review neutral dossier) → uncontaminated by nlr's output; safe to score nlr
  against. Note the *sampling* used nlr's edges, which is why precision (not recall) is the clean read.
- **`wow` baseline is null today.** S2 precision measures *splink-edge* precision, which equals
  *false-split-fix* precision only insofar as splink's edges are net-new over WoW (its design intent).
  For a literal false-split claim, gate on `wow` once bor populates it; until then footnote it and
  use `bor.eval.divergence` (the 760-crossing / 616-hiding numbers) for the net-new magnitude.
- **Data vintage / drift.** The frame is a frozen snapshot; Mode B may disagree on a few pairs if the
  model/threshold moved — report drift, don't hide it. Mode A avoids drift but only audits the frozen
  edges.
- **Partial annotation.** S2_model is n=150 and adjudication is ongoing → always print `n_adjudicated`
  and Wilson CIs; the estimate firms up as those 150 complete (prioritize them).
- **Mapping coverage.** ~99.9% of nodes map to a splink entity; a pair side that doesn't map is
  excluded — count and report it.
- **Exclude the agent.** Only human annotators are gold; `agent-*` annotations are the evaluated
  subject and must never enter `frame_gold.jsonl`.

## 5. Open items before implementing

1. Populate the `wow` baseline in `bor.eval.sample`/`blinding_key` (turns S2 precision into a literal
   false-split-fix precision; also lets the export gate on `wow=split`).
2. Finish adjudicating S2_model (150) and S4_hard_neg (120) for usable CIs.
3. Decide the multi-annotator resolution rule if a second human annotates (κ + adjudication).
