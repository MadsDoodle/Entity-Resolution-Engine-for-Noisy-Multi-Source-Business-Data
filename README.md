# Large-Scale Business Entity Resolution across Multi-Source Records

Resolving which business records, scattered across three independent and noisy
data sources with **no shared identifiers**, refer to the same real-world
business — at a scale of ~1.7M reference entities against a ~10M-record pool.

Built for a 72-hour ML hackathon. Scored on **macro-averaged F0.5** (precision
weighted 2× over recall), the pipeline reaches **~0.947 F0.5** on a held-out
validation split.

---

## Problem

Business identity data arrives from multiple independent sources, each a
partial, noisy fragment of the same real-world entities. The sources share no
common keys, so deciding which records refer to the same business is an
**Entity Resolution (ER)** problem.

- **Source 1** is the deduplicated reference source.
- For every Source-1 entity, find all matching records in **Source 2** and
  **Source 3**. An entity may match zero, one, or many records.
- Fields are `business_name`, `business_address`, `country` — all noisy.
- `country` is an **open set**: train covers US + India, but the test set adds
  **France**, which never appears in training. Nothing is hard-coded to a fixed
  country list.

**Noise we handle:** abbreviations (Corp/Corporation, Pvt/Private, Rd/Road),
legal-suffix inconsistencies, punctuation (`&` vs `and`), word-order
transpositions, typos, transliterations (Latin ↔ Devanagari / Tamil / Telugu /
Bengali / … ), partial and landmark-based addresses, municipal numbering, and
missing components.

**Why F0.5?** In real ER, wrongly merging two distinct businesses (false
positive) is more damaging than missing a link (false negative). F0.5 rewards
precision. Correctly leaving a singleton empty earns a full 1.0 on that entity,
so detecting "no match" matters as much as finding matches.

```
F0.5 = (1.25 × P × R) / (0.25 × P + R)   — computed per entity, then macro-averaged
```

---

## Results

| Metric | Value |
|---|---|
| Validation macro-F0.5 | **~0.947** |
| Blocking recall (candidate ceiling) | **~94%** |
| Source-1 entities (test) | 1,732,544 |
| Candidate pool (test S2 + S3) | ~9.97M records |
| Entities with ≥1 predicted match | 1,629,312 |
| Singletons (predicted no match) | 103,232 |

All models are **MIT-licensed and < 8B parameters**. **No external data, APIs,
or lookups** are used at any stage — only the provided training data.

---

## Approach (one line)

Multilingual embedding blocking (**BGE-M3**) unioned with IDF-weighted
token / phonetic / address inverted indexes → **LightGBM** pairwise matcher
with distractor-aware features → **one-to-one assignment** + per-entity
**expected-F0.5** cluster-size selection.

### Pipeline

```
raw TSVs
   │
   ▼  Phase 2 — Preprocessing
 normalize name + address (shared for S1/S2/S3, train + test)
   │
   ▼  Phase 3 — Blocking / candidate generation      [GPU]
 BGE-M3 embeddings + GPU-FAISS ANN
   ∪ IDF-weighted token index
   ∪ Metaphone phonetic index
   ∪ address inverted index
   → ranked, capped candidate list per entity   →  candidate_pairs.tsv
   │
   ▼  Phase 4 — Matcher training
 ~38 pairwise features → LightGBM + isotonic calibration
   │
   ▼  Phase 5 — Scoring + decision
 score candidates → one-to-one ownership → expected-F0.5 set selection
   → matching_results.tsv
```

---

## Key design decisions

### Preprocessing — build normalization from the data, not assumptions
- **Unicode-safe cleaning.** NFKC normalize, lowercase, `&`/`+` → `and`,
  strip danda / smart quotes / leading & trailing junk — while **preserving**
  non-Latin scripts (Devanagari, Tamil, Telugu, Bengali, Odia, …). Verified
  **idempotent** (`clean(clean(x)) == clean(x)`) and that no field is emptied.
- **Suffix canonicalization over stripping.** Legal suffixes are collapsed to a
  single canonical token (`ltd/limited/lim → limited`) and *kept* in the name.
  An A/B test showed this lifts true-pair Jaccard (0.638 vs 0.618) without
  raising collision similarity — more informative than deleting the suffix.
  Entity type (llc / llp / inc / …) is also emitted as a side feature.
- **Category-word guard.** Words like `traders`, `foods`, `solutions` are
  protected from suffix logic so distinct businesses aren't merged on a shared
  descriptor.
- **Order-invariant address core.** Expand street abbreviations
  (`rd → road`, `ave → avenue`), normalize leading zeros, drop `N/A`/junk,
  pull landmarks (`near`, `opp`, `behind`) into a *separate* field, then sort
  tokens — so component reordering doesn't hurt overlap. State code is a side
  feature, never injected into the core (that hurt idempotency for ~0 gain).
- **Explicit missing-data contract.** ~3.3% of S2/S3 rows have no address; these
  route to a verified **name-only** matching mode instead of being dropped.

### Blocking — maximize recall, then cut the search space hard
Blocking sets the recall ceiling, so it unions complementary signals rather
than relying on one:
- **BGE-M3 multilingual embeddings** (fp16, cached to a memmap) are the
  workhorse — they bridge transliteration and heavy paraphrase that string
  methods miss. A **GPU-FAISS** flat-IP index is built **per country** and
  queried top-K for every Source-1 entity.
- **IDF-weighted token index** rewards rare, discriminative brand tokens.
- **Metaphone phonetic index** catches typo / spelling variants that share a
  sound.
- **Address inverted index** adds address-token overlap.
- High-frequency postings are **capped (MAXP=300)** so generic tokens don't
  explode the candidate set; candidates are scored (embedding weighted highest),
  ranked, and **capped per entity** (400 for training, 150 at inference for a
  France-safe recall margin).
- Everything is **cached and resumable** — the expensive full-pool embedding is
  a one-time cost; downstream re-runs skip it.

Result: ~94% recall while reducing the match space by 5–6 orders of magnitude.

### Matching — a calibrated LightGBM scorer with ~38 features
- **String similarity** (RapidFuzz, C-level batched via `process.cdist`): ratio,
  token-sort, token-set, partial, no-space partial — for both name and address.
- **Set / overlap:** name & address Jaccard, digit-set Jaccard.
- **Signal provenance:** which blocking stage found the pair, embedding score,
  rank position, rank margin, candidate count.
- **Distractor-lift features** — the sharpest precision gain. The generator
  builds hard negatives by appending specific tokens; mining token *lift* in
  unmatched vs matched pool names fingerprints these. Features over the tokens a
  candidate *adds* vs the Source-1 name (max/sum distractor lift, count of
  distractor vs benign additions, "added-only-distractor") pushed validation
  F0.5 from ~0.856 to ~0.947.
- **House-number drift** — the generator jitters house numbers by a small amount;
  `hn_equal` / `hn_close_jitter` / `hn_diff` capture that tell.
- Trained with **entity-grouped** splits (no leakage across the train/val
  boundary) and **isotonic-calibrated** so probabilities are threshold-stable.
  Negatives are capped per entity; any true pair blocking missed is force-added.

### Decision layer — optimize the metric directly
- **One-to-one ownership:** each Source-2/3 record is assigned to the single
  Source-1 entity that scores it highest, preventing one record leaking into
  many clusters.
- **Expected-F0.5 set selection:** rather than one global threshold, each
  entity's match set is chosen by maximizing *expected* F0.5 over its calibrated
  probabilities (Monte-Carlo over candidate inclusion). This naturally produces
  empty sets for singletons — worth a full 1.0 each under the metric.

---

## Repository structure

```
.
├── README.md                               # this file
├── Amazon_Challenge (2).ipynb              # full working notebook (Colab / A100)
├── student_resource/                       # challenge data + validator
│   ├── dataset/
│   │   ├── train/   train_source{1,2,3}.tsv, train_ground_truth.tsv
│   │   └── test/    test_source{1,2,3}.tsv
│   └── utils/validate_submission.py        # format validator (stdlib only)
└── submission/
    ├── Documentation_template.md           # methodology write-up
    ├── output/
    │   ├── matching_results.tsv            # final matches (leaderboard file)
    │   └── candidate_pairs.tsv             # blocking candidate set
    └── code/business_entity_resolution/
        ├── README.md                       # reproduction guide
        ├── requirements.txt                # pinned deps
        └── src/
            ├── phase2_preprocess.py        # normalization
            ├── phase3_blocking.py          # candidate generation
            ├── phase4_train_matcher.py     # LightGBM + calibration
            ├── phase5_generate.py          # scoring + decision + output
            └── metric_f_beta.py            # F0.5 scorer
```

> Dataset size note: the `.tsv` files under `student_resource/dataset/` are
> large (tens of millions of rows). If you don't want them in git history, add
> `student_resource/dataset/` to `.gitignore` and document where to fetch them.

---

## Reproduce

**Environment:** Python 3.10+, CUDA 12 GPU (A100 used), ~64 GB RAM for the
full-scale run.

```bash
pip install -r submission/code/business_entity_resolution/requirements.txt

# run in order (each stage caches its outputs; the embed step is one-time)
python submission/code/business_entity_resolution/src/phase2_preprocess.py
python submission/code/business_entity_resolution/src/phase3_blocking.py       # GPU
python submission/code/business_entity_resolution/src/phase4_train_matcher.py
python submission/code/business_entity_resolution/src/phase5_generate.py
```

Validate the output format before submitting (from `student_resource/`):

```bash
python3 utils/validate_submission.py \
    --matching ../submission/output/matching_results.tsv \
    --candidate ../submission/output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected: `PASS` (exit 0).

> The notebook paths point at a Colab Drive working dir (`WORK/ROOT/DATA`);
> adjust those constants to run locally. The `src/` files are the notebook cells
> extracted verbatim in execution order.

---

## Tech stack

`sentence-transformers` (BGE-M3) · `faiss-gpu` · `lightgbm` · `rapidfuzz` ·
`jellyfish` (Metaphone) · `scikit-learn` (isotonic) · `wordninja` · `pandas` /
`numpy`.
