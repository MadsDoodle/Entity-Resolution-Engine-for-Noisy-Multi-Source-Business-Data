# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** rubixiin
**Team Members:** Madhav Sukla Baidya(Team Leader) , Chirag Chawla, Shivayan Chakravarty , Ankur Gupta 
**Submission Date:** 27/09/2026

---

## 1. Executive Summary

We resolve business entities with a three-stage pipeline: a **multilingual-embedding
blocking** stage (BGE-M3) unioned with token/address inverted indexes to generate
high-recall candidates, a **LightGBM pairwise matcher** over difference-aware string,
embedding, and adversarial-distractor features, and a **decision layer** that applies
one-to-one assignment followed by per-entity expected-F0.5 cluster-size selection. Our
key innovations are (a) using a script-agnostic multilingual embedding as the blocking
workhorse so Latin↔Indic↔French names align without per-language rules, and (b) an
expected-F0.5 decision rule that directly optimizes the precision-heavy scoring metric
rather than a fixed threshold.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data surfaced the patterns that shaped every design decision:

- **Only four fields** (name, address, country) and no shared identifiers, so the task
  is learned similarity over noisy strings, not key joins.
- **Scale rules out brute force:** ~1.7M Source-1 test entities against ~10M pooled
  Source-2/3 records (~17B pairs), making blocking the crux of the problem.
- **Heavy, structured noise:** legal-suffix add/drop (Ltd/Limited, Pvt/Private), word
  reorder, punctuation, typos, accents, and — critically — **8+ Indic scripts** where a
  Latin Source-1 name matches a Devanagari/Bengali Source-2/3 name (no string rule
  bridges this; multilingual embeddings are mandatory).
- **Domain-style names** (`teamair.com`) appear only in Source-2/3 and need
  word-segmentation to recover the underlying brand.
- **Addresses degrade heavily** via truncation and reorder, so address features must be
  order-invariant and degrade gracefully; many true matches have an empty address.
- **F0.5 weights precision 2×**, and singletons are scored — a false merge costs more
  than a miss, and correctly abstaining earns a full 1.0.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (hybrid: embedding + lexical union blocking,
gradient-boosted matcher, expected-F0.5 decision layer).

**Core Innovation:** A script-agnostic multilingual embedding (BGE-M3) as the primary
blocking signal, unioned with IDF-weighted token/address indexes for lexical recall;
paired with a decision rule that maximizes *expected F0.5* per entity instead of a
global threshold, aligning the output directly with the precision-heavy metric.

---

## 3. Candidate Generation (Blocking)

We reduced ~17B possible pairs to a mean of ~100-150 candidates per Source-1 entity.

- **Blocking keys used:**
  - **BGE-M3 embeddings** (BAAI/bge-m3, MIT, 568M params) over `name [+ address]`,
    searched with **GPU-FAISS** (inner-product, fp16) per country, top-K nearest
    neighbours. This is the workhorse and the only path that recovers cross-script and
    semantically-similar-but-string-disjoint matches.
  - **Token inverted index** on brand/category words, IDF-weighted, keyed by
    `(country, token)`, with high-frequency postings capped (MAXP=300) to control
    candidate-set size.
  - **Address inverted index** on locality/street tokens (non-numeric), keyed by
    `(country, token)`.
  - Scores from all paths are fused per candidate and ranked; the top-CAP list per
    entity is frozen as `candidate_pairs.tsv`.

- **Candidate pairs generated:** mean ~100-150 candidates / Source-1 entity
  (reduction ratio ~1e-5 against the full pool).

- **How we ensured true matches were not lost:** we measured recall against a held-out
  ground-truth validation split. **Embedding-only recall was 94.96%; the union with
  token/address indexes raised the blocking recall ceiling to 96.67%.** A miss analysis
  bucketed the residual loss: ~25.8% of all true matches are exact-normalized-name
  matches (of which we retrieve 98.1%), and the remaining loss is dominated by
  cross-script transliteration, generic-token-only names with empty addresses, and a
  small floor of genuinely unrecoverable "invented" names. Country is handled as an
  **open set of string labels** — nothing is hard-coded to {US, India}, so France
  (test-only) flows through the identical pipeline.

---

## 4. Matching Model

**Features used:**

- **Name features:** Levenshtein ratio, token-sort ratio, token-set ratio, partial
  ratio, no-space partial ratio, token Jaccard, length ratio, script-difference flag.
- **Address features:** Levenshtein / token-sort / token-set / partial ratios, token
  Jaccard, digit-set Jaccard, length ratio, address-missing flag, and **house-number
  features** (equal / close-jitter / absolute difference) to catch municipal numbering
  variance.
- **Embedding & retrieval:** embedding cosine similarity, found-by-embed / found-by-token
  / found-by-address provenance flags.
- **Rank features:** rank position, candidate count, and rank margin (separation from the
  next-best candidate).
- **Adversarial distractor features:** per-token "distractor lift" statistics that detect
  planted distractor tokens the challenge inserts (max/sum lift, counts of distractor vs.
  benign tokens added/removed, an "added-only-distractor" flag). Verified as genuine
  signal on a disjoint-lexicon test, not leakage.
- **Category/entity-type agreement** and conflict flags.

**Model type:** LightGBM gradient-boosted classifier with **isotonic probability
calibration**. (MIT-licensed, far below the 8B parameter limit.)

**Threshold selection method:** F0.5 optimization on a held-out, entity-grouped
validation split. Rather than a single global threshold, the decision layer performs
**one-to-one assignment** (each Source-2/3 record is assigned to its single
highest-probability Source-1 owner) and then selects each entity's cluster size *k* by
**maximizing expected F0.5** over the calibrated match probabilities via Monte-Carlo,
which naturally allows abstaining (predicting a singleton) when confidence is low.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** **0.9473** on the disjoint-lexicon validation test
  (leak-verified); oracle F0.5 (perfect matcher on our candidates) is 0.988. Public
  leaderboard: **~0.94**.

- **Common false positives (wrong merges):** near-duplicate businesses that share an
  address or a generic name (e.g. "Ss Food" vs "Ss Clothes"); planted distractor
  variants that add adversarial tokens to a real name. The distractor-lift features and
  the precision-heavy decision rule were added specifically to suppress these.

- **Common false negatives (missed matches):** (1) cross-script transliteration pairs
  ranked just outside the embedding top-K; (2) records whose name is entirely
  high-frequency generic tokens (`... private limited`) with an empty address — all
  retrieval paths weaken simultaneously; (3) a small residual floor of matches whose
  Source-2/3 name is effectively invented and shares no recoverable signal with the
  Source-1 name.

---

## 6. Conclusion

A multilingual-embedding blocking stage set the recall ceiling (96.67%), a calibrated
LightGBM matcher with difference-aware and adversarial-distractor features converted
candidates to matches at 0.9473 validation F0.5, and an expected-F0.5 decision layer
aligned the output with the precision-heavy metric. The main lesson: on this data,
**blocking recall is the binding constraint**, and a script-agnostic embedding plus a
lexical union is what lifts it without language-specific rules or external data.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline ships under `code/business_entity_resolution/` with all
source in `src/`, a `README.md` (exact run order), and a pinned `requirements.txt`. Entry
points, run in order:

1. `src/phase2_preprocess.py` — record normalization / `process_record`.
2. `src/phase3_blocking.py` — BGE-M3 embedding + GPU-FAISS + token/address union →
   `candidate_pairs.tsv` and cached ranked candidates.
3. `src/phase4_train_matcher.py` — feature build + LightGBM training + isotonic
   calibration → `final_model.pkl`, `iso.pkl`, `meta.pkl`.
4. `src/phase5_generate.py` — scoring + one-to-one + expected-F0.5 decision →
   `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Expensive stages cache to a working directory, so re-runs skip completed work (the
full-pool embedding is a one-time cost).

### B. Additional Results

- Blocking recall: embedding-only **94.96%** → embedding + token/address union
  **96.67%** (validation).
- Exact-normalized-name matches: **25.8%** of all true matches; pipeline retrieves
  **98.1%** of them.
- Oracle F0.5 (perfect matcher on our candidate set): **0.988**.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
