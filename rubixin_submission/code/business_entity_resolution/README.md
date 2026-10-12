# Business Entity Resolution — Reproduction Guide

End-to-end pipeline that, for every Source-1 business, finds all matching
Source-2 / Source-3 records. Output is scored with macro-averaged F0.5.

## Approach (one line)

Multilingual embedding blocking (BGE-M3) unioned with token/address inverted
indexes → LightGBM pairwise matcher → one-to-one assignment + per-entity
expected-F0.5 cluster-size selection.

All components are MIT-licensed and well under 8B parameters. No external data,
APIs, or lookups are used at any stage.

## Environment

- Python 3.10+
- GPU with CUDA 12 (an NVIDIA A100 / L4 was used; a GPU is required for the
  full-pool embedding step and GPU-FAISS search).
- ~64 GB+ system RAM for the full test-scale run (10M pool records).

Install:

```
pip install -r requirements.txt
```

## Data layout

Place the challenge data as provided:

```
student_resource/
  dataset/
    train/  train_source1.tsv train_source2.tsv train_source3.tsv train_ground_truth.tsv
    test/   test_source1.tsv  test_source2.tsv  test_source3.tsv
  utils/validate_submission.py
```

## Pipeline stages (run in order)

The code is organised as sequential stages. Each stage caches its outputs to a
working directory (`work/`) so a re-run skips completed work — this is what
makes the expensive embedding step a one-time cost.

1. **Phase 2 — Preprocessing** (`src/phase2_preprocess.py`)
   Normalizes each record: lowercase, unicode/accent fold, punctuation strip,
   legal-suffix canonicalization, address abbreviation expansion, brand-word
   extraction, and domain segmentation for `.com`-style names. Produces a
   `process_record()` function used by every later stage.

2. **Phase 3 — Blocking / candidate generation** (`src/phase3_blocking.py`)
   - Embeds the full Source-2/3 pool with **BGE-M3** (fp16, cached to a memmap).
   - Builds a **GPU-FAISS** index per country and retrieves the top-K nearest
     neighbours for each Source-1 entity.
   - Unions those with **token** and **address** inverted-index hits
     (IDF-weighted, high-frequency postings capped at MAXP=300).
   - Ranks the union and caps per entity. Writes `candidate_pairs.tsv` and the
     cached ranked candidate lists.

3. **Phase 4 — Matcher training** (`src/phase4_train_matcher.py`)
   Builds pairwise features over the candidate set (name/address string
   similarities, embedding similarity, rank features, distractor-lift features,
   house-number features) and trains a **LightGBM** classifier with isotonic
   calibration. Trained on an entity-grouped train split; validated on a
   held-out entity split with the F0.5 metric.

4. **Phase 5 — Scoring + decision + output** (`src/phase5_generate.py`)
   Scores every candidate for the test set, applies **one-to-one assignment**
   (each Source-2/3 record is assigned to its single best Source-1 owner), then
   selects each entity's match set by maximizing **expected F0.5** over the
   calibrated probabilities. Writes `output/matching_results.tsv` and
   `output/candidate_pairs.tsv`.

## Reproduce the output

```
python src/phase2_preprocess.py       # defines/caches preprocessing
python src/phase3_blocking.py         # -> candidate lists  (GPU, ~1-2 h test scale, one-time embed)
python src/phase4_train_matcher.py    # -> final_model.pkl, iso.pkl, meta.pkl
python src/phase5_generate.py         # -> output/matching_results.tsv, output/candidate_pairs.tsv
```

Then validate before submitting (from the `student_resource/` directory):

```
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected: `PASS` (exit 0).

## Notes

- `candidate_pairs.tsv` is written as the exact set the matcher scores, so every
  matched ID is a subset of its candidate list (as the validator requires).
- Country is treated as an open set of string labels — nothing is hard-coded to
  {US, India}; France (test-only) flows through the same pipeline.
- The embedding step dominates runtime and is cached; downstream re-runs are fast.
