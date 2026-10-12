"""
phase4_train_matcher.py

Extracted verbatim from the working Colab notebook that produced the
submitted matching_results.tsv (macro-F0.5 ~0.94). Cells are concatenated in
execution order. Paths point at the Colab Drive working dir (WORK/ROOT/DATA);
adjust those constants to your environment. See ../README.md for run order.

Source notebook cells: [88, 91, 96]
"""

# ================================================================================
# --- notebook cell 88 ---
# ================================================================================
# =====================================================================================
# PHASE 4 — FEATURE ENGINEERING + LIGHTGBM MATCHER (A100, time-constrained build)
#   Reuses S1, POOL, ranked_all (capped candidates), embed_hits, tfidf_hits, true_map,
#   idf/aidf, ctoks, has_nonlatin from Phase 2/3. Vectorized via rapidfuzz.process.cdist
#   per-entity (C-level batch, not python-loop-per-pair) — this is the speed-critical part.
# =====================================================================================
import time, math, numpy as np, pandas as pd
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from rapidfuzz import fuzz, process
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

t0 = time.time()
CHOSEN_CAP = chosen if "chosen" in dir() else 400   # reuse your Phase 3 CAP if it exists

# Initialize tfidf_hits if not defined by previous phase
if 'tfidf_hits' not in locals() and 'tfidf_hits' not in globals():
    tfidf_hits = defaultdict(list)

# -------------------------------------------------------------------------------------
# 0) helpers
# -------------------------------------------------------------------------------------
def toks(s): return set(x for x in (s or "").split() if len(x) > 1)

def digit_set(s):
    return set(re.findall(r"\d+", s or "")) if s else set()

import re
def jacc(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def nospace(s): return (s or "").replace(" ", "")

FEATURE_COLS = [
    "name_lev", "name_tsort", "name_tset", "name_partial", "name_jacc",
    "name_nospace_partial",
    "addr_lev", "addr_tsort", "addr_tset", "addr_partial", "addr_jacc",
    "digit_jacc", "addr_missing",
    "emb_sim", "tfidf_sim", "found_by_embed", "found_by_tfidf", "found_by_token", "found_by_addr",
    "name_len_ratio", "addr_len_ratio",
    "country_match", "script_differs",
    "category_conflict", "entity_type_agree",
    "rank_pos", "rank_margin", "n_candidates",
]

# -------------------------------------------------------------------------------------
# 1) build the flat candidate-pair table from ranked_all (already capped by blocking)
# -------------------------------------------------------------------------------------
print("building flat candidate table ...")
pairs_s1, pairs_cand = [], []
for e, cids in ranked_all.items():
    cc = cids[:CHOSEN_CAP]
    pairs_s1.extend([e] * len(cc))
    pairs_cand.extend(cc)
print(f"  total candidate pairs: {len(pairs_s1):,}  ({time.time()-t0:.0f}s)")

# -------------------------------------------------------------------------------------
# 2) per-entity feature extraction using rapidfuzz.process.cdist (C-level batch —
#    this is what makes this fast: ONE call per S1 scores all its ~400 candidates,
#    instead of 400 separate python-level fuzz.ratio() calls per entity).
# -------------------------------------------------------------------------------------
def features_for_entity(e, cand_ids):
    r = S1[e]
    n1, a1 = r["canon_name"], r["addr_core"] if r["has_addr"] else ""
    c1 = r["country"]
    cat1 = set(r.get("category_words", []) or [])
    ent1 = set(r.get("entity_types", []) or [])
    nl1 = has_nonlatin(n1)

    cand_names = [POOL[cid]["canon_name"] for cid in cand_ids]
    cand_addrs = [POOL[cid]["addr_core"] if POOL[cid]["has_addr"] else "" for cid in cand_ids]

    # C-level batched scoring: 1 query vs N candidates, per metric
    name_lev     = process.cdist([n1], cand_names, scorer=fuzz.ratio)[0] / 100.0
    name_tsort   = process.cdist([n1], cand_names, scorer=fuzz.token_sort_ratio)[0] / 100.0
    name_tset    = process.cdist([n1], cand_names, scorer=fuzz.token_set_ratio)[0] / 100.0
    name_partial = process.cdist([n1], cand_names, scorer=fuzz.partial_ratio)[0] / 100.0
    ns1 = nospace(n1)
    name_nospace = process.cdist([ns1], [nospace(x) for x in cand_names], scorer=fuzz.partial_ratio)[0] / 100.0

    addr_lev     = process.cdist([a1], cand_addrs, scorer=fuzz.ratio)[0] / 100.0
    addr_tsort   = process.cdist([a1], cand_addrs, scorer=fuzz.token_sort_ratio)[0] / 100.0
    addr_tset    = process.cdist([a1], cand_addrs, scorer=fuzz.token_set_ratio)[0] / 100.0
    addr_partial = process.cdist([a1], cand_addrs, scorer=fuzz.partial_ratio)[0] / 100.0

    emb_dict   = dict(embed_hits.get(e, []))
    tfidf_dict = dict(tfidf_hits.get(e, []))
    d1 = digit_set(a1)
    t1 = toks(n1)

    rows = []
    n = len(cand_ids)
    for i, cid in enumerate(cand_ids):
        cr = POOL[cid]
        a2 = cand_addrs[i]
        rows.append({
            "s1_id": e, "cand_id": cid,
            "name_lev": name_lev[i], "name_tsort": name_tsort[i], "name_tset": name_tset[i],
            "name_partial": name_partial[i], "name_jacc": jacc(t1, toks(cand_names[i])),
            "name_nospace_partial": name_nospace[i],
            "addr_lev": addr_lev[i], "addr_tsort": addr_tsort[i], "addr_tset": addr_tset[i],
            "addr_partial": addr_partial[i], "addr_jacc": jacc(toks(a1), toks(a2)),
            "digit_jacc": jacc(d1, digit_set(a2)),
            "addr_missing": 1.0 if (not r["has_addr"] or not cr["has_addr"]) else 0.0,
            "emb_sim": emb_dict.get(cid, 0.0),
            "tfidf_sim": tfidf_dict.get(cid, 0.0),
            "found_by_embed": 1.0 if cid in emb_dict else 0.0,
            "found_by_tfidf": 1.0 if cid in tfidf_dict else 0.0,
            "found_by_token": 1.0 if t1 & toks(cand_names[i]) else 0.0,
            "found_by_addr": 1.0 if (a1 and a2 and (toks(a1) & toks(a2))) else 0.0,
            "name_len_ratio": min(len(n1), len(cand_names[i])) / max(len(n1), len(cand_names[i]), 1),
            "addr_len_ratio": min(len(a1), len(a2)) / max(len(a1), len(a2), 1),
            "country_match": 1.0 if c1 == cr["country"] else 0.0,
            "script_differs": 1.0 if nl1 != has_nonlatin(cand_names[i]) else 0.0,
            "category_conflict": 1.0 if (cat1 and set(cr.get("category_words", []) or []) and
                                          not (cat1 & set(cr.get("category_words", []) or []))) else 0.0,
            "entity_type_agree": 1.0 if (ent1 & set(cr.get("entity_types", []) or [])) else 0.0,
            "rank_pos": i / max(n - 1, 1),
            "n_candidates": float(n),
        })
    # margin: this candidate's combined score vs next-best (context feature from playbook)
    combo = 3.0 * name_tset + 2.0 * addr_tset
    order = np.argsort(-combo)
    margin = np.zeros(n)
    ranked_combo = combo[order]
    for pos, idx in enumerate(order):
        margin[idx] = ranked_combo[pos] - (ranked_combo[pos+1] if pos+1 < n else 0.0)
    for i, row in enumerate(rows):
        row["rank_margin"] = float(margin[i])
    return rows

print("extracting features per S1 entity (rapidfuzz.process.cdist, batched) ...")
all_rows = []
s1_to_cands = defaultdict(list)
for e, cid in zip(pairs_s1, pairs_cand):
    s1_to_cands[e].append(cid)

for e, cids in tqdm(s1_to_cands.items(), desc="feature extraction"):
    if not cids:
        continue
    all_rows.extend(features_for_entity(e, cids))

feat_df = pd.DataFrame(all_rows)
print(f"  feature rows: {len(feat_df):,}  ({time.time()-t0:.0f}s)")

# -------------------------------------------------------------------------------------
# 3) labels — force in any true positive blocking missed, cap negatives per S1
# -------------------------------------------------------------------------------------
print("assembling labeled training set ...")
feat_df["pair_key"] = list(zip(feat_df["s1_id"], feat_df["cand_id"]))
pos_keys = set()
for e, mids in true_map.items():
    for m in mids:
        if m in POOL:
            pos_keys.add((e, m))

feat_df["label"] = feat_df["pair_key"].isin(pos_keys).astype(int)
found_pos = set(feat_df.loc[feat_df.label == 1, "pair_key"])
missing_pos = pos_keys - found_pos
print(f"  blocking recall on positives (this set): {len(found_pos)/max(len(pos_keys),1):.4f} "
      f"({len(found_pos)}/{len(pos_keys)}) — adding {len(missing_pos)} missed positives directly")

if missing_pos:
    missing_rows = []
    for e, cid in missing_pos:
        missing_rows.extend(features_for_entity(e, [cid]))
    missing_df = pd.DataFrame(missing_rows)
    missing_df["label"] = 1
    feat_df = pd.concat([feat_df.drop(columns=["pair_key"]), missing_df], ignore_index=True)
else:
    feat_df = feat_df.drop(columns=["pair_key"])

MAX_NEG_PER_S1 = 15
pos_rows = feat_df[feat_df.label == 1]
neg_rows = feat_df[feat_df.label == 0]
neg_rows = (neg_rows.groupby("s1_id", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), MAX_NEG_PER_S1), random_state=42)))
full_train = pd.concat([pos_rows, neg_rows], ignore_index=True).drop_duplicates(subset=["s1_id","cand_id"])
print(f"  final labeled rows: {len(full_train):,}  (positives: {(full_train.label==1).sum():,})  "
      f"({time.time()-t0:.0f}s)")

# -------------------------------------------------------------------------------------
# 4) grouped train/val split + LightGBM (GPU-accelerated if built with GPU support;
#    CPU is fine too at this row count — GBDT training is not your bottleneck here)
# -------------------------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, val_idx = next(gss.split(full_train, groups=full_train["s1_id"]))
tr_df, val_df = full_train.iloc[tr_idx].reset_index(drop=True), full_train.iloc[val_idx].reset_index(drop=True)

LGBM_PARAMS = dict(
    objective="binary", metric=["auc", "binary_logloss"], boosting_type="gbdt",
    num_leaves=63, learning_rate=0.05, feature_fraction=0.85, bagging_fraction=0.85,
    bagging_freq=1, min_child_samples=50, n_estimators=2000, random_state=42, n_jobs=-1,
    device_type="cpu",   # GBDT on this feature count trains fast on CPU; switch to "gpu" if lgb built with GPU support
)

print("training LightGBM ...")
model = lgb.LGBMClassifier(**LGBM_PARAMS)
model.fit(
    tr_df[FEATURE_COLS], tr_df["label"],
    eval_set=[(val_df[FEATURE_COLS], val_df["label"])], eval_metric="auc",
    callbacks=[lgb.early_stopping(60, verbose=True), lgb.log_evaluation(100)],
)
print(f"  training done ({time.time()-t0:.0f}s)")

print("\nfeature importance:")
imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(imp)

# -------------------------------------------------------------------------------------
# 5) calibrate threshold directly against macro F0.5 (the actual leaderboard metric)
# -------------------------------------------------------------------------------------
def f_beta_entity(true_set, pred_set, beta=0.5):
    if not true_set: return 1.0 if not pred_set else 0.0
    if not pred_set: return 0.0
    tp = len(true_set & pred_set)
    if tp == 0: return 0.0
    p, r = tp/len(pred_set), tp/len(true_set)
    b2 = beta**2
    return (1+b2)*p*r/(b2*p+r)

def macro_f_beta(val_df, true_map, threshold, score_col="score"):
    preds = defaultdict(set)
    for e, cid, s in zip(val_df.s1_id, val_df.cand_id, val_df[score_col]):
        if s >= threshold: preds[e].add(cid)
    all_e = set(val_df.s1_id) | set(true_map.keys())
    return float(np.mean([f_beta_entity(set(true_map.get(e, [])), preds.get(e, set())) for e in all_e]))

val_df = val_df.copy()
val_df["score"] = model.predict_proba(val_df[FEATURE_COLS])[:, 1]
grid = np.linspace(0.05, 0.95, 37)
best_t, best_f = 0.5, -1.0
for t in grid:
    f = macro_f_beta(val_df, true_map, t)
    if f > best_f: best_t, best_f = t, f
print(f"\nbest threshold={best_t:.3f}  macro F0.5={best_f:.4f}  (total {time.time()-t0:.0f}s)")

import joblib
joblib.dump(model, f"{WORK}/lgbm_matcher.pkl")
import json
json.dump({"threshold": best_t, "val_macro_f05": best_f, "feature_cols": FEATURE_COLS},
          open(f"{WORK}/run_config.json", "w"), indent=2)
print("saved model + config")
print("PHASE 4 COMPLETE.")


# ================================================================================
# --- notebook cell 91 ---
# ================================================================================
# =====================================================================================
# PHASE 4c — DISTRACTOR-LIFT FEATURES + RETRAIN + DECISION LAYER (push toward 0.90)
#   Adds the generator-fingerprint features the model is missing. Reuses feat_df, S1,
#   POOL, true_map, ranked_all, val_entities, FEATURE_COLS, math/re/np/pd/lgb.
# =====================================================================================
import numpy as np, pandas as pd, re, math, time
from collections import defaultdict, Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
t0=time.time()

# -------------------------------------------------------------------------------------
# 1) MINE the distractor lexicon: token lift in UNMATCHED vs MATCHED pool names
#    (the generator appends these to distractors; true matches add different words)
# -------------------------------------------------------------------------------------
matched_ids=set()
for e,mids in true_map.items(): matched_ids.update(m for m in mids if m in POOL)
mtok=Counter(); utok=Counter(); nm=nu=0
for pid,r in POOL.items():
    ts=set(t for t in r["brand_base"].split() if len(t)>1)
    if pid in matched_ids: nm+=1; mtok.update(ts)
    else: nu+=1; utok.update(ts)
def lift(t):
    pu=utok.get(t,0)/max(nu,1); pm=mtok.get(t,0)/max(nm,1)
    return pu/pm if pm>0 else (50.0 if pu>0 else 1.0)
# continuous lift weight for tokens seen enough times
DISTRACTOR_LIFT={t:min(lift(t),50.0) for t in utok if utok[t]>=30 and lift(t)>=2.0}
# benign-addition tokens (added in TRUE matches -> low/negative signal)
BENIGN={"ltd","limited","inc","llc","corp","center","services","partners","company",
        "incorporated","private","pvt","llp","the","and","co","group","enterprises"}
print(f"distractor lexicon: {len(DISTRACTOR_LIFT)} tokens | top:",
      sorted(DISTRACTOR_LIFT.items(),key=lambda x:-x[1])[:12])

# -------------------------------------------------------------------------------------
# 2) NEW difference-aware features (the generator fingerprint) — compute per pair
# -------------------------------------------------------------------------------------
def toks(s): return set(x for x in (s or "").split() if len(x)>1)
def house_no(a):
    m=re.match(r"\s*(\d+)", a or ""); return int(m.group(1)) if m else None

def new_feats(s1_id, cand_id):
    s1=S1[s1_id]; m=POOL[cand_id]
    t1=toks(s1["brand_base"]); t2=toks(m["brand_base"])
    added = t2 - t1            # tokens candidate ADDED vs S1 (distractor signature lives here)
    removed = t1 - t2
    # distractor-lift signal of added tokens (HIGH = likely distractor)
    dl = [DISTRACTOR_LIFT.get(t,1.0) for t in added]
    max_distr = max(dl) if dl else 1.0
    sum_distr = sum(x for x in dl if x>2.0)
    n_distr_added = sum(1 for t in added if t in DISTRACTOR_LIFT)
    n_benign_added = sum(1 for t in added if t in BENIGN)
    # house-number drift (generator jitters it by a small amount)
    h1,h2 = house_no(s1["addr_core"]), house_no(m["addr_core"])
    if h1 is not None and h2 is not None:
        hn_equal = 1.0 if h1==h2 else 0.0
        hn_close = 1.0 if abs(h1-h2)<=3 and h1!=h2 else 0.0   # jittered but close = distractor tell
        hn_diff  = abs(h1-h2)
    else:
        hn_equal=hn_close=0.0; hn_diff=-1.0
    return {
        "max_distractor_lift": max_distr,
        "sum_distractor_lift": min(sum_distr,100.0),
        "n_distractor_added": float(n_distr_added),
        "n_benign_added": float(n_benign_added),
        "n_tokens_added": float(len(added)),
        "n_tokens_removed": float(len(removed)),
        "added_only_distractor": 1.0 if (added and all(t in DISTRACTOR_LIFT for t in added)) else 0.0,
        "hn_equal": hn_equal, "hn_close_jitter": hn_close, "hn_diff": float(hn_diff),
    }

NEW_COLS=["max_distractor_lift","sum_distractor_lift","n_distractor_added","n_benign_added",
          "n_tokens_added","n_tokens_removed","added_only_distractor",
          "hn_equal","hn_close_jitter","hn_diff"]

print("computing new features for all candidate rows ...")
from tqdm.auto import tqdm
newvals=[new_feats(e,c) for e,c in tqdm(zip(feat_df.s1_id,feat_df.cand_id), total=len(feat_df))]
newdf=pd.DataFrame(newvals)
feat_df2=pd.concat([feat_df.reset_index(drop=True), newdf], axis=1)
print(f"  new feats done ({time.time()-t0:.0f}s)")

# -------------------------------------------------------------------------------------
# 3) rebuild labeled training set (drop dead features, add new ones)
# -------------------------------------------------------------------------------------
DEAD={"tfidf_sim","found_by_tfidf","country_match"}
FEATS2=[c for c in FEATURE_COLS if c not in DEAD]+NEW_COLS
feat_df2["label"]=[1 if c in truth_val_all.get(e,set()) else 0
                   for e,c in zip(feat_df2.s1_id,feat_df2.cand_id)] if "truth_val_all" in dir() else \
                  [1 if (e,c) in pos_keys else 0 for e,c in zip(feat_df2.s1_id,feat_df2.cand_id)]
# (pos_keys from Phase 4 defines true pairs)

pos=feat_df2[feat_df2.label==1]
neg=feat_df2[feat_df2.label==0].sample(frac=1,random_state=42).groupby("s1_id",group_keys=False).head(15)
train=pd.concat([pos,neg],ignore_index=True).drop_duplicates(["s1_id","cand_id"])
print(f"  train rows {len(train):,} pos {(train.label==1).sum():,}")

gss=GroupShuffleSplit(1,test_size=0.15,random_state=42)
tr,va=next(gss.split(train,groups=train.s1_id))
trd,vad=train.iloc[tr],train.iloc[va]
val_entities2=set(vad.s1_id.unique())

# -------------------------------------------------------------------------------------
# 4) retrain LightGBM (precision-focused: heavier leaves, scale_pos handled by F0.5 later)
# -------------------------------------------------------------------------------------
mdl=lgb.LGBMClassifier(objective="binary",metric="auc",num_leaves=127,learning_rate=0.04,
    feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,min_child_samples=40,
    n_estimators=3000,random_state=42,n_jobs=-1)
mdl.fit(trd[FEATS2],trd.label,eval_set=[(vad[FEATS2],vad.label)],eval_metric="auc",
        callbacks=[lgb.early_stopping(80),lgb.log_evaluation(200)])
print("\ntop features:")
print(pd.Series(mdl.feature_importances_,index=FEATS2).sort_values(ascending=False).head(15))

# -------------------------------------------------------------------------------------
# 5) score full candidate sets for held-out entities + calibrate + expected-F0.5 select
# -------------------------------------------------------------------------------------
ev=feat_df2[feat_df2.s1_id.isin(val_entities2)].copy()
ev["score"]=mdl.predict_proba(ev[FEATS2])[:,1]
ev["lab"]=[1 if c in set(true_map.get(e,[])) else 0 for e,c in zip(ev.s1_id,ev.cand_id)]
iso=IsotonicRegression(out_of_bounds="clip"); iso.fit(ev.score,ev.lab); ev["p"]=iso.transform(ev.score)
truth={e:set(m for m in true_map.get(e,[]) if m in POOL) for e in val_entities2}

def fb(t,p,beta=0.5):
    if not t: return 1.0 if not p else 0.0
    if not p: return 0.0
    tp=len(t&p)
    if not tp: return 0.0
    pr,rc=tp/len(p),tp/len(t); b2=beta**2
    return (1+b2)*pr*rc/(b2*pr+rc)
def best_k(ps,beta=0.5,R=300):
    ps=np.array(ps)
    if not len(ps): return 0
    b2=beta**2; ef0=np.prod(1-ps); bk,bef=0,ef0
    dr=(np.random.rand(R,len(ps))<ps)
    tt=dr.sum(1)
    for k in range(1,len(ps)+1):
        tp=dr[:,:k].sum(1); pr=tp/k; rc=np.where(tt>0,tp/tt,(tp==0).astype(float))
        d=b2*pr+rc; f=np.where(d>0,(1+b2)*pr*rc/d,0.0); ef=f.mean()
        if ef>bef: bk,bef=k,ef
    return bk
np.random.seed(42)
bye=defaultdict(list)
for e,c,p in zip(ev.s1_id,ev.cand_id,ev.p): bye[e].append((p,c))
preds=defaultdict(set)
for e in val_entities2:
    cs=sorted(bye[e],reverse=True); k=best_k([p for p,_ in cs])
    for p,c in cs[:k]: preds[e].add(c)
score=np.mean([fb(truth[e],preds.get(e,set())) for e in val_entities2])
print(f"\n*** F0.5 with distractor features + decision layer: {score:.4f} ***")
print(f"(was 0.8558 without distractor features) | total {time.time()-t0:.0f}s")

import joblib,json
joblib.dump(mdl,f"{WORK}/lgbm_matcher_v2.pkl")
json.dump({"features":FEATS2,"val_f05":float(score)},open(f"{WORK}/run_config_v2.json","w"))
print("saved v2 model")


# ================================================================================
# --- notebook cell 96 ---
# ================================================================================
# =====================================================================================
# TRAIN FINAL MODEL — 100% of training candidates + distractor features + calibration.
#   Trains on ALL data (no held-out fold — you already verified 0.947 separately).
#   Dumps to Drive with clear names. Visible timing + progress.
# =====================================================================================
import os, time, math, pickle, numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
W="/content/drive/MyDrive/unzipped/student_resource/work"
t0=time.time(); el=lambda:f"{time.time()-t0:.0f}s"
print("="*70); print("TRAINING FINAL MODEL (full data)"); print("="*70)

# ---- 0) load feat_df2 if not in memory ----
if "feat_df2" not in dir():
    if os.path.exists(f"{W}/feat_df2.parquet"):
        print("loading feat_df2 from cache ..."); feat_df2=pd.read_parquet(f"{W}/feat_df2.parquet")
    else:
        raise RuntimeError("feat_df2 not in memory and no parquet cache — run Phase 4 feature extraction first.")
print(f"feature table: {len(feat_df2):,} rows ({el()})")

# ---- 1) feature columns (drop dead features) ----
FEATURE_COLS=["name_lev","name_tsort","name_tset","name_partial","name_jacc","name_nospace_partial",
    "addr_lev","addr_tsort","addr_tset","addr_partial","addr_jacc","digit_jacc","addr_missing",
    "emb_sim","tfidf_sim","found_by_embed","found_by_tfidf","found_by_token","found_by_addr",
    "name_len_ratio","addr_len_ratio","country_match","script_differs","category_conflict",
    "entity_type_agree","rank_pos","rank_margin","n_candidates"]
NEW_COLS=["max_distractor_lift","sum_distractor_lift","n_distractor_added","n_benign_added",
    "n_tokens_added","n_tokens_removed","added_only_distractor","hn_equal","hn_close_jitter","hn_diff"]
DEAD={"tfidf_sim","found_by_tfidf","country_match"}
FINAL_FEATS=[c for c in FEATURE_COLS if c not in DEAD]+NEW_COLS
# safety: confirm all features exist in feat_df2
missing=[c for c in FINAL_FEATS if c not in feat_df2.columns]
assert not missing, f"feat_df2 missing feature columns: {missing}"
print(f"using {len(FINAL_FEATS)} features")

# ---- 2) labels ----
pk={(e,m) for e,mids in true_map.items() for m in mids if m in POOL}
feat_df2["label"]=[1 if (e,c) in pk else 0 for e,c in zip(feat_df2.s1_id,feat_df2.cand_id)]
print(f"positives in feature table: {int(feat_df2.label.sum()):,} / {len(feat_df2):,}")

# ---- 3) hold out 10% ENTITIES purely for CALIBRATION (not for measuring — just calib) ----
cal_ids=set(np.random.RandomState(7).choice(list(set(feat_df2.s1_id)),
             size=int(0.1*feat_df2.s1_id.nunique()),replace=False))
# training = all NON-calibration entities, positives + capped negatives
pos=feat_df2[feat_df2.label==1]
neg=(feat_df2[feat_df2.label==0].sample(frac=1,random_state=42)
     .groupby("s1_id",group_keys=False).head(15))
train=pd.concat([pos,neg]).drop_duplicates(["s1_id","cand_id"])
train=train[~train.s1_id.isin(cal_ids)]
cal=feat_df2[feat_df2.s1_id.isin(cal_ids)]
print(f"train rows: {len(train):,} (pos {int(train.label.sum()):,}) | calib rows: {len(cal):,} ({el()})")

# ---- 4) TRAIN (with live progress every 200 rounds) ----
print("\ntraining LightGBM on full data ...")
tr_start=time.time()
final_model=lgb.LGBMClassifier(objective="binary",metric="auc",num_leaves=127,learning_rate=0.04,
    feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,min_child_samples=40,
    n_estimators=2806,random_state=42,n_jobs=-1)
final_model.fit(train[FINAL_FEATS], train.label,
    eval_set=[(cal[FINAL_FEATS], cal.label)], eval_metric="auc",
    callbacks=[lgb.log_evaluation(200)])
print(f"training took {time.time()-tr_start:.0f}s ({el()} total)")

# ---- 5) calibrate on held-out calibration entities ----
iso=IsotonicRegression(out_of_bounds="clip")
iso.fit(final_model.predict_proba(cal[FINAL_FEATS])[:,1], cal.label)
print("calibrator fit on held-out entities")

# ---- 6) feature importance (sanity) ----
print("\ntop 15 features:")
print(pd.Series(final_model.feature_importances_,index=FINAL_FEATS).sort_values(ascending=False).head(15))

# ---- 7) DUMP TO DRIVE — clear names, visible confirmation ----
print("\n"+"="*70); print("DUMPING MODEL TO DRIVE"); print("="*70)
joblib.dump(final_model, f"{W}/final_model.pkl")
joblib.dump(iso,         f"{W}/iso.pkl")
joblib.dump({"FINAL_FEATS":FINAL_FEATS,"DISTRACTOR_LIFT":DISTRACTOR_LIFT,"BENIGN":BENIGN,
             "FEATURE_COLS":FEATURE_COLS,"NEW_COLS":NEW_COLS}, f"{W}/meta.pkl")
for f in ["final_model.pkl","iso.pkl","meta.pkl"]:
    p=f"{W}/{f}"
    print(f"  ✓ saved {f:18s} ({os.path.getsize(p)/1e6:.1f} MB)  ->  {p}")
print(f"\n✅ FINAL MODEL READY. Total time: {el()}")
print("   Phase 5 (test generation) will load final_model.pkl + iso.pkl + meta.pkl")


