"""
phase5_generate.py

Extracted verbatim from the working Colab notebook that produced the
submitted matching_results.tsv (macro-F0.5 ~0.94). Cells are concatenated in
execution order. Paths point at the Colab Drive working dir (WORK/ROOT/DATA);
adjust those constants to your environment. See ../README.md for run order.

Source notebook cells: [101]
"""

# ================================================================================
# --- notebook cell 101 ---
# ================================================================================
# =====================================================================================
# PHASE 5 — FINAL (memory-clean, fully cached, threaded fast scoring). A100.
#   - caches EVERYTHING: test proc, embed, tembed(ANN), test_ranked(blocking)
#   - CAP=150 for scoring (recall-safe, France-margin), ThreadPoolExecutor (GIL-free rapidfuzz)
#   - flushes validation clutter to avoid OOM
#   Assumes: Phase 2 funcs + m3 + FILES in memory. Loads model from cache.
# =====================================================================================
import os,re,math,time,pickle,csv,shutil,gc,numpy as np,pandas as pd,torch,faiss,jellyfish,joblib
from collections import defaultdict,Counter
from tqdm.auto import tqdm
from rapidfuzz import fuzz,process
from concurrent.futures import ThreadPoolExecutor
t0=time.time(); el=lambda:f"{time.time()-t0:.0f}s"; elapsed=el
WORK="/content/drive/MyDrive/unzipped/student_resource/work"; ROOT="/content/drive/MyDrive/unzipped/student_resource"; DATA=f"{ROOT}/dataset"; TEAM="team"

# ---- FLUSH validation clutter (Phase 5 needs none of it) ----
for v in ["feat_df2","feat_df","full_train","train","trainfull","POOL","S1","embed_hits",
          "ranked_all","tok_index","phon_index","addr_index","pool_vecs","tfidf_hits",
          "true_map","results","ev","eval_df","sub","preds","preds2"]:
    if v in dir():
        try: del globals()[v]
        except: pass
gc.collect()
import psutil; print(f"RAM after flush: {psutil.virtual_memory().used/1e9:.0f}/{psutil.virtual_memory().total/1e9:.0f} GB")

# ---- prereq check ----
for _n in ["m3","process_record","brand_base_v2","has_nonlatin","read_tsv_chunks","FILES"]:
    assert _n in dir(), f"MISSING {_n} — reload Phase 2 funcs / m3 first"
print("prereqs OK")

# ---- 0) MODEL from cache ----
final_model=joblib.load(f"{WORK}/final_model.pkl"); iso=joblib.load(f"{WORK}/iso.pkl")
_m=joblib.load(f"{WORK}/meta.pkl"); FINAL_FEATS=_m["FINAL_FEATS"]; DISTRACTOR_LIFT=_m["DISTRACTOR_LIFT"]; BENIGN=_m["BENIGN"]
print(f"[0] model loaded ({len(FINAL_FEATS)} features)")

# ---- 1) TEST PROC (cached) ----
def load_proc(path,label):
    recs={}; n=sum(1 for _ in open(path,"rb"))-1
    for ch in tqdm(read_tsv_chunks(path,cols=["entity_id","business_name","business_address","country"]),total=(n//1_000_000)+1,desc=f"proc {label}"):
        for eid,nm,ad,ct in zip(ch.entity_id,ch.business_name,ch.business_address,ch.country):
            r=process_record(nm,ad); r["country"]=ct.strip(); r["brand_base"]=brand_base_v2(r["brand_base"] or r["canon_name"]); recs[eid]=r
    return recs
if os.path.exists(f"{WORK}/test_S1.pkl"):
    TS1=pickle.load(open(f"{WORK}/test_S1.pkl","rb")); TPOOL=pickle.load(open(f"{WORK}/test_POOL.pkl","rb")); print("[1] test proc CACHED")
else:
    TS1=load_proc(FILES["test_S1"],"S1"); TPOOL={}
    for k in ["test_S2","test_S3"]: TPOOL.update(load_proc(FILES[k],k))
    pickle.dump(TS1,open(f"{WORK}/test_S1.pkl","wb")); pickle.dump(TPOOL,open(f"{WORK}/test_POOL.pkl","wb"))
ts1_ids=list(TS1); tp_ids=list(TPOOL)
tp_by_c=defaultdict(list); ts1_by_c=defaultdict(list)
for e in tp_ids: tp_by_c[TPOOL[e]["country"]].append(e)
for e in ts1_ids: ts1_by_c[TS1[e]["country"]].append(e)
print(f"[1] test_S1={len(TS1):,} POOL={len(TPOOL):,} ({el()})")
STOP={"the","and","of","private","limited","llc","llp","inc","corp","co","company","ltd","pvt","incorporated","corporation","services","group","center","pllc","lp","pc"}
def ctoks(r): return [x for x in r["brand_base"].split() if x not in STOP and len(x)>1]

# ---- 2) POOL EMBED (resumable cache) ----
DIM=1024; BS=4096
def etext(r): return r["canon_name"] if not r["has_addr"] else f"{r['canon_name']} {r['addr_core']}"
def bgem3(t,bs=BS): return m3.encode(t,batch_size=bs,normalize_embeddings=True,convert_to_numpy=True).astype("float32")
TVEC=f"{WORK}/test_pool_vecs.dat"; TIDC=f"{WORK}/test_pool_ids.npy"; TPRG=f"{WORK}/test_embed_prog.txt"
ok=os.path.exists(TIDC) and list(np.load(TIDC,allow_pickle=True))==tp_ids
mode="r+" if (os.path.exists(TVEC) and ok) else "w+"
tp_vecs=np.memmap(TVEC,dtype="float16",mode=mode,shape=(len(tp_ids),DIM))
if mode=="w+": np.save(TIDC,np.array(tp_ids,dtype=object))
start=int(open(TPRG).read()) if (os.path.exists(TPRG) and ok) else 0
if start>=len(tp_ids): print("[2] pool embed CACHED")
else:
    txt=[etext(TPOOL[e]) for e in tp_ids]
    for i in tqdm(range(start,len(txt),BS),desc="embed POOL"):
        tp_vecs[i:i+BS]=bgem3(txt[i:i+BS]).astype("float16"); open(TPRG,"w").write(str(min(i+BS,len(txt))))
    tp_vecs.flush()
tp_row={e:i for i,e in enumerate(tp_ids)}
print(f"[2] pool embed done ({el()})")

# ---- 3+4) ANN (tembed) + BLOCKING (test_ranked) — NOW CACHED ----
if os.path.exists(f"{WORK}/tembed.pkl") and os.path.exists(f"{WORK}/test_ranked.pkl"):
    tembed=pickle.load(open(f"{WORK}/tembed.pkl","rb")); test_ranked=pickle.load(open(f"{WORK}/test_ranked.pkl","rb"))
    print("[3+4] tembed + test_ranked CACHED — skipped index/ANN/blocking")
else:
    Nt=len(TPOOL); tdf=Counter(); ttok=defaultdict(list); tphon=defaultdict(list); tadr=defaultdict(list)
    for e in tqdm(tp_ids,desc="indexes"):
        r=TPOOL[e]; c=r["country"]
        for t in ctoks(r):
            tdf[t]+=1; ttok[(c,t)].append(e)
            try: tphon[(c,jellyfish.metaphone(t))].append(e)
            except: pass
        if r["has_addr"]:
            for at in set(r["addr_core"].split()):
                if len(at)>2 and not at.isdigit(): tadr[(c,at)].append(e)
    def tidf(t): return math.log(1+Nt/(1+tdf.get(t,0)))
    res=faiss.StandardGpuResources(); cfg=faiss.GpuIndexFlatConfig(); cfg.useFloat16=True; cfg.device=0
    tfaiss={}
    for c,ids in tp_by_c.items():
        idx=faiss.GpuIndexFlatIP(res,DIM,cfg)
        for s in range(0,len(ids),500000): idx.add(np.asarray(tp_vecs[[tp_row[e] for e in ids[s:s+500000]]],dtype="float32"))
        tfaiss[c]=(idx,ids)
    tembed=defaultdict(list)
    for c,ents in ts1_by_c.items():
        if c not in tfaiss: continue
        idx,ids=tfaiss[c]
        for s in tqdm(range(0,len(ents),20000),desc=f"ANN {c}"):
            ch=ents[s:s+20000]; Q=bgem3([etext(TS1[e]) for e in ch]); D,I=idx.search(Q,50)
            for row,e in enumerate(ch): tembed[e]=[(ids[j],float(D[row][k])) for k,j in enumerate(I[row]) if j>=0]
    pickle.dump(dict(tembed),open(f"{WORK}/tembed.pkl","wb"))
    print(f"[3] ANN done+cached ({el()})")
    MAXP=300; CAP=400
    ttok_ok={k:v for k,v in ttok.items() if len(v)<=MAXP}; tadr_ok={k:v for k,v in tadr.items() if len(v)<=MAXP}
    test_ranked={}
    for e in tqdm(ts1_ids,desc="blocking"):
        r=TS1[e]; c=r["country"]; sc=defaultdict(float)
        for cid,cos in tembed.get(e,[]): sc[cid]+=3.0*cos
        for t in ctoks(r):
            post=ttok_ok.get((c,t))
            if post:
                w=tidf(t)
                for cid in post: sc[cid]+=w
        if r["has_addr"]:
            for at in set(r["addr_core"].split()):
                if len(at)>2 and not at.isdigit():
                    post=tadr_ok.get((c,at))
                    if post:
                        for cid in post: sc[cid]+=0.3
        test_ranked[e]=sorted(sc,key=sc.get,reverse=True)[:CAP]
    pickle.dump(test_ranked,open(f"{WORK}/test_ranked.pkl","wb"))
    del ttok,tphon,tadr,ttok_ok,tadr_ok,tfaiss; gc.collect()
    print(f"[4] blocking done+cached ({el()})")

# ---- 4b) CAP=150 for scoring (recall-safe, France margin) ----
SCORE_CAP=150
test_ranked={e:cids[:SCORE_CAP] for e,cids in test_ranked.items()}
print(f"[4b] scoring CAP={SCORE_CAP}, mean cand={np.mean([len(v) for v in test_ranked.values()]):.0f}")

# ---- 5) FEATURE ARRAYS (vectorized) ----
def toks(s): return set(x for x in (s or "").split() if len(x)>1)
def digit_set(s): return set(re.findall(r"\d+",s or "")) if s else set()
def nospace(s): return (s or "").replace(" ","")
def house_no(a):
    m=re.match(r"\s*(\d+)",a or ""); return int(m.group(1)) if m else None
def jc(a,b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)
def feats_arrays(e,cids):
    r=TS1[e]; n1=r["canon_name"]; a1=r["addr_core"] if r["has_addr"] else ""
    cat1=set(r.get("category_words",[])or[]); ent1=set(r.get("entity_types",[])or[]); nl1=has_nonlatin(n1)
    cn=[TPOOL[c]["canon_name"] for c in cids]; ca=[TPOOL[c]["addr_core"] if TPOOL[c]["has_addr"] else "" for c in cids]
    NL=lambda s,q,arr: np.asarray(process.cdist([q],arr,scorer=s)[0])/100.0
    nlv=NL(fuzz.ratio,n1,cn); nts=NL(fuzz.token_sort_ratio,n1,cn); ntt=NL(fuzz.token_set_ratio,n1,cn)
    npr=NL(fuzz.partial_ratio,n1,cn); nns=NL(fuzz.partial_ratio,nospace(n1),[nospace(x) for x in cn])
    alv=NL(fuzz.ratio,a1,ca); ats=NL(fuzz.token_sort_ratio,a1,ca); att=NL(fuzz.token_set_ratio,a1,ca); apr=NL(fuzz.partial_ratio,a1,ca)
    emb=dict(tembed.get(e,[])); d1=digit_set(a1); t1=toks(n1); h1=house_no(a1); nc=len(cids)
    t2s=[toks(x) for x in cn]; adds=[t2-t1 for t2 in t2s]
    combo=3.0*ntt+2.0*att; order=np.argsort(-combo); marg=np.zeros(nc); rq=combo[order]
    for p,ix in enumerate(order): marg[ix]=rq[p]-(rq[p+1] if p+1<nc else 0.0)
    hn_e=np.zeros(nc); hn_c=np.zeros(nc); hn_d=np.zeros(nc)
    for i,c in enumerate(cids):
        h2=house_no(TPOOL[c]["addr_core"])
        if h1 is not None and h2 is not None: hn_e[i]=1.0 if h1==h2 else 0; hn_c[i]=1.0 if 0<abs(h1-h2)<=3 else 0; hn_d[i]=abs(h1-h2)
        else: hn_d[i]=-1.0
    return {"s1_id":np.array([e]*nc,dtype=object),"cand_id":np.array(cids,dtype=object),
        "name_lev":nlv,"name_tsort":nts,"name_tset":ntt,"name_partial":npr,
        "name_jacc":np.array([jc(t1,t) for t in t2s]),"name_nospace_partial":nns,
        "addr_lev":alv,"addr_tsort":ats,"addr_tset":att,"addr_partial":apr,
        "addr_jacc":np.array([jc(toks(a1),toks(x)) for x in ca]),"digit_jacc":np.array([jc(d1,digit_set(x)) for x in ca]),
        "addr_missing":np.array([0.0 if(r["has_addr"] and TPOOL[c]["has_addr"])else 1.0 for c in cids]),
        "emb_sim":np.array([emb.get(c,0.0) for c in cids]),"found_by_embed":np.array([1.0 if c in emb else 0 for c in cids]),
        "found_by_token":np.array([1.0 if t1&t else 0 for t in t2s]),
        "found_by_addr":np.array([1.0 if(a1 and x and toks(a1)&toks(x))else 0 for x in ca]),
        "name_len_ratio":np.array([min(len(n1),len(x))/max(len(n1),len(x),1) for x in cn]),
        "addr_len_ratio":np.array([min(len(a1),len(x))/max(len(a1),len(x),1) for x in ca]),
        "script_differs":np.array([1.0 if nl1!=has_nonlatin(x) else 0 for x in cn]),
        "category_conflict":np.array([1.0 if(cat1 and set(TPOOL[c].get("category_words",[])or[]) and not(cat1&set(TPOOL[c].get("category_words",[])or[])))else 0 for c in cids]),
        "entity_type_agree":np.array([1.0 if(ent1&set(TPOOL[c].get("entity_types",[])or[]))else 0 for c in cids]),
        "rank_pos":np.arange(nc)/max(nc-1,1),"n_candidates":np.full(nc,float(nc)),"rank_margin":marg,
        "max_distractor_lift":np.array([max([DISTRACTOR_LIFT.get(t,1.0) for t in a])if a else 1.0 for a in adds]),
        "sum_distractor_lift":np.array([min(sum(x for x in[DISTRACTOR_LIFT.get(t,1.0) for t in a]if x>2),100.0) for a in adds]),
        "n_distractor_added":np.array([float(sum(1 for t in a if t in DISTRACTOR_LIFT)) for a in adds]),
        "n_benign_added":np.array([float(sum(1 for t in a if t in BENIGN)) for a in adds]),
        "n_tokens_added":np.array([float(len(a)) for a in adds]),"n_tokens_removed":np.array([float(len(t1-t)) for t in t2s]),
        "added_only_distractor":np.array([1.0 if(a and all(t in DISTRACTOR_LIFT for t in a))else 0 for a in adds]),
        "hn_equal":hn_e,"hn_close_jitter":hn_c,"hn_diff":hn_d}

_probe=feats_arrays(ts1_ids[0], test_ranked[ts1_ids[0]] or [tp_ids[0]])
assert not set(FINAL_FEATS)-set(_probe.keys()), "feature parity FAIL"
print("[5] feature parity OK")

# ---- 6) THREADED PASS 1 (rapidfuzz releases GIL -> real multicore) ----
import os as _os
N_WORKERS=min(8,_os.cpu_count() or 8)
entity_preds={}; cand_owner={}
def flush_parallel(elist):
    def one(e):
        cc=test_ranked[e]
        return e,(feats_arrays(e,cc) if cc else None)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results=list(ex.map(one,elist))
    cols=defaultdict(list)
    for e,fa in results:
        if fa is None: entity_preds[e]=[]; continue
        for k,v in fa.items(): cols[k].append(v)
    if not cols: return
    df=pd.DataFrame({k:np.concatenate(v) for k,v in cols.items()})
    df["p"]=iso.transform(final_model.predict_proba(df[FINAL_FEATS])[:,1])
    for e,cid,p in zip(df.s1_id,df.cand_id,df.p):
        entity_preds.setdefault(e,[]).append((float(p),cid))
        if cid not in cand_owner or p>cand_owner[cid][0]: cand_owner[cid]=(p,e)
print(f"[6] PASS 1 scoring (threaded x{N_WORKERS}) ...")
BATCH=4000
for i in tqdm(range(0,len(ts1_ids),BATCH),desc="score(threaded)"):
    flush_parallel(ts1_ids[i:i+BATCH])
for e in ts1_ids:
    if e not in entity_preds: entity_preds[e]=[]
print(f"[6] scored ({el()})")

# ---- 6b) PASS 2 one-to-one + expected-F0.5 + write ----
def best_k(ps,R=300,b=0.5):
    ps=np.array(ps)
    if not len(ps): return 0
    b2=b*b; bk,bef=0,np.prod(1-ps); dr=(np.random.rand(R,len(ps))<ps); tt=dr.sum(1)
    for k in range(1,len(ps)+1):
        tp=dr[:,:k].sum(1); pr=tp/k; rc=np.where(tt>0,tp/tt,(tp==0).astype(float))
        d=b2*pr+rc; f=np.where(d>0,(1+b2)*pr*rc/d,0.0)
        if f.mean()>bef: bk,bef=k,f.mean()
    return bk
np.random.seed(42); os.makedirs(f"{WORK}/output",exist_ok=True)
mr=open(f"{WORK}/output/matching_results.tsv","w",newline=""); mw=csv.writer(mr,delimiter="\t"); mw.writerow(["source1_entity_id","matched_entity_ids"])
cp=open(f"{WORK}/output/candidate_pairs.tsv","w",newline=""); cw=csv.writer(cp,delimiter="\t"); cw.writerow(["source1_entity_id","candidate_entity_ids"])
nm2=ne2=0
for e in tqdm(ts1_ids,desc="write"):
    pl=[(p,cid) for p,cid in entity_preds[e] if cand_owner.get(cid,(0,None))[1]==e]
    cs=sorted(pl,reverse=True); k=best_k([p for p,_ in cs]); matched=[cid for _,cid in cs[:k]]
    nm2+= 1 if matched else 0; ne2+= 0 if matched else 1
    mw.writerow([e,",".join(matched)]); cw.writerow([e,",".join(test_ranked[e])])
mr.close(); cp.close()
print(f"[6b] wrote {nm2:,} matched, {ne2:,} singleton ({el()})")

# ---- 7) VALIDATE ----
import subprocess
v=subprocess.run(["python3","utils/validate_submission.py","--matching",f"{WORK}/output/matching_results.tsv",
    "--candidate",f"{WORK}/output/candidate_pairs.tsv","--test-dir",f"{DATA}/test"],cwd=ROOT,capture_output=True,text=True)
print(v.stdout[-1500:]);
if v.stderr: print("STDERR:",v.stderr[-400:])
PASS="PASS" in v.stdout
print("VALIDATOR:","PASS ✅" if PASS else "FAIL ❌")

# ---- 8) ZIP ----
if PASS:
    stage=f"{WORK}/{TEAM}_submission"
    if os.path.exists(stage): shutil.rmtree(stage)
    os.makedirs(f"{stage}/output",exist_ok=True); os.makedirs(f"{stage}/code/business_entity_resolution/src",exist_ok=True)
    shutil.copy(f"{WORK}/output/matching_results.tsv",f"{stage}/output/"); shutil.copy(f"{WORK}/output/candidate_pairs.tsv",f"{stage}/output/")
    joblib.dump(final_model,f"{stage}/code/business_entity_resolution/src/final_model.pkl")
    open(f"{stage}/code/business_entity_resolution/requirements.txt","w").write("pandas\nnumpy\nscikit-learn\nlightgbm\nrapidfuzz\njellyfish\nsentence-transformers\nfaiss-gpu-cu12\nwordninja\n")
    open(f"{stage}/code/business_entity_resolution/README.md","w").write("# Business ER Pipeline\nBGE-M3 (MIT) blocking + LightGBM (MIT) matcher + one-to-one/expected-F0.5 decision layer. <8B params, MIT-licensed, no external data.\n")
    tmpl=f"{ROOT}/Documentation_template.md"
    if os.path.exists(tmpl): shutil.copy(tmpl,f"{stage}/Documentation_template.md")
    shutil.make_archive(f"{WORK}/{TEAM}_submission","zip",stage)
    print(f"✅ ZIP: {WORK}/{TEAM}_submission.zip")
print(f"TOTAL: {el()}")


