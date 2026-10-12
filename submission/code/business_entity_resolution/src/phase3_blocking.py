"""
phase3_blocking.py

Extracted verbatim from the working Colab notebook that produced the
submitted matching_results.tsv (macro-F0.5 ~0.94). Cells are concatenated in
execution order. Paths point at the Colab Drive working dir (WORK/ROOT/DATA);
adjust those constants to your environment. See ../README.md for run order.

Source notebook cells: [77, 81]
"""

# ================================================================================
# --- notebook cell 77 ---
# ================================================================================
# =====================================================================================
# PHASE 3 — BLOCKING (A100, full-recall). NO CPU TF-IDF. Full-pool GPU embedding is the
#   workhorse (FAISS high-K) + token/phonetic/address hash unions + rarity ranking.
#   Full-pool vectors cached to fp16 memmap -> re-runs skip the embed.
#   Assumes Phase 2 in memory: process_record, has_nonlatin, read_tsv, read_tsv_chunks,
#   FILES, WORK, SEED.
# =====================================================================================
import os, re, time, pickle, math, numpy as np, torch, threading
from collections import defaultdict, Counter
from tqdm.auto import tqdm

assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))
os.system("pip -q install jellyfish wordninja faiss-gpu-cu12 sentence-transformers >/dev/null 2>&1")
import jellyfish, wordninja, faiss
from sentence_transformers import SentenceTransformer

t0=time.time(); elapsed=lambda: f"{time.time()-t0:.0f}s"
class HB:
    def __init__(s,l): s.l=l; s.stop=False
    def __enter__(s):
        s.s=time.time()
        def b():
            while not s.stop:
                time.sleep(20)
                if not s.stop: print(f"    ...{s.l} {time.time()-s.s:.0f}s (total {elapsed()})",flush=True)
        s.t=threading.Thread(target=b,daemon=True); s.t.start(); return s
    def __exit__(s,*a): s.stop=True

# =====================================================================================
# 0) brand_base_v2 (domain/glued segmentation)
# =====================================================================================
TLD={"com","c0m","net","org","in","co","io","biz","info","us","fr","cor","comm","www"}
def _deseg(tok):
    forms={tok}; core=tok
    for tld in sorted(TLD,key=len,reverse=True):
        if core.endswith(tld) and len(core)>len(tld)+3: core=core[:-len(tld)]; break
    if core.isalpha() and len(core)>=10:
        forms.update(p for p in wordninja.split(core) if len(p)>=3)
    elif core!=tok: forms.add(core)
    return forms
def brand_base_v2(cn):
    out=set()
    for t in cn.split():
        if t in TLD: continue
        out|=_deseg(t)
    return " ".join(sorted(out))

# =====================================================================================
# 1) LOAD + PROCESS (cached)
# =====================================================================================
val_ids=set(pd.read_csv(f"{WORK}/split_val_s1_ids.csv")["source1_entity_id"])
gt=read_tsv(FILES["GT"]); gt["ml"]=gt["matched_entity_ids"].apply(lambda s:[x for x in s.split(",") if x])
true_map={r.source1_entity_id:r.ml for _,r in gt[gt.source1_entity_id.isin(val_ids)].iterrows()}
def load_processed(path, keep=None, label=""):
    recs={}; n=sum(1 for _ in open(path,"rb"))-1
    for ch in tqdm(read_tsv_chunks(path, cols=["entity_id","business_name","business_address","country"]),
                   total=(n//1_000_000)+1, desc=f"process {label}"):
        if keep is not None: ch=ch[ch.entity_id.isin(keep)]
        for eid,nm,ad,ct in zip(ch.entity_id,ch.business_name,ch.business_address,ch.country):
            r=process_record(nm,ad); r["country"]=ct.strip()
            r["brand_base"]=brand_base_v2(r["brand_base"] or r["canon_name"]); recs[eid]=r
    return recs
S1_PKL=f"{WORK}/proc_S1val_v2.pkl"; POOL_PKL=f"{WORK}/proc_POOL_v2.pkl"
if os.path.exists(S1_PKL) and os.path.exists(POOL_PKL):
    S1=pickle.load(open(S1_PKL,"rb")); POOL=pickle.load(open(POOL_PKL,"rb")); print("processed(v2) cache")
else:
    S1=load_processed(FILES["S1"], keep=val_ids, label="S1(val)")
    POOL={}
    for k in ["S2","S3"]: POOL.update(load_processed(FILES[k], label=k))
    pickle.dump(S1,open(S1_PKL,"wb")); pickle.dump(POOL,open(POOL_PKL,"wb"))
print(f"[stage] load+process: {elapsed()} | S1={len(S1):,} POOL={len(POOL):,}")

pool_ids=list(POOL); s1_ids=list(S1)
pool_by_c=defaultdict(list); s1_by_c=defaultdict(list)
for e in pool_ids: pool_by_c[POOL[e]["country"]].append(e)
for e in s1_ids:  s1_by_c[S1[e]["country"]].append(e)
STOP={"the","and","of","private","limited","llc","llp","inc","corp","co","company","ltd",
      "pvt","incorporated","corporation","services","group","center","pllc","lp","pc"}
def ctoks(r): return [t for t in r["brand_base"].split() if t not in STOP and len(t)>1]

# =====================================================================================
# 2) IDF + hash indexes (token / phonetic / address)
# =====================================================================================
STAGE=time.time()
N=len(POOL); tok_df=Counter(); adr_df=Counter()
for r in POOL.values():
    for t in set(ctoks(r)): tok_df[t]+=1
    if r["has_addr"]:
        for at in set(r["addr_core"].split()):
            if len(at)>2 and not at.isdigit(): adr_df[at]+=1
def idf(t): return math.log(1+N/(1+tok_df.get(t,0)))
def aidf(t): return math.log(1+N/(1+adr_df.get(t,0)))
tok_index=defaultdict(list); phon_index=defaultdict(list); addr_index=defaultdict(list)
for e in tqdm(pool_ids, desc="build indexes"):
    r=POOL[e]; c=r["country"]
    for t in ctoks(r):
        tok_index[(c,t)].append(e)
        try: phon_index[(c,jellyfish.metaphone(t))].append(e)
        except: pass
    if r["has_addr"]:
        for at in set(r["addr_core"].split()):
            if len(at)>2 and not at.isdigit(): addr_index[(c,at)].append(e)
print(f"[stage] indexes: {time.time()-STAGE:.0f}s (total {elapsed()})")

# =====================================================================================
# 3) FULL-POOL GPU EMBEDDING (workhorse) -> fp16 memmap cache. A100: ~20-25 min once.
# =====================================================================================
STAGE=time.time()
m3=SentenceTransformer("BAAI/bge-m3", device="cuda").half()
DIM=m3.get_sentence_embedding_dimension()
def etext(r): return r["canon_name"] if not r["has_addr"] else f"{r['canon_name']} {r['addr_core']}"
def bgem3(txts,bs=1024): return m3.encode(txts,batch_size=bs,normalize_embeddings=True,convert_to_numpy=True).astype("float32")
VEC=f"{WORK}/pool_vecs_fp16_v2.dat"; IDC=f"{WORK}/pool_ids_v2.npy"
if os.path.exists(VEC) and os.path.exists(IDC) and list(np.load(IDC,allow_pickle=True))==pool_ids:
    pool_vecs=np.memmap(VEC,dtype="float16",mode="r",shape=(len(pool_ids),DIM)); print("pool vecs cache")
else:
    pool_vecs=np.memmap(VEC,dtype="float16",mode="w+",shape=(len(pool_ids),DIM))
    texts=[etext(POOL[e]) for e in pool_ids]; bs=1024
    for i in tqdm(range(0,len(texts),bs), desc="embed FULL POOL (A100)"):
        pool_vecs[i:i+bs]=bgem3(texts[i:i+bs],bs=bs).astype("float16")
    pool_vecs.flush(); np.save(IDC,np.array(pool_ids,dtype=object))
id2row={e:i for i,e in enumerate(pool_ids)}
print(f"[stage] full-pool embed: {time.time()-STAGE:.0f}s (total {elapsed()})")

# =====================================================================================
# 4) GPU FAISS per country (fp16 index in GPU RAM) + high-K search for ALL S1
# =====================================================================================
STAGE=time.time()
res=faiss.StandardGpuResources()
cfg=faiss.GpuIndexFlatConfig(); cfg.useFloat16=True; cfg.device=0
faiss_idx={}
for c,ids in pool_by_c.items():
    idx=faiss.GpuIndexFlatIP(res,DIM,cfg)
    B=500_000
    for s in range(0,len(ids),B):
        idx.add(np.asarray(pool_vecs[[id2row[e] for e in ids[s:s+B]]],dtype="float32"))
    faiss_idx[c]=(idx,ids)
print(f"  FAISS GPU indexes built (total {elapsed()})")
ANN_K=50
embed_hits=defaultdict(list)
for c,ents in s1_by_c.items():
    if c not in faiss_idx: continue
    idx,ids=faiss_idx[c]
    for s in tqdm(range(0,len(ents),20000), desc=f"ANN {c}"):
        chunk=ents[s:s+20000]
        Q=bgem3([etext(S1[e]) for e in chunk])
        D,I=idx.search(Q, min(ANN_K,len(ids)))
        for row,e in enumerate(chunk):
            embed_hits[e]=[(ids[j],float(D[row][k])) for k,j in enumerate(I[row]) if j>=0]
print(f"[stage] FAISS+ANN: {time.time()-STAGE:.0f}s (total {elapsed()})")

# =====================================================================================
# 5) UNION + RARITY RANKING
# =====================================================================================
STAGE=time.time()
def candidates(e):
    r=S1[e]; c=r["country"]; prov=defaultdict(set); score=defaultdict(float)
    for t in ctoks(r):
        w=idf(t)
        for cid in tok_index.get((c,t),[]): prov[cid].add("token"); score[cid]+=w
        try: mp=jellyfish.metaphone(t)
        except: mp=None
        if mp:
            for cid in phon_index.get((c,mp),[]): prov[cid].add("phon"); score[cid]+=0.5*w
    if r["has_addr"]:
        for at in set(r["addr_core"].split()):
            if len(at)>2 and not at.isdigit():
                for cid in addr_index.get((c,at),[]): prov[cid].add("addr"); score[cid]+=0.5*aidf(at)
    for cid,cos in embed_hits.get(e,[]): prov[cid].add("embed"); score[cid]+=3.0*cos
    return prov,score
CAPS=[50,100,200,400,10**9]
recW={k:[0,0] for k in CAPS}; recNoE={k:[0,0] for k in CAPS}; ranked_all={}
for e in tqdm(s1_ids, desc="union+rank"):
    prov,score=candidates(e)
    ranked=[cid for cid,_ in sorted(score.items(),key=lambda kv:-kv[1])]
    ranked_all[e]=ranked
    tr=set(true_map.get(e,[]))&set(POOL)
    if tr:
        for cap in CAPS:
            recW[cap][0]+=len(tr&set(ranked[:cap])); recW[cap][1]+=len(tr)
            noE=[cid for cid in ranked if prov[cid]!={"embed"}][:cap]
            recNoE[cap][0]+=len(tr&set(noE)); recNoE[cap][1]+=len(tr)
print(f"[stage] union+rank: {time.time()-STAGE:.0f}s (total {elapsed()})")

# =====================================================================================
# RESULTS + freeze
# =====================================================================================
print("\n"+"="*60); print("PHASE 3 RESULTS (validation) — recall @ CAP"); print("="*60)
print(f"  {'CAP':>8s}{'recall(all)':>14s}{'recall(no-embed)':>18s}")
for cap in CAPS:
    lbl="no cap" if cap>10**8 else str(cap)
    print(f"  {lbl:>8s}{100*recW[cap][0]/recW[cap][1]:>13.2f}%{100*recNoE[cap][0]/recNoE[cap][1]:>17.2f}%")
base=recW[10**9][0]/recW[10**9][1]
chosen=next((c for c in [50,100,200,400] if recW[c][0]/recW[c][1]>=base-0.002),400)
print(f"\n  no-cap recall = {100*base:.2f}%  | chosen CAP = {chosen}")
print(f"  embed marginal = +{100*(recW[10**9][0]-recNoE[10**9][0])/recW[10**9][1]:.2f}%")
capped=[min(len(ranked_all[e]),chosen) for e in s1_ids]
print(f"  candidates/S1: mean={np.mean(capped):.0f} p95={np.percentile(capped,95):.0f}")
print(f"  reduction ratio: {np.mean(capped)/len(POOL):.7f}")
import csv
with open(f"{WORK}/candidate_pairs_val.tsv","w",newline="") as f:
    w=csv.writer(f,delimiter="\t"); w.writerow(["source1_entity_id","candidate_entity_ids"])
    for e in s1_ids: w.writerow([e, ",".join(ranked_all[e][:chosen])])
print(f"\nsaved: {WORK}/candidate_pairs_val.tsv (CAP={chosen}) | total {elapsed()}")
print("PHASE 3 COMPLETE.")


# ================================================================================
# --- notebook cell 81 ---
# ================================================================================
# =====================================================================================
# PHASE 3 (cached) — validation blocking. Saves ranked_all + embed_hits + indexes to Drive.
#   Reloads all from cache on re-run/account-switch. Assumes Phase 2 funcs + m3 in memory.
# =====================================================================================
import os,time,pickle,math,numpy as np,torch,faiss,jellyfish
from collections import defaultdict,Counter
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
W="/content/drive/MyDrive/unzipped/student_resource/work"; DATA="/content/drive/MyDrive/unzipped/student_resource/dataset"
t0=time.time(); el=lambda:f"{time.time()-t0:.0f}s"
if "m3" not in dir(): m3=SentenceTransformer("BAAI/bge-m3",device="cuda").half()
DIM=1024
def etext(r): return r["canon_name"] if not r["has_addr"] else f"{r['canon_name']} {r['addr_core']}"
def bgem3(t,bs=4096): return m3.encode(t,batch_size=bs,normalize_embeddings=True,convert_to_numpy=True).astype("float32")

# --- 1) processed val records (cache) ---
val_ids=set(pd.read_csv(f"{W}/split_val_s1_ids.csv")["source1_entity_id"])
gt=read_tsv(FILES["GT"]); gt["ml"]=gt["matched_entity_ids"].apply(lambda s:[x for x in s.split(",") if x])
true_map={r.source1_entity_id:r.ml for _,r in gt[gt.source1_entity_id.isin(val_ids)].iterrows()}
def load_proc(path,keep,label):
    recs={}; n=sum(1 for _ in open(path,"rb"))-1
    for ch in tqdm(read_tsv_chunks(path,cols=["entity_id","business_name","business_address","country"]),total=(n//1_000_000)+1,desc=f"proc {label}"):
        if keep is not None: ch=ch[ch.entity_id.isin(keep)]
        for eid,nm,ad,ct in zip(ch.entity_id,ch.business_name,ch.business_address,ch.country):
            r=process_record(nm,ad); r["country"]=ct.strip(); r["brand_base"]=brand_base_v2(r["brand_base"] or r["canon_name"]); recs[eid]=r
    return recs
if os.path.exists(f"{W}/proc_S1val_v2.pkl"):
    S1=pickle.load(open(f"{W}/proc_S1val_v2.pkl","rb")); POOL=pickle.load(open(f"{W}/proc_POOL_v2.pkl","rb")); print("[1] proc CACHED")
else:
    S1=load_proc(FILES["S1"],val_ids,"S1val"); POOL={}
    for k in ["S2","S3"]: POOL.update(load_proc(FILES[k],None,k))
    pickle.dump(S1,open(f"{W}/proc_S1val_v2.pkl","wb")); pickle.dump(POOL,open(f"{W}/proc_POOL_v2.pkl","wb"))
pool_ids=list(POOL); s1_ids=list(S1)
pool_by_c=defaultdict(list); s1_by_c=defaultdict(list)
for e in pool_ids: pool_by_c[POOL[e]["country"]].append(e)
for e in s1_ids: s1_by_c[S1[e]["country"]].append(e)
STOP={"the","and","of","private","limited","llc","llp","inc","corp","co","company","ltd","pvt","incorporated","corporation","services","group","center","pllc","lp","pc"}
def ctoks(r): return [x for x in r["brand_base"].split() if x not in STOP and len(x)>1]
print(f"[1] S1={len(S1):,} POOL={len(POOL):,} ({el()})")

# --- 2) pool embeddings (RESUMABLE memmap cache) ---
VEC=f"{W}/pool_vecs_fp16_v2.dat"; IDC=f"{W}/pool_ids_v2.npy"; PRG=f"{W}/pool_embed_prog.txt"
match=os.path.exists(IDC) and list(np.load(IDC,allow_pickle=True))==pool_ids
mode="r+" if (os.path.exists(VEC) and match) else "w+"
pool_vecs=np.memmap(VEC,dtype="float16",mode=mode,shape=(len(pool_ids),DIM))
if mode=="w+": np.save(IDC,np.array(pool_ids,dtype=object))
start=int(open(PRG).read()) if (os.path.exists(PRG) and match) else 0
if start>=len(pool_ids): print("[2] pool vecs CACHED")
else:
    txt=[etext(POOL[e]) for e in pool_ids]
    for i in tqdm(range(start,len(txt),4096),desc="embed pool (resumable)"):
        pool_vecs[i:i+4096]=bgem3(txt[i:i+4096]).astype("float16"); open(PRG,"w").write(str(min(i+4096,len(txt))))
    pool_vecs.flush()
id2row={e:i for i,e in enumerate(pool_ids)}
print(f"[2] pool embed done ({el()})")

# --- 3) indexes + FAISS + ANN -> embed_hits (cache) ---
if os.path.exists(f"{W}/embed_hits_val.pkl") and os.path.exists(f"{W}/idx_val.pkl"):
    embed_hits=pickle.load(open(f"{W}/embed_hits_val.pkl","rb"))
    IDX=pickle.load(open(f"{W}/idx_val.pkl","rb")); tok_index,phon_index,addr_index,tok_df,adr_df=IDX
    print("[3] embed_hits + indexes CACHED")
else:
    N=len(POOL); tok_df=Counter(); adr_df=Counter(); tok_index=defaultdict(list); phon_index=defaultdict(list); addr_index=defaultdict(list)
    for e in tqdm(pool_ids,desc="indexes"):
        r=POOL[e]; c=r["country"]
        for t in ctoks(r):
            tok_df[t]+=1; tok_index[(c,t)].append(e)
            try: phon_index[(c,jellyfish.metaphone(t))].append(e)
            except: pass
        if r["has_addr"]:
            for at in set(r["addr_core"].split()):
                if len(at)>2 and not at.isdigit(): adr_df[at]+=1; addr_index[(c,at)].append(e)
    res=faiss.StandardGpuResources(); cfg=faiss.GpuIndexFlatConfig(); cfg.useFloat16=True; cfg.device=0
    fx={}
    for c,ids in pool_by_c.items():
        idx=faiss.GpuIndexFlatIP(res,DIM,cfg)
        for s in range(0,len(ids),500000): idx.add(np.asarray(pool_vecs[[id2row[e] for e in ids[s:s+500000]]],dtype="float32"))
        fx[c]=(idx,ids)
    embed_hits=defaultdict(list)
    for c,ents in s1_by_c.items():
        idx,ids=fx[c]
        for s in tqdm(range(0,len(ents),20000),desc=f"ANN {c}"):
            ch=ents[s:s+20000]; Q=bgem3([etext(S1[e]) for e in ch]); D,I=idx.search(Q,50)
            for row,e in enumerate(ch): embed_hits[e]=[(ids[j],float(D[row][k])) for k,j in enumerate(I[row]) if j>=0]
    pickle.dump(dict(embed_hits),open(f"{W}/embed_hits_val.pkl","wb"))
    pickle.dump((dict(tok_index),dict(phon_index),dict(addr_index),tok_df,adr_df),open(f"{W}/idx_val.pkl","wb"))
    print(f"[3] embed_hits+indexes done+saved ({el()})")

# --- 4) blocking union -> ranked_all (cache) ---
if os.path.exists(f"{W}/ranked_all_val.pkl"):
    ranked_all=pickle.load(open(f"{W}/ranked_all_val.pkl","rb")); print("[4] ranked_all CACHED")
else:
    N=len(POOL); MAXP=300; CAP=400
    def idf(t): return math.log(1+N/(1+tok_df.get(t,0)))
    tok_ok={k:v for k,v in tok_index.items() if len(v)<=MAXP}; addr_ok={k:v for k,v in addr_index.items() if len(v)<=MAXP}
    ranked_all={}
    for e in tqdm(s1_ids,desc="blocking"):
        r=S1[e]; c=r["country"]; sc=defaultdict(float)
        for cid,cos in embed_hits.get(e,[]): sc[cid]+=3.0*cos
        for t in ctoks(r):
            post=tok_ok.get((c,t))
            if post:
                w=idf(t)
                for cid in post: sc[cid]+=w
        if r["has_addr"]:
            for at in set(r["addr_core"].split()):
                if len(at)>2 and not at.isdigit():
                    post=addr_ok.get((c,at))
                    if post:
                        for cid in post: sc[cid]+=0.3
        ranked_all[e]=sorted(sc,key=sc.get,reverse=True)[:CAP]
    pickle.dump(ranked_all,open(f"{W}/ranked_all_val.pkl","wb"))
    print(f"[4] ranked_all done+saved ({el()})")
print(f"PHASE 3 COMPLETE ({el()})  cached: proc, pool_vecs, embed_hits, indexes, ranked_all")


