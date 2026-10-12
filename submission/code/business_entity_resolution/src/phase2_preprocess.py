"""
phase2_preprocess.py

Extracted verbatim from the working Colab notebook that produced the submitted
matching_results.tsv (macro-F0.5 ~0.94). Phase-2 cells are concatenated in
dependency order: base utilities -> field normalizers (suffix, address, name
split) -> record_profile -> process_record assembly -> brand_base_v2 domain
segmentation. process_record() is the single entry point used identically for
S1/S2/S3 in both train and test. See ../README.md for run order.

Source notebook cells: [43, 44, 48, 57, 58, 59, 60, 71]
"""

# ================================================================================
# --- notebook cell 43 ---
# ================================================================================
# =====================================================================================
# PHASE 2 — COMMON SETUP  (run ONCE; 2.1, 2.2, ... all reuse these imports/paths/helpers)
# =====================================================================================
import os, re, sys, json, time, random, unicodedata
import numpy as np, pandas as pd
from collections import Counter, defaultdict

SEED = 42; random.seed(SEED); np.random.seed(SEED)
pd.set_option("display.max_colwidth", 200)

BASE = "/content/drive/MyDrive/unzipped/student_resource"
DATA = f"{BASE}/dataset"
WORK = f"{BASE}/work"
os.makedirs(WORK, exist_ok=True)

FILES = {
    "S1": f"{DATA}/train/train_source1.tsv",
    "S2": f"{DATA}/train/train_source2.tsv",
    "S3": f"{DATA}/train/train_source3.tsv",
    "GT": f"{DATA}/train/train_ground_truth.tsv",
    "test_S1": f"{DATA}/test/test_source1.tsv",
    "test_S2": f"{DATA}/test/test_source2.tsv",
    "test_S3": f"{DATA}/test/test_source3.tsv",
}

# ---- shared TSV reader (always strings, keep empties as "") ----
def read_tsv(path, cols=None, nrows=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       encoding="utf-8", usecols=cols, nrows=nrows)

def read_tsv_chunks(path, cols=None, chunksize=1_000_000):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       encoding="utf-8", usecols=cols, chunksize=chunksize)

# ---- shared unicode script ranges (India is multi-script; see observations O13) ----
SCRIPT_RANGES = {
    "Devanagari": r"\u0900-\u097F", "Bengali": r"\u0980-\u09FF", "Gurmukhi": r"\u0A00-\u0A7F",
    "Gujarati": r"\u0A80-\u0AFF", "Oriya": r"\u0B00-\u0B7F", "Tamil": r"\u0B80-\u0BFF",
    "Telugu": r"\u0C00-\u0C7F", "Kannada": r"\u0C80-\u0CFF", "Malayalam": r"\u0D00-\u0D7F",
}
RE_INDIC    = re.compile("[" + "".join(SCRIPT_RANGES.values()) + "]")
RE_NONLATIN = re.compile(r"[^\x00-\x7F]")
RE_TOKEN    = re.compile(r"[a-z0-9]+")

# ---- shared quick helpers reused across sub-modules ----
def has_indic(s):    return bool(RE_INDIC.search(s or ""))
def has_nonlatin(s): return bool(RE_NONLATIN.search(s or ""))

def sample_file(path, n, cols=None):
    """Uniform random sample of n data rows (fair, memory-safe for the 5M files)."""
    total = sum(1 for _ in open(path, "rb")) - 1
    if n >= total:
        return read_tsv(path, cols=cols)
    keep = set(random.sample(range(1, total + 1), n))
    skip = [i for i in range(1, total + 1) if i not in keep]
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       encoding="utf-8", usecols=cols, skiprows=skip)

# ---- registry so sub-modules can compose their normalizers in order ----
#     (2.1 registers the base cleaner; later parts append to this)
NORMALIZERS = {}   # name -> function(str) -> str
def register_normalizer(name, fn):
    NORMALIZERS[name] = fn
    print(f"registered normalizer: {name}  (pipeline now: {list(NORMALIZERS)})")

print("PHASE 2 COMMON SETUP READY")
print(f"  BASE exists: {os.path.isdir(BASE)} | WORK: {WORK}")


# ================================================================================
# --- notebook cell 44 ---
# ================================================================================
# =====================================================================================
# PHASE 2.1 — TEXT CLEANING PIPELINE  (assumes COMMON SETUP is in memory)
#   Identical across S1/S2/S3: NFKC -> lowercase -> &/+ -> "and" -> leading-junk strip
#   -> script-punctuation strip -> punctuation-to-space -> whitespace collapse.
#   Latin + multilingual safe. Registers `clean_text`. Verified with real examples,
#   scale-audited on BOTH name and address, and checked for idempotency.
# =====================================================================================

_RE_SCRIPT_PUNCT = re.compile(r"[\u0964\u0965\u2013\u2014\u2018\u2019\u201C\u201D\u2026]")  # danda, en/em dash, smart quotes, ellipsis
_RE_AMP          = re.compile(r"\s*&\s*")
_RE_LEAD_JUNK    = re.compile(r"^[^\w\u00C0-\uD7FF]+")   # leading --, <<, #, +, digits-as-symbols
_RE_TRAIL_JUNK   = re.compile(r"[^\w\u00C0-\uD7FF]+$")   # trailing junk (GAP #2)
_RE_PUNCT        = re.compile(r"[^\w\s\u00C0-\uD7FF]")   # remaining punctuation -> space
_RE_WS           = re.compile(r"\s+")

def clean_text(s):
    """Base cleaner, identical for name and address, all sources, all scripts."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)          # 1) canonical unicode (keeps scripts)
    s = s.lower()                                 # 2) lowercase (O6; no-op on Indic)
    s = _RE_AMP.sub(" and ", s)                   # 3) & -> "and" BEFORE punct strip
    s = s.replace("+", " and ")                   #    + -> "and"
    s = _RE_SCRIPT_PUNCT.sub(" ", s)              # 4) strip danda/smart-quotes (GAP #1)
    s = _RE_LEAD_JUNK.sub("", s)                  # 5) leading junk (--, <<, #, @)
    s = _RE_TRAIL_JUNK.sub("", s)                 # 6) trailing junk (GAP #2)
    s = _RE_PUNCT.sub(" ", s)                     # 7) remaining punctuation -> space
    s = _RE_WS.sub(" ", s).strip()                # 8) collapse whitespace
    return s

register_normalizer("clean_text", clean_text)

# -------------------------------------------------------------------------------------
# VERIFY on real examples (each targets one known noise pattern from Phase 1)
# -------------------------------------------------------------------------------------
checks = [
    ("Davis Family Office",                 "case -> lower (O6)"),
    ("-- Holloway Peak Inc Seafood",        "leading junk strip"),
    ("<< Team Ecole",                       "leading junk strip"),
    ("Unique & Sons Private Limited",       "& -> and"),
    ("B+ Retail Inc",                       "+ -> and"),
    ("FH Business Pvt-Ltd.",                "trailing '.' strip + punct"),
    ("#scholarshipcouncil",                 "leading # strip"),
    ("Grand Connecticut LLC --",            "TRAILING junk strip (GAP #2)"),
    ("Clairvoyant Récord Private Ltd",      "accent PRESERVED (French, O11)"),
    ("सुप्रीम आईटी प्राइवेट लिमिटेड।",         "Devanagari kept + danda stripped (GAP #1)"),
    ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",       "Tamil PRESERVED (O13)"),
    ("ବ୍ଲାକ୍ ଅରିହନ୍ତ",                        "Odia PRESERVED (O13)"),
    ("88 Olive Circle, Lebanon, TN",        "address punct + case"),
    ("उत्तर प्रदेश",                          "Devanagari ADDRESS preserved (O13)"),
]
print("2.1 CLEAN_TEXT — before / after:\n")
for raw, why in checks:
    print(f"  [{why}]")
    print(f"    raw : {raw}")
    print(f"    out : {clean_text(raw)}\n")

# -------------------------------------------------------------------------------------
# GAP #2 — TRAILING-JUNK FREQUENCY CHECK (data-driven, don't assume it's a non-issue)
# -------------------------------------------------------------------------------------
print("2.1 TRAILING-JUNK frequency (raw names/addresses ending in punctuation):")
_trail_raw = re.compile(r"[^\w\s\u00C0-\uD7FF]$")
for k in ["S1","S2","S3"]:
    df = sample_file(FILES[k], 30_000, cols=["business_name","business_address"])
    nm_t = df.business_name.str.rstrip().str.contains(_trail_raw, regex=True).mean()
    ad_t = df.business_address.str.rstrip().str.contains(_trail_raw, regex=True).mean()
    print(f"  {k}: names ending in punct = {100*nm_t:.2f}%  |  addresses = {100*ad_t:.2f}%")

# -------------------------------------------------------------------------------------
# SCALE AUDIT on BOTH name AND address (GAP #3): nothing emptied, scripts preserved
# -------------------------------------------------------------------------------------
print("\n2.1 SCALE AUDIT (30k/source) — name AND address:")
for k in ["S1","S2","S3"]:
    df = sample_file(FILES[k], 30_000, cols=["business_name","business_address"])
    for field in ["business_name","business_address"]:
        raw = df[field]; out = raw.map(clean_text)
        emptied = ((raw.str.strip()!="") & (out.str.strip()=="")).sum()
        idx = raw.map(has_indic)
        kept = raw[idx].map(clean_text).map(has_indic).sum() if idx.any() else 0
        print(f"  {k} {field:16s}: emptied={emptied} (want 0) | "
              f"indic in={int(idx.sum())} kept={int(kept)}")

# -------------------------------------------------------------------------------------
# GAP #4 — IDEMPOTENCY: clean_text(clean_text(x)) == clean_text(x)
# -------------------------------------------------------------------------------------
print("\n2.1 IDEMPOTENCY CHECK (clean applied twice must equal once):")
mismatches = 0; tested = 0
for k in ["S1","S2","S3"]:
    df = sample_file(FILES[k], 20_000, cols=["business_name","business_address"])
    for field in ["business_name","business_address"]:
        once = df[field].map(clean_text)
        twice = once.map(clean_text)
        m = (once != twice).sum(); mismatches += m; tested += len(once)
        if m:
            bad = df[field][once != twice].head(3).tolist()
            print(f"  {k} {field}: {m} non-idempotent  e.g. {bad}")
print(f"  total non-idempotent: {mismatches} / {tested}  (want 0)")

print("\nPHASE 2.1 COMPLETE — clean_text registered; danda note, trailing-junk check, "
      "address audit, and idempotency all included.")


# ================================================================================
# --- notebook cell 48 ---
# ================================================================================
# =====================================================================================
# PHASE 2.2 — FINALIZED. Canonicalize-to-token (A/B winner) + entity_types side feature.
#   Change from prior run: name keeps a CANONICAL suffix token (variants collapsed) rather
#   than stripping it — A/B showed higher true-match lift (0.638 vs 0.618) and equal/safer
#   collision similarity (0.586 vs 0.589). Pruned 9 non-firing non-Latin surfaces.
#   Re-registers NORMALIZERS['suffix_base'] as the canonicalizer. Assumes clean_text.
# =====================================================================================

SUFFIX_MAP = {
    "incorporated": ["inc","incorporated","incorp"], "corporation": ["corp","corporation","corpn"],
    "limited": ["ltd","limited","lim"], "private": ["pvt","private","pvtltd","pv"],
    "llc": ["llc"], "llp": ["llp"], "pllc": ["pllc"], "lp": ["lp"], "pc": ["pc"],
    "company": ["co","company","cmpny"], "opc": ["opc"], "plc": ["plc"], "pte": ["pte"],
    "gmbh": ["gmbh"], "sarl": ["sarl"], "sas": ["sas"], "sa": ["sa"],
}
# PRUNED: removed the 9 non-Latin surfaces that never fired (company forms in all scripts,
# plus इंक / प्राईवेट / ఎల్ఎల్పీ / ಎಲ್ಎಲ್ಪಿ). Kept only empirically-firing forms.
SUFFIX_MAP_NONLATIN = {
    "private":      ["प्राइवेट","प्रा","பிரைவேட்","ప్రైవేట్","প্রাইভেট","પ્રાઇવેટ","ಪ್ರೈವೇಟ್","ପ୍ରାଇଭେଟ୍"],
    "limited":      ["लिमिटेड","लि","லிமிடெட்","లిమిటెడ్","লিমিটেড","લિમિટેડ","ಲಿಮಿಟೆಡ್","ଲିମିଟେଡ୍"],
    "llp":          ["एलएलपी","எல்எல்பி","এলএলপি","એલએલપી"],
}
CATEGORY_GUARD = {"consultants","traders","trading","foods","food","spa","salon","clinic","hospital",
                  "hotel","restaurant","bakery","motors","industries","enterprises","solutions",
                  "technologies","tech","infotech","software","finance","investments","logistics",
                  "exports","imports","electronics","furniture","jewellers","stores","mart","agencies",
                  "services","supplies","products","group","associates","builders","developers",
                  "realty","properties","constructions","construction","capital"}
LAT2CANON = {v:c for c,vs in SUFFIX_MAP.items() for v in vs if v not in CATEGORY_GUARD}
NON2CANON = {v:c for c,vs in SUFFIX_MAP_NONLATIN.items() for v in vs}
def _canon(tok): return LAT2CANON.get(tok) or NON2CANON.get(tok)

def normalize_suffix(cleaned):
    """CANONICALIZE variants at the END to their canonical token (kept in name).
       Returns (canonical_name, entity_types). End-only; never touches the sole token."""
    if not cleaned: return "", []
    toks = cleaned.split()
    types = []
    i = len(toks)
    while i > 1:                          # end-only, uniform across scripts
        c = _canon(toks[i-1])
        if c is None: break
        types.append(c); toks[i-1] = c    # canonicalize IN PLACE (keep token in name)
        i -= 1
    # collapse consecutive duplicate canonical tokens (e.g. pvt ltd already distinct; ltd ltd -> ltd)
    out = []
    for t in toks:
        if out and out[-1] == t and _canon(t): continue
        out.append(t)
    name = " ".join(out).strip() or cleaned
    return name, sorted(set(types))

def suffix_base(s):  return normalize_suffix(clean_text(s))[0]   # canonical NAME (pipeline step)
def suffix_types(s): return normalize_suffix(clean_text(s))[1]   # side FEATURE
register_normalizer("suffix_base", suffix_base)

# ---- READ-THROUGH: confirm canonicalize behavior + all guardrails still hold ----
print("\nFINALIZED READ-THROUGH — clean -> canonicalize (name | entity_types):\n")
demo = ["Maure Williams Colombier Inc","Maure Williams Colombier","Hotel Food Limited",
        "Grand Connecticut LLC","FH Business Pvt Ltd","Corp Corporation Corp",  # dup collapse
        "एसएस फूड प्राइवेट लिमिटेड","ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "Aadinath Consultants","Aadinath Traders","Co Operative Society",
        "प्राइवेट लिमिटेड इंटरप्राइजेज","Inc"]
for raw in demo:
    name, types = normalize_suffix(clean_text(raw))
    print(f"  raw : {raw}\n  ->  : name={name!r}  types={types}\n")

# ---- confirm the canonicalize A/B advantage holds on a fresh sample + safety checks ----
gt = read_tsv(FILES["GT"]); gt["ml"]=gt["matched_entity_ids"].apply(lambda s:[x for x in s.split(",") if x])
gt_nz = gt[gt.ml.str.len()>0]
s1n={};
for ch in read_tsv_chunks(FILES["S1"],cols=["entity_id","business_name"]): s1n.update(zip(ch.entity_id,ch.business_name))
mn={}
for k in ["S2","S3"]:
    for ch in read_tsv_chunks(FILES[k],cols=["entity_id","business_name"]): mn.update(zip(ch.entity_id,ch.business_name))
def jacc(a,b): A,B=set(a.split()),set(b.split()); return len(A&B)/max(len(A|B),1)

base=canon=0; n=0; emptied=0; mm=0
for _, r in gt_nz.sample(15_000, random_state=SEED).iterrows():
    a=s1n[r.source1_entity_id]
    for mid in r.ml[:1]:
        b=mn.get(mid,"")
        if not b: continue
        n+=1
        base += jacc(clean_text(a), clean_text(b))
        canon+= jacc(suffix_base(a), suffix_base(b))
for k in ["S1","S2","S3"]:
    df=sample_file(FILES[k],20000,cols=["business_name"])
    once=df.business_name.map(suffix_base)
    emptied += ((df.business_name.str.strip()!="") & (once.str.strip()=="")).sum()
    mm += (once != once.map(lambda x: normalize_suffix(x)[0])).sum()
print(f"true-match Jaccard: clean={base/n:.3f} -> canonicalize={canon/n:.3f}  (want increase)")
print(f"emptied names: {emptied} (want 0) | non-idempotent: {mm} (want 0)")
print("\nPHASE 2.2 FINALIZED — NORMALIZERS['suffix_base'] = canonicalize; suffix_types = side feature.")


# ================================================================================
# --- notebook cell 57 ---
# ================================================================================
# =====================================================================================
# PHASE 2.4 — ADDRESS PARSING / NORMALIZATION  (FINAL, single self-contained cell)
#   address_core (order-invariant token string, used for overlap/blocking):
#     A) street-abbreviation expansion (rd->road, ave->avenue, cir->circle ...)  -- O10
#     B) leading-zero normalization (0010433 -> 10433)
#     C) N/A placeholder + single-letter junk removal
#     D) landmark isolation (near/opp/behind ...) pulled into a SEPARATE field    -- O27
#   address_state: optional state-code FEATURE, extracted end-anchored, NOT in core.
#   Design: NO in-core state canonicalization (it caused idempotency + city/street/state
#   bugs and added ~0 to separation). Idempotent by construction (only stable ops + sort).
#   Assumes COMMON SETUP (read_tsv, read_tsv_chunks, sample_file, clean_text, register_normalizer).
# =====================================================================================

# -------------------------------------------------------------------------------------
# STEP A — verify street-abbreviation pairs actually co-occur in true address pairs
# -------------------------------------------------------------------------------------
print("STEP A — verifying street abbreviations occur in true address pairs ...")
gt = read_tsv(FILES["GT"]); gt["ml"] = gt["matched_entity_ids"].apply(lambda s:[x for x in s.split(",") if x])
gt_nz = gt[gt.ml.str.len()>0]
s1a = {}
for ch in read_tsv_chunks(FILES["S1"], cols=["entity_id","business_address"]): s1a.update(zip(ch.entity_id, ch.business_address))
ma = {}
for k in ["S2","S3"]:
    for ch in read_tsv_chunks(FILES[k], cols=["entity_id","business_address"]): ma.update(zip(ch.entity_id, ch.business_address))

ABBREV_PAIRS = {
    "road":"rd","street":"st","avenue":"ave","boulevard":"blvd","lane":"ln","drive":"dr",
    "court":"ct","circle":"cir","highway":"hwy","square":"sq","place":"pl","terrace":"ter",
    "parkway":"pkwy","suite":"ste","apartment":"apt","floor":"fl","building":"bldg",
    "block":"blk","sector":"sec","number":"no","opposite":"opp","near":"nr",
}
cooc = Counter()
for _, r in gt_nz.sample(30_000, random_state=SEED).iterrows():
    ta = set(clean_text(s1a[r.source1_entity_id]).split())
    for mid in r.ml[:1]:
        tb = set(clean_text(ma.get(mid,"")).split())
        for full, abbr in ABBREV_PAIRS.items():
            if (full in ta and abbr in tb) or (abbr in ta and full in tb):
                cooc[(full,abbr)] += 1
print("  abbrev<->full co-occurrences across true pairs:")
for (f,ab),c in cooc.most_common(15): print(f"     {f} <-> {ab}: {c:,}")

# -------------------------------------------------------------------------------------
# STEP B — tables
# -------------------------------------------------------------------------------------
# Street abbrev -> canonical FULL form (expand so both sides converge). Applied to core.
STREET_CANON = {**{ab:full for full,ab in ABBREV_PAIRS.items()}, **{full:full for full in ABBREV_PAIRS}}

# State tables — used ONLY for the separate state_code feature, never injected into core.
US_STATES = {
 "alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co",
 "connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga","hawaii":"hi","idaho":"id",
 "illinois":"il","indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la",
 "maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi","minnesota":"mn",
 "mississippi":"ms","missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv",
 "ohio":"oh","oklahoma":"ok","oregon":"or","pennsylvania":"pa","tennessee":"tn","texas":"tx",
 "utah":"ut","vermont":"vt","virginia":"va","washington":"wa","wisconsin":"wi","wyoming":"wy",
}
IN_STATES = {
 "maharashtra":"mh","gujarat":"gj","rajasthan":"rj","karnataka":"ka","telangana":"tg",
 "kerala":"kl","punjab":"pb","haryana":"hr","bihar":"br","odisha":"od","assam":"as",
 "goa":"goa","jharkhand":"jh","chhattisgarh":"cg","uttarakhand":"uk","sikkim":"sk",
 "manipur":"mn","mizoram":"mz","nagaland":"nl","tripura":"tr","meghalaya":"ml",
}
IN_MULTI = {
 "uttar pradesh":"up","madhya pradesh":"mp","tamil nadu":"tn","andhra pradesh":"ap",
 "west bengal":"wb","himachal pradesh":"hp","arunachal pradesh":"ar","jammu kashmir":"jk",
}
STATE_ALL   = {**US_STATES, **IN_STATES}
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values()) | set(IN_MULTI.values())

RE_LANDMARK = re.compile(r"\b(near|nr|beside|behind|opp|opposite|adjacent|next to|back side|backside)\b")
RE_LEADZERO = re.compile(r"\b0+(\d+)\b")
JUNK_TOKENS = {"n","a","na","nan"}

# -------------------------------------------------------------------------------------
# STEP C — normalizer. Returns (core_sorted, landmarks, state_code). IDEMPOTENT.
# -------------------------------------------------------------------------------------
def _extract_state_code(cleaned):
    """End-anchored state read for the FEATURE only (not injected into core)."""
    for name, code in IN_MULTI.items():
        if re.search(rf"\b{name}\b\s*$", cleaned): return code
    for t in reversed(cleaned.split()[-3:]):
        if t in STATE_CODES: return t
        if t in STATE_ALL:   return STATE_ALL[t]
    return ""

def normalize_address(raw):
    s = clean_text(raw)
    if not s:
        return "", "", ""
    s = re.sub(r"\bn a\b", " ", s)
    s = re.sub(r"\bna\b", " ", s)
    state_code = _extract_state_code(s)          # feature computed once, never fed back in
    s = RE_LEADZERO.sub(r"\1", s)                # B) leading zeros
    toks = s.split()
    core, landmarks = [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if RE_LANDMARK.match(t):                 # D) landmark -> separate field
            landmarks.append(t)
            for j in range(i+1, min(i+3, len(toks))):
                landmarks.append(toks[j])
            i = min(i+3, len(toks)); continue
        t = STREET_CANON.get(t, t)               # A) street abbrev -> full (stable/idempotent)
        if t not in JUNK_TOKENS:                 # C) drop N/A junk / single letters
            core.append(t)
        i += 1
    return " ".join(sorted(set(core))), " ".join(landmarks), state_code

def address_core(raw):  return normalize_address(raw)[0]
def address_state(raw): return normalize_address(raw)[2]
register_normalizer("address_core", address_core)

# -------------------------------------------------------------------------------------
# READ-THROUGH #1 — raw -> (core | landmarks | state feature)
# -------------------------------------------------------------------------------------
print("\nREAD-THROUGH — raw -> (core | land | state_feat):\n")
demo = [
 "88 Olive Circle, Lebanon, TN", "OLIVE CIR, LEBANON, TN",
 "TX, 158 Simpson Lane, Somerset", "158 Simpson Lane, Somerset, Texas",
 "45, Teli Gulli Park Rd, Near Vishal Hall, Mumbai, Maharashtra",
 "Hno.0010433, Gali No.322, Sanjay Colony, Haryana",
 "Calumet City, 351 Hoxie Avenue, IL", "CALUMET CITY, 351 HOXIE AVE, N/A, IL",
 "2901 Connecticut Ave, Washington, DC",
 "A 19 Pratap Nagar, Lucknow, Uttar Pradesh",
]
for d in demo:
    core, lm, st = normalize_address(d)
    print(f"  raw   : {d}")
    print(f"  core  : {core}")
    print(f"  land  : {lm!r}   state_feat: {st!r}\n")

# -------------------------------------------------------------------------------------
# READ-THROUGH #2 — MATCHES: real true pairs where normalization raises Jaccard
# -------------------------------------------------------------------------------------
def jacc(a,b): A,B=set(a.split()),set(b.split()); return len(A&B)/max(len(A|B),1)
print("="*70); print("MATCHES — true address pairs (Jaccard before -> after)"); print("="*70)
shown=0
for _, r in gt_nz.sample(20_000, random_state=SEED).iterrows():
    a=s1a[r.source1_entity_id]
    for mid in r.ml[:1]:
        b=ma.get(mid,"")
        if not (a.strip() and b.strip()): continue
        jb=jacc(clean_text(a),clean_text(b)); ja=jacc(address_core(a),address_core(b))
        if ja>jb+0.05 and shown<8:
            print(f"  {jb:.2f}->{ja:.2f}")
            print(f"     S1: {a}\n     M : {b}")
            print(f"     S1 core: {address_core(a)}\n     M  core: {address_core(b)}\n")
            shown+=1
    if shown>=8: break

# -------------------------------------------------------------------------------------
# SEPARATION (true vs random) + SAFETY (emptied + idempotency)
# -------------------------------------------------------------------------------------
import random as _r; _r.seed(SEED); rand_addr=list(ma.values())
tb=ta=rb=ra=0; n=0
for _, r in gt_nz.sample(8000, random_state=SEED).iterrows():
    a=s1a[r.source1_entity_id]
    for mid in r.ml[:1]:
        b=ma.get(mid,"")
        if not (a.strip() and b.strip()): continue
        n+=1
        tb+=jacc(clean_text(a),clean_text(b)); ta+=jacc(address_core(a),address_core(b))
        rnd=_r.choice(rand_addr)
        rb+=jacc(clean_text(a),clean_text(rnd)); ra+=jacc(address_core(a),address_core(rnd))
print("="*70); print("SEPARATION — true vs random"); print("="*70)
print(f"  TRUE   Jaccard: clean={tb/n:.3f} -> normalized={ta/n:.3f}")
print(f"  RANDOM Jaccard: clean={rb/n:.3f} -> normalized={ra/n:.3f}")
print(f"  SEPARATION    : clean={(tb-rb)/n:.3f} -> normalized={(ta-ra)/n:.3f}")

em=mm=tot=0
for k in ["S1","S2","S3"]:
    df=sample_file(FILES[k],15000,cols=["business_address"])
    once=df.business_address.map(address_core); twice=once.map(address_core)
    em+=((df.business_address.str.strip()!="") & (once.str.strip()=="")).sum()
    mm+=(once!=twice).sum(); tot+=len(once)
print("\n" + "="*70); print("SAFETY"); print("="*70)
print(f"  emptied={em} (want 0) | non-idempotent={mm}/{tot} (want 0)")
print("\nPHASE 2.4 COMPLETE — address_core (idempotent) + address_state feature registered.")


# ================================================================================
# --- notebook cell 58 ---
# ================================================================================
# =====================================================================================
# PHASE 2.5 — MISSING-DATA HANDLING  (assumes COMMON SETUP + clean_text + address_core)
#   ~3.3% of S2/S3 have empty address (O2). Define explicit behavior: a record's match
#   readiness is a MODE flag, and downstream must fall back to name-only for these.
#   Deliverable: record_profile() -> which fields are usable + the fallback mode.
# =====================================================================================

def record_profile(name_raw, addr_raw):
    """Explicit missing-data contract for one record.
       Returns a dict the blocker/matcher reads to decide name-only vs name+address."""
    name_c = suffix_base(name_raw)                 # canonical name (2.2)
    addr_c = address_core(addr_raw)                # canonical address core (2.4)
    has_name = bool(name_c.strip())
    has_addr = bool(addr_c.strip())
    if has_name and has_addr:  mode = "name_and_address"
    elif has_name:             mode = "name_only"        # <- the ~3.3% empty-address path
    elif has_addr:             mode = "address_only"     # rare; name should never be empty (O1)
    else:                      mode = "unusable"         # should be ~0
    return {"name": name_c, "address": addr_c, "has_name": has_name,
            "has_addr": has_addr, "mode": mode}

# -------------------------------------------------------------------------------------
# MEASURE the mode distribution at scale (confirm the ~3.3% name_only path + ~0 unusable)
# -------------------------------------------------------------------------------------
print("2.5 — record mode distribution per source (30k sample):")
mode_counts = defaultdict(Counter)
for k in ["S1","S2","S3"]:
    df = sample_file(FILES[k], 30_000, cols=["business_name","business_address"])
    for nm, ad in zip(df.business_name, df.business_address):
        mode_counts[k][record_profile(nm, ad)["mode"]] += 1
    tot = sum(mode_counts[k].values())
    print(f"  {k}: " + "  ".join(f"{m}={100*c/tot:.2f}%" for m,c in mode_counts[k].most_common()))

# -------------------------------------------------------------------------------------
# READ-THROUGH — the empty-address records (the name-only fallback path)
# -------------------------------------------------------------------------------------
print("\nREAD-THROUGH — empty-address records -> name_only mode:")
shown=0
for ch in read_tsv_chunks(FILES["S2"], cols=["entity_id","business_name","business_address"]):
    for _, row in ch.iterrows():
        if row.business_address.strip()=="" and shown<6:
            p = record_profile(row.business_name, row.business_address)
            print(f"  {row.entity_id}  name={p['name']!r}  addr={p['address']!r}  -> mode={p['mode']}")
            shown+=1
    if shown>=6: break

# CRITICAL CHECK: are empty-address records still MATCHABLE via name? (they must be)
# -> confirm true matches exist for empty-address S2/S3 records, so name-only is viable.
print("\n2.5 — are empty-address matched records recoverable by NAME alone?")
gt = read_tsv(FILES["GT"]); gt["ml"]=gt["matched_entity_ids"].apply(lambda s:[x for x in s.split(",") if x])
gt_nz = gt[gt.ml.str.len()>0]
s1n={};
for c in read_tsv_chunks(FILES["S1"],cols=["entity_id","business_name"]): s1n.update(zip(c.entity_id,c.business_name))
mn={}; maddr={}
for k in ["S2","S3"]:
    for c in read_tsv_chunks(FILES[k],cols=["entity_id","business_name","business_address"]):
        mn.update(zip(c.entity_id,c.business_name)); maddr.update(zip(c.entity_id,c.business_address))
def jacc(a,b): A,B=set(a.split()),set(b.split()); return len(A&B)/max(len(A|B),1)
empty_addr_pairs=[];
for _, r in gt_nz.sample(60_000, random_state=SEED).iterrows():
    for mid in r.ml:
        if maddr.get(mid,"").strip()=="":
            empty_addr_pairs.append((s1n[r.source1_entity_id], mn.get(mid,""))); break
    if len(empty_addr_pairs)>=500: break
if empty_addr_pairs:
    nm_jacc = np.mean([jacc(suffix_base(a), suffix_base(b)) for a,b in empty_addr_pairs])
    print(f"  empty-address true pairs sampled: {len(empty_addr_pairs)}")
    print(f"  mean NAME Jaccard on these pairs : {nm_jacc:.3f}  (high => name-only fallback works)")
    for a,b in empty_addr_pairs[:5]: print(f"     '{suffix_base(a)}'  <>  '{suffix_base(b)}'")
print("\nPHASE 2.5 COMPLETE — record_profile() defines the name-only fallback contract.")


# ================================================================================
# --- notebook cell 59 ---
# ================================================================================
# =====================================================================================
# PHASE 2.6 — CATEGORY-WORD EXTRACTION  (assumes COMMON SETUP + clean_text/suffix_base)
#   Pull business-type words (traders/consultants/foods/...) out as a SEPARATE signal so
#   the system knows 'Aadinath Consultants' vs 'Aadinath Traders' are NOT the same (O30/T1).
#   Deliverable: split_name() -> (brand_base, category_words). Category MISMATCH is a
#   strong NON-match signal for the matcher (Phase 4). Category lexicon is DATA-DERIVED.
# =====================================================================================

# -------------------------------------------------------------------------------------
# STEP A — DERIVE the category lexicon from data, don't hardcode from memory.
#   Category words = frequent tokens that (a) recur across many DIFFERENT brand bases and
#   (b) are the ONLY differentiator in T1 collisions. Mine frequent non-suffix tokens.
# -------------------------------------------------------------------------------------
print("2.6 STEP A — deriving category lexicon from frequent shared tokens ...")
tok_df = Counter(); n_docs = 0
for k in ["S1","S2","S3"]:
    df = sample_file(FILES[k], 100_000, cols=["business_name"])
    for nm in df.business_name:
        n_docs += 1
        toks = set(suffix_base(nm).split())        # suffixes already canonicalized/removed-ish
        tok_df.update(toks)
# candidate category words: appear in >0.3% of names AND are dictionary-like business types
CATEGORY_SEED = {"traders","trading","consultants","consultancy","foods","food","spa","salon",
 "clinic","hospital","hotel","hotels","restaurant","bakery","cafe","motors","automobiles",
 "industries","enterprises","solutions","technologies","tech","infotech","software","finance",
 "investments","capital","logistics","transport","exports","imports","electronics","electricals",
 "hardware","furniture","jewellers","jewelry","stores","store","mart","agencies","agency",
 "services","supplies","products","group","associates","builders","developers","realty",
 "properties","constructions","construction","medical","pharmacy","clothing","apparel","textiles",
 "education","academy","school","institute","ventures","holdings","corporation","company"}
# keep only seeds that actually appear frequently in THIS data (data-validation of the lexicon)
CATEGORY_WORDS = {w for w in CATEGORY_SEED if tok_df.get(w,0)/n_docs > 0.001}
print(f"  category words kept (>0.1% doc-freq): {len(CATEGORY_WORDS)}")
print(f"  top by frequency: {sorted(CATEGORY_WORDS, key=lambda w:-tok_df[w])[:20]}")

# -------------------------------------------------------------------------------------
# STEP B — split_name(): brand base (category words removed) + the category words
# -------------------------------------------------------------------------------------
def split_name(raw):
    canon = suffix_base(raw)                        # clean + suffix-canonicalized name
    toks = canon.split()
    cats = [t for t in toks if t in CATEGORY_WORDS]
    base = [t for t in toks if t not in CATEGORY_WORDS]
    base_str = " ".join(base) if base else canon    # never empty (if name IS only a category word)
    return base_str, sorted(set(cats))

def name_base(raw):  return split_name(raw)[0]
def name_cats(raw):  return split_name(raw)[1]
register_normalizer("name_base", name_base)

# -------------------------------------------------------------------------------------
# READ-THROUGH — the T1 collision case + normal cases (see brand vs category split)
# -------------------------------------------------------------------------------------
print("\nREAD-THROUGH — split_name -> (brand_base | category_words):\n")
demo = ["Aadinath Consultants","Aadinath Traders","Aadinath Spa",
        "Maure Williams Colombier Inc","Big Shiva Logistics Private Limited",
        "Big Shiva Foods Private Limited","Sunrise Bakery","Consultants"]
for d in demo:
    base, cats = split_name(d)
    print(f"  {d!r}\n    -> base={base!r}  cats={cats}\n")

# -------------------------------------------------------------------------------------
# THE PAYOFF — category MISMATCH separates T1 collisions from true matches
#   For true pairs: category words should AGREE. For T1 collisions: they DISAGREE.
# -------------------------------------------------------------------------------------
def cat_agree(a, b):
    ca, cb = set(name_cats(a)), set(name_cats(b))
    if not ca and not cb: return 1.0               # neither has a category -> neutral
    if not ca or not cb:  return 0.5
    return len(ca & cb)/max(len(ca | cb),1)

print("="*68); print("PAYOFF — category agreement: TRUE matches vs T1 collisions"); print("="*68)
# true matches
tr=[];
for _, r in gt_nz.sample(8000, random_state=SEED).iterrows():
    a=s1n[r.source1_entity_id]
    for mid in r.ml[:1]:
        b=mn.get(mid,"")
        if b: tr.append(cat_agree(a,b)); break
print(f"  TRUE matches  : mean category-agreement = {np.mean(tr):.3f}  (want HIGH)")
# T1 collisions from hard-case set
try:
    hc = pd.read_csv(f"{WORK}/hard_cases_validation.csv", dtype=str, keep_default_na=False)
    t1_ag=[]
    for _, r in hc[hc["trap"]=="T1_category_collision"].iterrows():
        parts=[p.strip() for p in str(r["names"]).split("|") if p.strip()][:2]
        if len(parts)==2: t1_ag.append(cat_agree(parts[0], parts[1]))
    print(f"  T1 collisions : mean category-agreement = {np.mean(t1_ag):.3f}  (want LOWER -> separates!)")
    print("  examples (different businesses, category words differ):")
    for _, r in hc[hc["trap"]=="T1_category_collision"].head(5).iterrows():
        parts=[p.strip() for p in str(r["names"]).split("|") if p.strip()][:2]
        if len(parts)==2:
            print(f"     {name_cats(parts[0])} vs {name_cats(parts[1])}   |   {parts[0]}  <>  {parts[1]}")
except FileNotFoundError:
    print("  (hard_cases_validation.csv not found)")

print("\nPHASE 2.6 COMPLETE — split_name() gives brand_base + category words; "
      "category mismatch is a NON-match signal for Phase 4.")


# ================================================================================
# --- notebook cell 60 ---
# ================================================================================
# =====================================================================================
# PHASE 2.7 — PIPELINE ASSEMBLY + BGE-M3 WIRING  (assumes COMMON SETUP + all normalizers)
#   Assemble everything from 2.1-2.6 into ONE record processor, expose the full feature
#   set each record carries into blocking (Phase 3) and matching (Phase 4).
#   BGE-M3 is the multilingual engine (decided in 2.3) — exposed, not a text normalizer.
# =====================================================================================

def process_record(name_raw, addr_raw):
    """Single source of truth: raw (name,address) -> all normalized fields + features.
       Used identically for S1/S2/S3, train and test."""
    prof   = record_profile(name_raw, addr_raw)          # 2.5: mode + has_name/has_addr
    canon_name          = suffix_base(name_raw)          # 2.1+2.2: clean + suffix-canonical
    entity_types        = suffix_types(name_raw)         # 2.2: side feature
    brand_base, cats    = split_name(name_raw)           # 2.6: brand vs category words
    addr_c, landmarks, state = normalize_address(addr_raw)  # 2.4: core + landmark + state
    return {
        "mode":          prof["mode"],                   # name_and_address | name_only | ...
        "canon_name":    canon_name,                     # canonical full name (for embedding)
        "brand_base":    brand_base,                     # category-stripped brand (blocking key)
        "category_words":cats,                           # precision feature (2.6)
        "entity_types":  entity_types,                   # precision feature (2.2)
        "addr_core":     addr_c,                         # order-invariant address tokens
        "addr_landmarks":landmarks,                      # separated, low-signal
        "addr_state":    state,                          # optional locality feature
        "has_addr":      prof["has_addr"],
    }

# what text goes to BGE-M3: canonical name (+ address core when present) — one embedding per record
def embed_text(rec):
    return rec["canon_name"] if not rec["has_addr"] else f"{rec['canon_name']} {rec['addr_core']}"

# -------------------------------------------------------------------------------------
# READ-THROUGH — full pipeline on real records (see every field a record carries forward)
# -------------------------------------------------------------------------------------
print("2.7 — full record processing (all fields):\n")
samples = [
    ("Aadinath Consultants", "12 MG Road, Near City Mall, Pune, Maharashtra"),
    ("एसएस फूड प्राइवेट लिमिटेड", "Plot 5, Sector 18, Gurgaon, Haryana"),
    ("Maure Wilblims Colombier Inc", ""),                # empty-address -> name_only
    ("teamair.com", "158 Simpson Lane, Somerset, Texas"),
]
for nm, ad in samples:
    rec = process_record(nm, ad)
    print(f"  RAW name={nm!r} addr={ad!r}")
    for kk,vv in rec.items(): print(f"     {kk:16s}: {vv!r}")
    print(f"     embed_text     : {embed_text(rec)!r}\n")

print("NORMALIZERS registry:", list(NORMALIZERS))
print("BGE-M3: multilingual engine for blocking(Phase3) + matcher feature(Phase4) via bgem3_embed()")
print("PHASE 2.7 COMPLETE — unified process_record() ready.\n")

# =====================================================================================
# PHASE 2.8 — TEST-SET / FRANCE GENERALIZATION AUDIT
#   France appears ONLY in test. Audit EVERY normalizer: does it no-op SAFELY on French
#   text (not destroy it, not crash, not empty it) rather than assuming US/India patterns?
#   Uses real French records from the TEST files (first look at test data — read-only).
# =====================================================================================
print("2.8 — FRANCE GENERALIZATION AUDIT (test-only country)\n")

# pull real French records from test files (we never train on these; audit only)
fr_rows=[]
for k in ["test_S2","test_S3","test_S1"]:
    for ch in read_tsv_chunks(FILES[k], cols=["business_name","business_address","country"]):
        fr = ch[ch.country.str.strip()=="France"]
        fr_rows.extend(list(zip(fr.business_name, fr.business_address)))
        if len(fr_rows)>=3000: break
    if len(fr_rows)>=3000: break
print(f"  real French test records sampled: {len(fr_rows):,}")

# AUDIT 1: no normalizer empties a non-empty French field (safe no-op, not destruction)
emptied_name=emptied_addr=0
for nm, ad in fr_rows:
    if nm.strip() and not suffix_base(nm).strip(): emptied_name+=1
    if ad.strip() and not address_core(ad).strip(): emptied_addr+=1
print(f"  French names emptied by pipeline   : {emptied_name} (want 0)")
print(f"  French addresses emptied by pipeline: {emptied_addr} (want 0)")

# AUDIT 2: French legal suffixes (SARL/SAS/SA) actually normalize (not silently ignored)
fr_suffix_fire = Counter()
for nm, _ in fr_rows:
    for t in suffix_types(nm): fr_suffix_fire[t]+=1
print(f"  French entity-type firing (SARL/SAS/SA expected): {dict(fr_suffix_fire)}")

# AUDIT 3: French accents preserved (é/è/à not stripped to nothing), scripts intact
acc = [nm for nm,_ in fr_rows if RE_NONLATIN.search(nm)]
acc_kept = sum(1 for nm in acc if RE_NONLATIN.search(clean_text(nm)))
print(f"  French non-ASCII names: {len(acc)} | still non-ASCII after clean: {acc_kept} (want equal)")

# AUDIT 4: no country-gated branch fires — confirm pipeline is identical regardless of country
#   (French records shouldn't hit any India/US-specific dead path; they just process normally)
print("\n  READ-THROUGH — real French records through the FULL pipeline:")
for nm, ad in fr_rows[:6]:
    rec = process_record(nm, ad)
    print(f"     name={nm!r}")
    print(f"       -> canon={rec['canon_name']!r} base={rec['brand_base']!r} "
          f"cats={rec['category_words']} types={rec['entity_types']}")
    print(f"     addr={ad!r}")
    print(f"       -> core={rec['addr_core']!r} state={rec['addr_state']!r} land={rec['addr_landmarks']!r}\n")

# AUDIT 5: BGE-M3 handles French cross-source similarity (multilingual engine covers FR too)
try:
    fr_probe = [nm for nm,_ in fr_rows[:20] if nm.strip()][:6]
    if len(fr_probe)>=2:
        E = bgem3_embed([clean_text(x) for x in fr_probe])
        # same-name vs different-name sanity: identical text -> ~1.0, different -> lower
        same = float((bgem3_embed([clean_text(fr_probe[0])])[0]*bgem3_embed([clean_text(fr_probe[0])])[0]).sum())
        diff = float((E[0]*E[1]).sum())
        print(f"  BGE-M3 on French: identical={same:.3f} (want ~1.0)  two-different={diff:.3f} (want lower)")
except Exception as e:
    print(f"  (BGE-M3 French probe skipped: {e})")

print("\nPHASE 2.8 COMPLETE — France audit: no destruction, suffixes fire, accents kept, "
      "no country-gated paths, embedding covers FR.")


# ================================================================================
# --- notebook cell 71 ---
# ================================================================================
# =====================================================================================
# PRIORITY 1a — robust domain/glued-token segmentation in brand_base (Phase-2 upgrade).
#   Strip corrupted TLDs (com/c0m/net/..), segment long glued alphabetic tokens with
#   wordninja, KEEP BOTH glued + segmented (union) so either can match. Re-verify.
# =====================================================================================
import re, wordninja

TLD = {"com","c0m","net","org","in","co","io","biz","info","us","fr","cor","comm"}
def desegment_token(tok):
    """Return set of forms for a possibly-glued/domain token: {original, *wordninja pieces}."""
    forms={tok}
    core = tok
    # strip a trailing TLD-ish chunk if glued (geniusbrotherscom -> geniusbrothers)
    for tld in sorted(TLD, key=len, reverse=True):
        if core.endswith(tld) and len(core)>len(tld)+3:
            core=core[:-len(tld)]; break
    # segment long alphabetic tokens (glued names/domains) — keep pieces of length>=3
    if core.isalpha() and len(core)>=10:
        pieces=[p for p in wordninja.split(core) if len(p)>=3]
        if len(pieces)>=2: forms.update(pieces)
    elif core!=tok:
        forms.add(core)
    return forms

def brand_base_v2(canon_name):
    """Upgraded brand_base: expand glued/domain tokens into their segmented forms too."""
    toks=canon_name.split()
    out=set()
    for t in toks:
        if t in TLD: continue                    # drop bare TLD token 'com'
        out |= desegment_token(t)
    return " ".join(sorted(out))

# apply to the existing brand_base of each record (cheap, no re-embed) — patch S1 + POOL
print("re-segmenting brand_base for glued/domain tokens ...")
for d in (S1, POOL):
    for e,r in d.items():
        r["brand_base"] = brand_base_v2(r["brand_base"] or r["canon_name"])

# re-verify the same name-unrelated class now recovers by NAME token overlap
def toks(s): return set(t for t in s.split() if len(t)>1)
recovered=still_dead=0
examples=[]
auto_hard=[]
for e, mids in true_map.items():
    if e not in S1: continue
    # NOTE: recompute overlap on the NEW brand_base
    for mid in mids:
        if mid in POOL: auto_hard.append((e,mid))
    if len(auto_hard)>=400: break
checked=0
for e,mid in auto_hard:
    a=toks(S1[e]["brand_base"]); b=toks(POOL[mid]["brand_base"])
    # only look at pairs that were name-unrelated under the ORIGINAL (proxy: still test overlap now)
    if not a or not b: continue
    checked+=1
    if a & b: recovered+=1
    else:
        still_dead+=1
        if len(examples)<12: examples.append((S1[e]["brand_base"], POOL[mid]["brand_base"]))
print(f"\nafter segmentation, of {checked} true pairs: share NAME tokens now = {recovered} "
      f"({100*recovered/max(checked,1):.1f}%)")
print("remaining zero-name-overlap (genuinely dead / acronym / script) examples:")
for a,b in examples: print(f"   {a!r}  <>  {b!r}")
print("\n-> DOMAIN/glued class should now match by NAME; residual = genuine-dead + acronym + script (Priority 1b).")


