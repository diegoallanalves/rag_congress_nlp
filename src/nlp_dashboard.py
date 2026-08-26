
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

THEME_LEXICON = {
    "Procurement / Market Exclusion": {
        "weight": 25,
        "terms": [
            "prohibit procurement", "procurement ban", "prohibition on procurement",
            "shall not procure", "may not procure", "exclude", "excluded",
            "restriction on procurement", "chinese origin", "prc origin",
            "people's republic of china-origin", "government procurement"
        ],
    },
    "Trade / Tariffs / Import Controls": {
        "weight": 20,
        "terms": [
            "tariff", "duties", "duty", "import restriction", "restrict imports",
            "importation", "export control", "export license", "trade restriction",
            "most-favored-nation", "normal trade relations", "customs"
        ],
    },
    "Sanctions / Entity Restrictions": {
        "weight": 20,
        "terms": [
            "sanction", "sanctions", "entity list", "restricted entity",
            "chinese entity", "prc entity", "blocked person", "designation",
            "special designated", "economic sanctions"
        ],
    },
    "Ownership / Investment Controls": {
        "weight": 20,
        "terms": [
            "cfius", "foreign investment", "chinese-owned", "prc-owned",
            "ownership restriction", "foreign ownership", "acquisition",
            "investment restriction", "covered transaction", "national security review"
        ],
    },
    "Domestic Supply Chain / Reshoring": {
        "weight": 15,
        "terms": [
            "buy america", "buy american", "domestic content", "domestic manufacturing",
            "reshoring", "onshoring", "supply chain security", "supply chain resilience",
            "reduce dependence", "reduce reliance", "critical supply chain",
            "made in america", "domestic source", "trusted supplier"
        ],
    },
}

OPPORTUNITY_TERMS = [
    "nearshoring", "usmca", "duty-free", "duty free", "manufacturing incentive",
    "tax credit", "grant", "financing", "grid investment", "infrastructure investment",
    "domestic manufacturing opportunity", "mexico manufacturing"
]

def congress_from_url(url):
    s = str(url or "")
    patterns = [
        r"/bills/(\d{3})/",
        r"/congress/bills/(\d{3})/",
        r"/(\d{3})/(?:s|hr|hjres|hres|sres|sjres)/",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return f"{m.group(1)}th"
    return "Unknown"

def _safe_text(v):
    if pd.isna(v):
        return ""
    return str(v)

def normalize_confidence(v):
    s = str(v or "").strip().upper()
    return {"HIGH": 0.95, "MEDIUM": 0.65, "LOW": 0.35}.get(s, 0.50)

def classify_theme_hits(text):
    t = text.lower()
    hits = {}
    for theme, cfg in THEME_LEXICON.items():
        matched = [term for term in cfg["terms"] if term in t]
        if matched:
            hits[theme] = matched
    return hits

def exclusion_score(row):
    combined = " ".join([
        _safe_text(row.get("Mechanism")),
        _safe_text(row.get("Analytical Summary")),
        _safe_text(row.get("Evidence Text")),
    ]).lower()

    hits = classify_theme_hits(combined)
    score = sum(THEME_LEXICON[k]["weight"] for k in hits)

    relevant = str(row.get("Relevant", "")).upper() == "YES"
    classification = str(row.get("Primary Classification", "")).upper()
    directness = str(row.get("Directness", "")).upper()

    if relevant:
        score += 8
    if "RISK" in classification:
        score += 7
    if directness == "DIRECT":
        score += 5
    elif "SENTIMENT" in directness or "PRESSURE" in directness:
        score += 2

    # Opportunity language tempers, rather than eliminates, exclusion pressure.
    opp_count = sum(1 for term in OPPORTUNITY_TERMS if term in combined)
    score -= min(opp_count * 3, 12)

    return max(0, min(100, score)), hits

def score_band(score):
    if score >= 70:
        return "High"
    if score >= 45:
        return "Elevated"
    if score >= 20:
        return "Watch"
    return "Low"

def load_run(path, run_name):
    all_df = pd.read_excel(path, sheet_name="All Results")
    try:
        ev = pd.read_excel(path, sheet_name="Evidence")
    except Exception:
        ev = pd.DataFrame()

    all_df.columns = [str(c).strip() for c in all_df.columns]
    if not ev.empty:
        ev.columns = [str(c).strip() for c in ev.columns]

    if not ev.empty and "Bill" in ev.columns:
        ev["Evidence Combined"] = (
            ev.get("Quote", "").fillna("").astype(str)
            + " | "
            + ev.get("Why It Matters", "").fillna("").astype(str)
        )
        grouped = (
            ev.groupby("Bill", dropna=False)
              .agg({
                  "Evidence Combined": lambda s: " || ".join(x for x in s if x),
                  "Quote": lambda s: " || ".join(str(x) for x in s.dropna().head(8)),
                  "Why It Matters": lambda s: " || ".join(str(x) for x in s.dropna().head(8)),
              })
              .reset_index()
        )
        grouped = grouped.rename(columns={
            "Evidence Combined": "Evidence Text",
            "Quote": "Evidence Quotes",
            "Why It Matters": "Evidence Why",
        })
        all_df = all_df.merge(grouped, on="Bill", how="left")
    else:
        all_df["Evidence Text"] = ""
        all_df["Evidence Quotes"] = ""
        all_df["Evidence Why"] = ""

    all_df["Run"] = run_name
    all_df["Congress"] = all_df["Source URL"].apply(congress_from_url)
    all_df["Confidence Numeric"] = all_df.get("Confidence", "").apply(normalize_confidence)

    scores, themes, counts = [], [], []
    for _, row in all_df.iterrows():
        sc, hit = exclusion_score(row)
        scores.append(sc)
        themes.append("; ".join(hit.keys()))
        counts.append(sum(len(v) for v in hit.values()))
    all_df["China Exclusion Index"] = scores
    all_df["Exclusion Band"] = [score_band(s) for s in scores]
    all_df["NLP Themes"] = themes
    all_df["Restriction Signal Count"] = counts
    return all_df

def build_consensus(run1, run2):
    """
    Compare two RAG runs using Bill + Congress as the research key.
    Bill numbers can repeat across different Congresses, so Bill alone is unsafe.
    """
    cols = [
        "Bill", "Title", "Country", "Congress", "Relevant", "Primary Classification",
        "Policy Stage", "Directness", "Affected Scenarios", "Mechanism",
        "Analytical Summary", "Confidence Numeric", "Evidence Source", "Source URL",
        "Evidence Quotes", "Evidence Why", "Evidence Text", "China Exclusion Index",
        "Exclusion Band", "NLP Themes", "Restriction Signal Count"
    ]

    a = run1[[c for c in cols if c in run1.columns]].copy()
    b = run2[[c for c in cols if c in run2.columns]].copy()

    # Deduplicate within each run on the stable research key.
    a = a.sort_values(["Bill","Congress"]).drop_duplicates(["Bill","Congress"], keep="first")
    b = b.sort_values(["Bill","Congress"]).drop_duplicates(["Bill","Congress"], keep="first")

    key = ["Bill","Congress"]
    a = a.add_suffix(" Run 1").rename(columns={"Bill Run 1":"Bill","Congress Run 1":"Congress"})
    b = b.add_suffix(" Run 2").rename(columns={"Bill Run 2":"Bill","Congress Run 2":"Congress"})
    merged = a.merge(b, on=key, how="outer", validate="one_to_one")

    def coalesce(c1, c2):
        x = merged[c1] if c1 in merged else pd.Series([""] * len(merged), index=merged.index)
        y = merged[c2] if c2 in merged else pd.Series([""] * len(merged), index=merged.index)
        return x.where(x.notna() & (x.astype(str) != ""), y)

    result = pd.DataFrame({"Bill": merged["Bill"], "Congress": merged["Congress"]})
    for base_col in ["Title","Country","Policy Stage","Affected Scenarios","Source URL"]:
        result[base_col] = coalesce(f"{base_col} Run 1", f"{base_col} Run 2")

    r1 = merged.get("Relevant Run 1", pd.Series([""]*len(merged), index=merged.index)).fillna("")
    r2 = merged.get("Relevant Run 2", pd.Series([""]*len(merged), index=merged.index)).fillna("")
    c1 = merged.get("Primary Classification Run 1", pd.Series([""]*len(merged), index=merged.index)).fillna("")
    c2 = merged.get("Primary Classification Run 2", pd.Series([""]*len(merged), index=merged.index)).fillna("")

    result["Run Agreement"] = np.where(
        (r1 == r2) & (c1 == c2), "Full agreement",
        np.where(r1 == r2, "Relevance agreement", "Disagreement")
    )
    result["Relevant"] = np.where(
        r1 == r2, r1,
        np.where((r1=="YES") | (r2=="YES"), "YES", "NO")
    )
    result["Primary Classification"] = np.where(
        c1 == c2, c1, c1.where(c1 != "", c2)
    )

    s1 = pd.to_numeric(merged.get("China Exclusion Index Run 1", 0), errors="coerce").fillna(0)
    s2 = pd.to_numeric(merged.get("China Exclusion Index Run 2", 0), errors="coerce").fillna(0)
    both_present = (
        merged.get("China Exclusion Index Run 1", pd.Series([np.nan]*len(merged),index=merged.index)).notna()
        & merged.get("China Exclusion Index Run 2", pd.Series([np.nan]*len(merged),index=merged.index)).notna()
    )
    result["China Exclusion Index"] = np.where(
        both_present, (s1+s2)/2, np.where(s1>0,s1,s2)
    ).round(1)
    result["Exclusion Band"] = result["China Exclusion Index"].apply(score_band)

    conf1 = pd.to_numeric(merged.get("Confidence Numeric Run 1", 0.5), errors="coerce").fillna(0.5)
    conf2 = pd.to_numeric(merged.get("Confidence Numeric Run 2", 0.5), errors="coerce").fillna(0.5)
    agreement_bonus = result["Run Agreement"].map(
        {"Full agreement":0.10,"Relevance agreement":0.04,"Disagreement":-0.12}
    ).fillna(0)
    result["Consensus Confidence"] = (((conf1+conf2)/2 + agreement_bonus).clip(0,1)*100).round(0)

    result["NLP Themes"] = (
        coalesce("NLP Themes Run 1","NLP Themes Run 2").fillna("")
        + "; "
        + coalesce("NLP Themes Run 2","NLP Themes Run 1").fillna("")
    ).str.strip("; ").apply(
        lambda x: "; ".join(dict.fromkeys([p.strip() for p in x.split(";") if p.strip()]))
    )

    result["Mechanism"] = coalesce("Mechanism Run 1","Mechanism Run 2")
    result["Analytical Summary"] = coalesce("Analytical Summary Run 1","Analytical Summary Run 2")
    result["Evidence Quotes"] = coalesce("Evidence Quotes Run 1","Evidence Quotes Run 2")
    result["Evidence Why"] = coalesce("Evidence Why Run 1","Evidence Why Run 2")
    result["Evidence Source"] = coalesce("Evidence Source Run 1","Evidence Source Run 2")

    return result

def theme_counts(df):
    counter = Counter()
    for val in df.get("NLP Themes", pd.Series(dtype=str)).fillna(""):
        for theme in [x.strip() for x in str(val).split(";") if x.strip()]:
            counter[theme] += 1
    return pd.DataFrame(counter.most_common(), columns=["Theme","Actions"])

def top_tfidf_phrases(df, n=25):
    docs = (
        df.get("Mechanism", pd.Series(dtype=str)).fillna("").astype(str)
        + " "
        + df.get("Analytical Summary", pd.Series(dtype=str)).fillna("").astype(str)
        + " "
        + df.get("Evidence Quotes", pd.Series(dtype=str)).fillna("").astype(str)
    )
    docs = docs[docs.str.len() > 20]
    if len(docs) < 2:
        return pd.DataFrame(columns=["Phrase","Importance"])
    vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.90,
        max_features=2500
    )
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["Phrase","Importance"])
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.asarray(vec.get_feature_names_out())
    idx = scores.argsort()[::-1][:n]
    return pd.DataFrame({"Phrase":terms[idx], "Importance":scores[idx]})
