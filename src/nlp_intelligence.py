
from collections import Counter
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def split_evidence(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return [re.sub(r"\s+"," ",x).strip() for x in str(text).split("||") if str(x).strip() and str(x).strip().lower()!="nan"]

def congress_number(value):
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999

def theme_counts(df):
    counter = Counter()
    for value in df.get("NLP Themes", pd.Series(dtype=str)).fillna(""):
        for theme in [x.strip() for x in str(value).split(";") if x.strip()]:
            counter[theme] += 1
    return pd.DataFrame(counter.most_common(), columns=["Theme","Actions"])

def theme_by_congress(df):
    rows=[]
    for _,r in df.iterrows():
        for theme in [x.strip() for x in str(r.get("NLP Themes","")).split(";") if x.strip()]:
            rows.append({"Congress":r.get("Congress","Unknown"),"Theme":theme})
    if not rows:
        return pd.DataFrame(columns=["Congress","Theme","Actions"])
    x=pd.DataFrame(rows).groupby(["Congress","Theme"]).size().reset_index(name="Actions")
    return x

def top_evidence_phrases(df, n=30):
    docs = (
        df.get("Evidence Quotes", pd.Series(dtype=str)).fillna("").astype(str)
        + " "
        + df.get("Evidence Why It Matters", pd.Series(dtype=str)).fillna("").astype(str)
    )
    docs=docs[docs.str.len()>20]
    if len(docs)<2:
        return pd.DataFrame(columns=["Phrase","Importance"])
    vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_df=.92,max_features=4000)
    try:
        X=vec.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["Phrase","Importance"])
    importance=np.asarray(X.mean(axis=0)).ravel()
    terms=np.asarray(vec.get_feature_names_out())
    idx=importance.argsort()[::-1][:n]
    return pd.DataFrame({"Phrase":terms[idx],"Importance":importance[idx]})

def evidence_context(df, max_actions=12, max_quote_chars=750):
    if df.empty:
        return ""
    ranked=df.sort_values(["China Exclusion Index","Consensus Confidence","Evidence Count"],ascending=False).head(max_actions)
    blocks=[]
    for _,r in ranked.iterrows():
        quotes=split_evidence(r.get("Evidence Quotes",""))
        why=split_evidence(r.get("Evidence Why It Matters",""))
        quote=" | ".join(quotes[:2])[:max_quote_chars]
        why_text=" | ".join(why[:2])[:max_quote_chars]
        blocks.append(
            f"BILL: {r.get('Bill','')} ({r.get('Congress','')})\n"
            f"TITLE: {r.get('Title','')}\n"
            f"CLASSIFICATION: {r.get('Consensus Classification','')}\n"
            f"EXCLUSION INDEX: {r.get('China Exclusion Index',0)}\n"
            f"THEMES: {r.get('NLP Themes','')}\n"
            f"EVIDENCE: {quote}\n"
            f"WHY IT MATTERS: {why_text}\n"
            f"SOURCE: {r.get('Source URL','')}"
        )
    return "\n\n---\n\n".join(blocks)
