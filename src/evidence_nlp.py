from __future__ import annotations

from collections import Counter
from io import BytesIO
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


REQUIRED_SHEETS = {"All Results", "Evidence"}

THEMES = {
    "Procurement / Market Exclusion": {
        "weight": 25,
        "terms": [
            "prohibition on procurement", "prohibit procurement", "procurement ban",
            "shall not procure", "may not procure", "restriction on procurement",
            "exclude", "excluded", "chinese origin", "prc origin",
            "people's republic of china-origin", "government procurement",
        ],
    },
    "Trade / Tariffs / Import Controls": {
        "weight": 20,
        "terms": [
            "tariff", "tariffs", "duty", "duties", "import restriction",
            "restrict imports", "importation", "trade restriction",
            "normal trade relations", "most-favored-nation", "most favored nation",
            "customs", "export control", "export controls",
        ],
    },
    "Sanctions / Entity Restrictions": {
        "weight": 20,
        "terms": [
            "sanction", "sanctions", "entity list", "restricted entity",
            "chinese entity", "prc entity", "blocked person", "designation",
            "economic sanctions",
        ],
    },
    "Ownership / Investment Controls": {
        "weight": 20,
        "terms": [
            "cfius", "foreign investment", "chinese-owned", "prc-owned",
            "foreign ownership", "ownership restriction", "investment restriction",
            "covered transaction", "acquisition", "national security review",
        ],
    },
    "Domestic Supply Chain / Reshoring": {
        "weight": 15,
        "terms": [
            "buy america", "buy american", "domestic content", "domestic manufacturing",
            "reshoring", "onshoring", "supply chain security", "supply chain resilience",
            "reduce dependence", "reduce reliance", "critical supply chain",
            "made in america", "domestic source", "trusted supplier",
        ],
    },
    "Forced Labor / Import Compliance": {
        "weight": 18,
        "terms": [
            "forced labor", "forced labour", "uyghur", "xinjiang",
            "uflpa", "withhold release order", "import compliance",
        ],
    },
}

OPPORTUNITY_TERMS = [
    "nearshoring", "usmca", "duty-free", "duty free",
    "manufacturing incentive", "tax credit", "grant", "financing",
    "grid investment", "infrastructure investment", "mexico manufacturing",
]


def safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def congress_from_url(url) -> str:
    text = safe_text(url)
    patterns = [
        r"/bills/(\d{3})/",
        r"/congress/bills/(\d{3})/",
        r"/(\d{3})/(?:s|hr|hjres|hres|sres|sjres)/",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            n = int(match.group(1))
            suffix = "th"
            if n % 100 not in (11, 12, 13):
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            return f"{n}{suffix}"
    return "Unknown"


def congress_from_filename(name) -> str:
    match = re.search(r"(?<!\d)(1\d{2})(?!\d)", safe_text(name))
    if not match:
        return "Unknown"
    n = int(match.group(1))
    suffix = "th"
    if n % 100 not in (11, 12, 13):
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def congress_number(value) -> int:
    match = re.search(r"(\d+)", safe_text(value))
    return int(match.group(1)) if match else 999


def confidence_number(value) -> float:
    text = safe_text(value).upper()
    return {"HIGH": 0.95, "MEDIUM": 0.65, "LOW": 0.35}.get(text, 0.50)


def split_evidence(value) -> list[str]:
    text = safe_text(value)
    if not text:
        return []
    return [safe_text(x) for x in text.split("||") if safe_text(x)]


def join_unique(values) -> str:
    seen = set()
    output = []
    for value in values:
        for part in split_evidence(value):
            key = part.lower()
            if key not in seen:
                seen.add(key)
                output.append(part)
    return " || ".join(output)


def _read_excel_blob(blob: bytes):
    return pd.ExcelFile(BytesIO(blob))


def inspect_export(name: str, blob: bytes) -> dict:
    try:
        xls = _read_excel_blob(blob)
        missing = sorted(REQUIRED_SHEETS.difference(xls.sheet_names))
        if missing:
            return {
                "File": name,
                "Status": "INVALID",
                "Actions": 0,
                "Evidence": 0,
                "Congress": congress_from_filename(name),
                "Message": "Missing sheets: " + ", ".join(missing),
            }

        actions = pd.read_excel(BytesIO(blob), sheet_name="All Results")
        evidence = pd.read_excel(BytesIO(blob), sheet_name="Evidence")
        congresses = set()
        if "Source URL" in actions.columns:
            congresses.update(
                congress_from_url(x)
                for x in actions["Source URL"].dropna().tolist()
            )
        congresses.discard("Unknown")
        if not congresses:
            congresses = {congress_from_filename(name)}
        return {
            "File": name,
            "Status": "READY",
            "Actions": len(actions),
            "Evidence": len(evidence),
            "Congress": ", ".join(sorted(congresses, key=congress_number)),
            "Message": "V10 structure detected",
        }
    except Exception as exc:
        return {
            "File": name,
            "Status": "ERROR",
            "Actions": 0,
            "Evidence": 0,
            "Congress": "Unknown",
            "Message": str(exc),
        }


def load_v10_export(name: str, blob: bytes):
    xls = _read_excel_blob(blob)
    missing = REQUIRED_SHEETS.difference(xls.sheet_names)
    if missing:
        raise ValueError(f"{name}: missing required sheets {sorted(missing)}")

    actions = pd.read_excel(BytesIO(blob), sheet_name="All Results")
    evidence = pd.read_excel(BytesIO(blob), sheet_name="Evidence")
    actions.columns = [safe_text(c) for c in actions.columns]
    evidence.columns = [safe_text(c) for c in evidence.columns]

    actions["Source File"] = name
    evidence["Source File"] = name

    file_congress = congress_from_filename(name)

    if "Source URL" not in actions.columns:
        actions["Source URL"] = ""
    if "Source URL" not in evidence.columns:
        evidence["Source URL"] = ""

    actions["Congress"] = actions["Source URL"].apply(congress_from_url)
    evidence["Congress"] = evidence["Source URL"].apply(congress_from_url)

    actions.loc[actions["Congress"] == "Unknown", "Congress"] = file_congress
    evidence.loc[evidence["Congress"] == "Unknown", "Congress"] = file_congress

    if "Country" not in actions.columns:
        actions["Country"] = ""

    lookup = (
        actions[["Bill", "Congress", "Country"]]
        .dropna(subset=["Bill"])
        .drop_duplicates()
    )
    evidence = evidence.merge(lookup, on=["Bill", "Congress"], how="left")

    return actions, evidence


def merge_v10_exports(sources: list[tuple[str, bytes]]):
    action_parts = []
    evidence_parts = []
    quality = []

    for name, blob in sources:
        status = inspect_export(name, blob)
        quality.append(status)
        if status["Status"] != "READY":
            continue
        actions, evidence = load_v10_export(name, blob)
        action_parts.append(actions)
        evidence_parts.append(evidence)

    if not action_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(quality)

    actions = pd.concat(action_parts, ignore_index=True, sort=False)
    evidence = (
        pd.concat(evidence_parts, ignore_index=True, sort=False)
        if evidence_parts else pd.DataFrame()
    )

    for col in [
        "Analysis Status", "Bill", "Title", "Country", "Relevant",
        "Primary Classification", "Policy Stage", "Directness",
        "Affected Scenarios", "Mechanism", "Analytical Summary",
        "Confidence", "Evidence Source", "Source URL",
        "Error Type", "Error Message", "Congress", "Source File",
    ]:
        if col not in actions.columns:
            actions[col] = ""

    actions["Research Key"] = (
        actions["Congress"].fillna("").astype(str).str.strip()
        + " | "
        + actions["Country"].fillna("").astype(str).str.strip()
        + " | "
        + actions["Bill"].fillna("").astype(str).str.strip()
    )

    actions["_success_rank"] = (
        actions["Analysis Status"].fillna("").astype(str).str.upper() == "SUCCESS"
    ).astype(int)
    actions["_relevant_rank"] = (
        actions["Relevant"].fillna("").astype(str).str.upper() == "YES"
    ).astype(int)
    actions = (
        actions.sort_values(
            ["Research Key", "_success_rank", "_relevant_rank"],
            ascending=[True, False, False],
        )
        .drop_duplicates("Research Key", keep="first")
        .drop(columns=["_success_rank", "_relevant_rank"])
        .reset_index(drop=True)
    )

    if evidence.empty:
        evidence = pd.DataFrame(
            columns=[
                "Bill", "Congress", "Country", "Quote", "Why It Matters",
                "Source URL", "Source File",
            ]
        )

    for col in [
        "Bill", "Congress", "Country", "Quote", "Why It Matters",
        "Source URL", "Source File",
    ]:
        if col not in evidence.columns:
            evidence[col] = ""

    evidence["Research Key"] = (
        evidence["Congress"].fillna("").astype(str).str.strip()
        + " | "
        + evidence["Country"].fillna("").astype(str).str.strip()
        + " | "
        + evidence["Bill"].fillna("").astype(str).str.strip()
    )

    evidence["_quote_key"] = evidence["Quote"].fillna("").astype(str).str.strip().str.lower()
    evidence = (
        evidence.drop_duplicates(["Research Key", "_quote_key"], keep="first")
        .drop(columns=["_quote_key"])
        .reset_index(drop=True)
    )

    grouped = (
        evidence.groupby("Research Key", dropna=False)
        .agg(
            **{
                "Evidence Quotes": ("Quote", lambda s: join_unique(s.tolist())),
                "Evidence Why It Matters": (
                    "Why It Matters",
                    lambda s: join_unique(s.tolist()),
                ),
                "Evidence Count": (
                    "Quote",
                    lambda s: sum(1 for x in s if safe_text(x)),
                ),
            }
        )
        .reset_index()
    )

    actions = actions.merge(grouped, on="Research Key", how="left")
    actions["Evidence Quotes"] = actions["Evidence Quotes"].fillna("")
    actions["Evidence Why It Matters"] = actions["Evidence Why It Matters"].fillna("")
    actions["Evidence Count"] = (
        pd.to_numeric(actions["Evidence Count"], errors="coerce").fillna(0).astype(int)
    )

    return actions, evidence, pd.DataFrame(quality)


def score_action(row):
    evidence_text = (
        safe_text(row.get("Evidence Quotes"))
        + " "
        + safe_text(row.get("Evidence Why It Matters"))
    ).lower()
    fallback_text = (
        safe_text(row.get("Mechanism"))
        + " "
        + safe_text(row.get("Analytical Summary"))
    ).lower()
    text = evidence_text if evidence_text.strip() else fallback_text

    themes = []
    terms = []
    score = 0

    for theme, cfg in THEMES.items():
        matched = [term for term in cfg["terms"] if term in text]
        if matched:
            themes.append(theme)
            terms.extend(matched)
            score += cfg["weight"]

    if safe_text(row.get("Relevant")).upper() == "YES":
        score += 5
    if "RISK" in safe_text(row.get("Primary Classification")).upper():
        score += 5
    if safe_text(row.get("Directness")).upper() == "DIRECT":
        score += 4

    score -= min(10, sum(2 for term in OPPORTUNITY_TERMS if term in text))
    score = max(0, min(100, score))

    if score >= 70:
        band = "High"
    elif score >= 45:
        band = "Elevated"
    elif score >= 20:
        band = "Watch"
    else:
        band = "Low"

    return score, band, "; ".join(themes), "; ".join(dict.fromkeys(terms))


def build_intelligence(actions: pd.DataFrame):
    if actions.empty:
        return actions.copy()

    out = actions.copy()
    scores = []
    bands = []
    themes = []
    terms = []

    for _, row in out.iterrows():
        score, band, theme_text, term_text = score_action(row)
        scores.append(score)
        bands.append(band)
        themes.append(theme_text)
        terms.append(term_text)

    out["China Exclusion Index"] = scores
    out["Exclusion Band"] = bands
    out["NLP Themes"] = themes
    out["Matched Evidence Terms"] = terms
    out["Confidence Numeric"] = out["Confidence"].apply(confidence_number)
    out["Evidence Strength"] = (
        (out["Evidence Count"].clip(upper=5) / 5 * 60)
        + (out["Confidence Numeric"] * 40)
    ).round(0)

    return out


def theme_counts(df):
    counter = Counter()
    for value in df.get("NLP Themes", pd.Series(dtype=str)).fillna(""):
        for theme in [x.strip() for x in safe_text(value).split(";") if x.strip()]:
            counter[theme] += 1
    return pd.DataFrame(counter.most_common(), columns=["Theme", "Actions"])


def theme_by_congress(df):
    rows = []
    for _, row in df.iterrows():
        for theme in [
            x.strip()
            for x in safe_text(row.get("NLP Themes")).split(";")
            if x.strip()
        ]:
            rows.append({"Congress": row.get("Congress", "Unknown"), "Theme": theme})
    if not rows:
        return pd.DataFrame(columns=["Congress", "Theme", "Actions"])
    return (
        pd.DataFrame(rows)
        .groupby(["Congress", "Theme"])
        .size()
        .reset_index(name="Actions")
    )


def top_evidence_phrases(df, n=30):
    docs = (
        df.get("Evidence Quotes", pd.Series(dtype=str)).fillna("").astype(str)
        + " "
        + df.get("Evidence Why It Matters", pd.Series(dtype=str)).fillna("").astype(str)
    )
    docs = docs[docs.str.len() > 20]
    if len(docs) < 2:
        return pd.DataFrame(columns=["Phrase", "Importance"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        max_features=4000,
    )
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["Phrase", "Importance"])

    importance = np.asarray(matrix.mean(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    idx = importance.argsort()[::-1][:n]
    return pd.DataFrame(
        {"Phrase": terms[idx], "Importance": importance[idx]}
    )


def evidence_context(df, max_actions=12, max_quote_chars=800):
    if df.empty:
        return ""

    ranked = df.sort_values(
        ["China Exclusion Index", "Evidence Strength", "Evidence Count"],
        ascending=False,
    ).head(max_actions)

    blocks = []
    for _, row in ranked.iterrows():
        quotes = split_evidence(row.get("Evidence Quotes"))
        why = split_evidence(row.get("Evidence Why It Matters"))

        blocks.append(
            "\n".join(
                [
                    f"BILL: {row.get('Bill', '')} ({row.get('Congress', '')})",
                    f"COUNTRY: {row.get('Country', '')}",
                    f"TITLE: {row.get('Title', '')}",
                    f"CLASSIFICATION: {row.get('Primary Classification', '')}",
                    f"EXCLUSION INDEX: {row.get('China Exclusion Index', 0)}",
                    f"THEMES: {row.get('NLP Themes', '')}",
                    f"EVIDENCE: {' | '.join(quotes[:2])[:max_quote_chars]}",
                    f"WHY IT MATTERS: {' | '.join(why[:2])[:max_quote_chars]}",
                    f"SOURCE: {row.get('Source URL', '')}",
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)
