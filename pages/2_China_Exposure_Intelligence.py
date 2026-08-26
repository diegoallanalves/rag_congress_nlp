
from pathlib import Path
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.nlp_dashboard import load_run, build_consensus, theme_counts, top_tfidf_phrases

st.set_page_config(page_title="DAQO China Exposure Intelligence", page_icon="📡", layout="wide")

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"

def discover_default_files():
    files = sorted(DATA_DIR.glob("*.xlsx")) if DATA_DIR.exists() else []
    preferred = [p for p in files if "analysis" in p.name.lower() or "rag" in p.name.lower()]
    files = preferred if preferred else files
    return files[:2]

DEFAULT_FILES = discover_default_files()
DEFAULT_1 = DEFAULT_FILES[0] if len(DEFAULT_FILES) > 0 else None
DEFAULT_2 = DEFAULT_FILES[1] if len(DEFAULT_FILES) > 1 else None

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
div[data-testid="stMetric"] {background: linear-gradient(135deg,#0f2740,#173b5d); border:1px solid #315878; padding:16px; border-radius:14px;}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {color:white;}
.insight-card {padding:14px 16px;border:1px solid #d9e2ec;border-radius:12px;background:#f8fafc;margin-bottom:10px;}
.risk-high {border-left:5px solid #dc2626;}
.risk-elevated {border-left:5px solid #f59e0b;}
.risk-watch {border-left:5px solid #3b82f6;}
</style>
""", unsafe_allow_html=True)

st.title("📡 DAQO — China Strategic Exposure Intelligence")
st.caption("NLP trend analysis built on the evidence and classifications created by the Congressional RAG pipeline.")

with st.expander("🧠 How this dashboard works", expanded=False):
    st.markdown("""
This page **does not re-run the OpenAI classification**. It analyses the RAG output locally using transparent NLP rules and TF‑IDF phrase extraction.

**China Exclusion Index (0–100)** is driven by observable language in mechanisms, summaries and evidence:
procurement exclusion, tariffs/import controls, sanctions/entity restrictions, ownership/investment controls, and domestic supply-chain/reshoring signals.

The score is a research screening indicator — not a prediction that a bill will become law.
""")

st.subheader("📂 Analysis data")
mode = st.radio("Data input", ["Use packaged RAG exports", "Upload two RAG exports"], horizontal=True)

if mode == "Upload two RAG exports":
    c1,c2 = st.columns(2)
    up1 = c1.file_uploader("RAG analysis — Run 1", type=["xlsx"], key="run1")
    up2 = c2.file_uploader("RAG analysis — Run 2", type=["xlsx"], key="run2")
    if not up1 or not up2:
        st.info("Upload both DAQO_RAG_Analysis Excel files to continue.")
        st.stop()
    p1,p2 = up1,up2
else:
    if DEFAULT_1 is None or DEFAULT_2 is None:
        st.error("Two analysis Excel files were not found in data/.")
        st.info("Add two DAQO RAG analysis .xlsx files to the data folder, or choose Upload two RAG exports.")
        st.stop()
    p1,p2 = DEFAULT_1,DEFAULT_2
    st.caption(f"Run 1: {DEFAULT_1.name}")
    st.caption(f"Run 2: {DEFAULT_2.name}")

@st.cache_data(show_spinner=False)
def prepare(a,b):
    r1 = load_run(a, "Run 1")
    r2 = load_run(b, "Run 2")
    cons = build_consensus(r1,r2)
    return r1,r2,cons

with st.spinner("Building the NLP intelligence layer..."):
    run1,run2,df = prepare(p1,p2)

# Filters
st.subheader("🎛️ Intelligence filters")
fc1,fc2,fc3,fc4 = st.columns(4)

congresses = sorted([x for x in df["Congress"].dropna().unique() if x != "Unknown"])
sel_congress = fc1.multiselect("Congress", congresses, default=congresses)

classes = sorted([x for x in df["Primary Classification"].dropna().unique() if x and x != "NOT RELEVANT"])
sel_classes = fc2.multiselect("Classification", classes, default=classes)

bands = ["High","Elevated","Watch","Low"]
sel_bands = fc3.multiselect("Exclusion signal", bands, default=bands)

agreement_opts = ["Full agreement","Relevance agreement","Disagreement"]
sel_agreement = fc4.multiselect("Run agreement", agreement_opts, default=agreement_opts)

filtered = df.copy()
if sel_congress:
    filtered = filtered[filtered["Congress"].isin(sel_congress)]
if sel_classes:
    # Keep relevant rows matching selected classes plus NOT RELEVANT only if user has no class selection.
    filtered = filtered[(filtered["Primary Classification"].isin(sel_classes)) | (filtered["Relevant"] != "YES")]
if sel_bands:
    filtered = filtered[filtered["Exclusion Band"].isin(sel_bands)]
if sel_agreement:
    filtered = filtered[filtered["Run Agreement"].isin(sel_agreement)]

relevant = filtered[filtered["Relevant"]=="YES"].copy()

# KPIs
avg_index = relevant["China Exclusion Index"].mean() if len(relevant) else 0
high_share = (relevant["China Exclusion Index"]>=70).mean()*100 if len(relevant) else 0
agreement_share = (filtered["Run Agreement"]=="Full agreement").mean()*100 if len(filtered) else 0

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Actions in view", f"{len(filtered):,}")
k2.metric("Relevant actions", f"{len(relevant):,}")
k3.metric("China Exclusion Index", f"{avg_index:.1f}/100")
k4.metric("High exclusion signals", f"{high_share:.1f}%")
k5.metric("Two-run agreement", f"{agreement_share:.1f}%")

if avg_index >= 70:
    st.error("🔴 **High strategic exclusion signal:** the filtered evidence contains broad and repeated restrictive-China mechanisms.")
elif avg_index >= 45:
    st.warning("🟠 **Elevated strategic exclusion signal:** restrictive-China language is material and should be monitored by mechanism and Congress.")
elif avg_index >= 20:
    st.info("🔵 **Watch signal:** some exclusion mechanisms are present, but the evidence is mixed.")
else:
    st.success("🟢 **Low aggregate exclusion signal** in the current filtered view.")

left,right = st.columns([1.6,1])

with left:
    st.subheader("📈 China exclusion trend by Congress")
    if len(relevant):
        trend = relevant.groupby("Congress",as_index=False).agg(
            Exclusion_Index=("China Exclusion Index","mean"),
            Relevant_Actions=("Bill","count"),
            High_Signals=("China Exclusion Index",lambda x:(x>=70).sum())
        )
        order={f"{i}th":i for i in range(100,130)}
        trend["order"]=trend["Congress"].map(order).fillna(999)
        trend=trend.sort_values("order")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=trend["Congress"],y=trend["Exclusion_Index"],
                                 mode="lines+markers",name="Exclusion Index",line={"width":4}))
        fig.add_trace(go.Bar(x=trend["Congress"],y=trend["Relevant_Actions"],
                             name="Relevant actions",opacity=.28,yaxis="y2"))
        fig.update_layout(
            height=390, margin=dict(l=20,r=20,t=30,b=20),
            yaxis=dict(title="Exclusion Index",range=[0,100]),
            yaxis2=dict(title="Relevant actions",overlaying="y",side="right",showgrid=False),
            legend=dict(orientation="h",y=1.12)
        )
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("No relevant actions in the current filter.")

    st.subheader("🏛️ Risk / opportunity mix over time")
    if len(relevant):
        mix = relevant.groupby(["Congress","Primary Classification"]).size().reset_index(name="Actions")
        fig=px.bar(mix,x="Congress",y="Actions",color="Primary Classification",barmode="stack",
                   color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(height=390,margin=dict(l=20,r=20,t=20,b=20),legend_title_text="")
        st.plotly_chart(fig,use_container_width=True)

with right:
    st.subheader("🧭 Evidence explorer")
    choices = relevant.sort_values(["China Exclusion Index","Consensus Confidence"],ascending=False)
    if choices.empty:
        st.info("No relevant evidence to show for the current filters.")
    else:
        labels = choices.apply(
            lambda r:f"{r['Bill']} | {r['China Exclusion Index']:.0f}/100 | {str(r['Title'])[:55]}",axis=1
        ).tolist()
        chosen_label = st.selectbox("Choose an action",labels)
        row = choices.iloc[labels.index(chosen_label)]

        band = row["Exclusion Band"].lower()
        css = "risk-high" if band=="high" else "risk-elevated" if band=="elevated" else "risk-watch"
        st.markdown(
            f"""<div class="insight-card {css}">
            <b>{row['Bill']} — {row['Title']}</b><br>
            Congress: {row['Congress']}<br>
            Classification: {row['Primary Classification']}<br>
            Exclusion Index: <b>{row['China Exclusion Index']:.0f}/100</b><br>
            Consensus confidence: {row['Consensus Confidence']:.0f}%<br>
            Run agreement: {row['Run Agreement']}
            </div>""", unsafe_allow_html=True
        )
        st.write("**Mechanism**")
        st.write(row.get("Mechanism",""))
        st.write("**Why the RAG considered it relevant**")
        st.write(row.get("Analytical Summary",""))
        st.write("**NLP themes detected**")
        st.write(row.get("NLP Themes","") or "No explicit exclusion theme matched.")

        quotes = [x.strip() for x in str(row.get("Evidence Quotes","")).split("||") if x.strip()]
        whys = [x.strip() for x in str(row.get("Evidence Why","")).split("||") if x.strip()]
        st.write("**Supporting evidence**")
        for i,q in enumerate(quotes[:5]):
            why = whys[i] if i < len(whys) else ""
            st.info(f"Evidence {i+1}: {q}\n\nWhy it matters: {why}")

        url = str(row.get("Source URL",""))
        if url and url!="nan":
            st.link_button("🔗 Open Congressional source",url)

st.divider()
c1,c2 = st.columns(2)

with c1:
    st.subheader("🔥 Restrictive-China policy themes")
    themes = theme_counts(relevant)
    if not themes.empty:
        fig=px.bar(themes.sort_values("Actions"),x="Actions",y="Theme",orientation="h",
                   color="Actions",color_continuous_scale="Turbo")
        fig.update_layout(height=420,margin=dict(l=10,r=20,t=10,b=20),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)

with c2:
    st.subheader("☁️ NLP phrase landscape")
    phrases=top_tfidf_phrases(relevant,25)
    if not phrases.empty:
        # Treemap gives a dashboard-like "topic cloud" while remaining readable.
        phrases["Group"]="Top evidence phrases"
        fig=px.treemap(phrases,path=["Group","Phrase"],values="Importance",
                       color="Importance",color_continuous_scale="Blues")
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=10,b=10),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)

st.divider()
c3,c4 = st.columns(2)

with c3:
    st.subheader("🤝 Two-run consistency")
    agree = filtered["Run Agreement"].value_counts().reset_index()
    agree.columns=["Agreement","Actions"]
    fig=px.pie(agree,names="Agreement",values="Actions",hole=.55,
               color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=350,margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig,use_container_width=True)

with c4:
    st.subheader("🎯 Exclusion signal distribution")
    bands_df = relevant["Exclusion Band"].value_counts().reindex(["High","Elevated","Watch","Low"],fill_value=0).reset_index()
    bands_df.columns=["Signal","Actions"]
    fig=px.bar(bands_df,x="Signal",y="Actions",color="Signal",
               color_discrete_map={"High":"#dc2626","Elevated":"#f59e0b","Watch":"#3b82f6","Low":"#16a34a"})
    fig.update_layout(height=350,margin=dict(l=10,r=10,t=10,b=10),showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

st.subheader("📋 Evidence-backed intelligence table")
table_cols=["Bill","Title","Congress","Relevant","Primary Classification","China Exclusion Index",
            "Exclusion Band","Consensus Confidence","Run Agreement","NLP Themes","Source URL"]
st.dataframe(
    filtered[[c for c in table_cols if c in filtered.columns]]
        .sort_values("China Exclusion Index",ascending=False),
    use_container_width=True,hide_index=True
)

# Export dashboard intelligence
export_df = filtered.copy()
buf=io.BytesIO()
with pd.ExcelWriter(buf,engine="openpyxl") as writer:
    export_df.to_excel(writer,index=False,sheet_name="China Exposure Intelligence")
    theme_counts(relevant).to_excel(writer,index=False,sheet_name="Policy Themes")
    top_tfidf_phrases(relevant,50).to_excel(writer,index=False,sheet_name="NLP Phrases")

st.download_button(
    "⬇️ Export NLP intelligence to Excel",
    data=buf.getvalue(),
    file_name="DAQO_China_Exposure_Intelligence.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
