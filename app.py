
from pathlib import Path
import io
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from src.nlp_intelligence import (
    congress_number, evidence_context, split_evidence,
    theme_by_congress, theme_counts, top_evidence_phrases
)

load_dotenv()
BASE=Path(__file__).resolve().parent
DATA_PATH=BASE/"data"/"merged_evidence_consensus.xlsx"

st.set_page_config(page_title="DAQO China Evidence Intelligence",page_icon="📡",layout="wide")

st.markdown("""
<style>
.block-container{max-width:1550px;padding-top:1.1rem;padding-bottom:3rem}
[data-testid="stMetric"]{border:1px solid rgba(49,88,120,.30);padding:14px;border-radius:14px;background:rgba(30,55,80,.04)}
.evidence-card{padding:14px;border-radius:12px;border:1px solid #d7dee7;margin-bottom:10px;background:#fafcff}
.small-note{font-size:.88rem;color:#64748b}
</style>
""",unsafe_allow_html=True)

st.title("📡 DAQO — China Evidence Intelligence")
st.caption("Evidence-first NLP analysis of merged Congressional RAG results, with optional OpenAI evidence synthesis.")

@st.cache_data(show_spinner=False)
def load_data(path):
    df=pd.read_excel(path, sheet_name="Merged Evidence")
    df["China Exclusion Index"]=pd.to_numeric(df["China Exclusion Index"],errors="coerce").fillna(0)
    df["Consensus Confidence"]=pd.to_numeric(df["Consensus Confidence"],errors="coerce").fillna(0)
    df["Evidence Count"]=pd.to_numeric(df["Evidence Count"],errors="coerce").fillna(0)
    return df

if not DATA_PATH.exists():
    st.error("Missing data/merged_evidence_consensus.xlsx")
    st.stop()

df=load_data(DATA_PATH)

with st.expander("ℹ️ What this page measures",expanded=False):
    st.markdown("""
The **China Exclusion Index** is a transparent evidence-screening indicator, not a probability that a bill will pass.
It increases when the evidence contains mechanisms such as procurement exclusion, tariffs/import controls,
sanctions/entity restrictions, ownership/investment controls, or domestic supply-chain replacement language.

The app focuses on **evidence quotes + why-it-matters text**, while keeping the two original RAG runs visible through
the consensus and disagreement fields.
""")

# ---------------- Filters ----------------
st.subheader("🎛️ Filters")
f1,f2,f3,f4=st.columns(4)
congresses=sorted([x for x in df["Congress"].dropna().unique() if str(x)!="Unknown"],key=congress_number)
sel_congress=f1.multiselect("Congress",congresses,default=congresses)

classes=sorted([x for x in df["Consensus Classification"].dropna().unique() if str(x) not in ("","NOT RELEVANT","DISAGREEMENT")])
sel_classes=f2.multiselect("Classification",classes,default=classes)

sel_band=f3.multiselect("Exclusion signal",["High","Elevated","Watch","Low"],default=["High","Elevated","Watch","Low"])
sel_agree=f4.multiselect("RAG run agreement",["Full agreement","Relevance agreement","Disagreement"],default=["Full agreement","Relevance agreement","Disagreement"])

view=df.copy()
if sel_congress: view=view[view["Congress"].isin(sel_congress)]
if sel_band: view=view[view["Exclusion Band"].isin(sel_band)]
if sel_agree: view=view[view["Run Agreement"].isin(sel_agree)]
if sel_classes:
    view=view[(view["Consensus Classification"].isin(sel_classes)) | (view["Relevant"].astype(str).str.upper()!="YES")]

relevant=view[view["Relevant"].astype(str).str.upper()=="YES"].copy()

# ---------------- KPIs ----------------
avg_idx=relevant["China Exclusion Index"].mean() if len(relevant) else 0
high=(relevant["China Exclusion Index"]>=70).sum()
full_agree=(view["Run Agreement"]=="Full agreement").mean()*100 if len(view) else 0
evidence_total=int(relevant["Evidence Count"].sum()) if len(relevant) else 0

k1,k2,k3,k4,k5=st.columns(5)
k1.metric("Actions in view",f"{len(view):,}")
k2.metric("Relevant actions",f"{len(relevant):,}")
k3.metric("Evidence extracts",f"{evidence_total:,}")
k4.metric("China Exclusion Index",f"{avg_idx:.1f}/100")
k5.metric("Full RAG agreement",f"{full_agree:.1f}%")

if avg_idx>=70:
    st.error("🔴 High evidence-based exclusion pressure in the current filtered view.")
elif avg_idx>=45:
    st.warning("🟠 Elevated evidence-based exclusion pressure in the current filtered view.")
elif avg_idx>=20:
    st.info("🔵 Mixed / watch-level exclusion pressure in the current filtered view.")
else:
    st.success("🟢 Limited exclusion pressure in the current filtered view.")

# ---------------- Main row: Trend + evidence ----------------
left,right=st.columns([1.55,1])

with left:
    st.subheader("📈 Is exclusion pressure increasing?")
    if len(relevant):
        trend=relevant.groupby("Congress",as_index=False).agg(
            Exclusion_Index=("China Exclusion Index","mean"),
            Relevant_Actions=("Bill","count"),
            Evidence=("Evidence Count","sum"),
            High_Signals=("China Exclusion Index",lambda s:int((s>=70).sum()))
        )
        trend["Order"]=trend["Congress"].apply(congress_number)
        trend=trend.sort_values("Order")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=trend["Congress"],y=trend["Exclusion_Index"],mode="lines+markers",name="Exclusion Index"))
        fig.add_trace(go.Bar(x=trend["Congress"],y=trend["Relevant_Actions"],name="Relevant actions",opacity=.28,yaxis="y2"))
        fig.update_layout(
            height=390,margin=dict(l=10,r=10,t=25,b=10),
            yaxis=dict(title="Exclusion Index",range=[0,100]),
            yaxis2=dict(title="Relevant actions",overlaying="y",side="right",showgrid=False),
            legend=dict(orientation="h",y=1.13)
        )
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("🧩 Policy mechanisms by Congress")
    heat=theme_by_congress(relevant)
    if not heat.empty:
        pivot=heat.pivot(index="Theme",columns="Congress",values="Actions").fillna(0)
        ordered=[c for c in sorted(pivot.columns,key=congress_number)]
        pivot=pivot[ordered]
        fig=px.imshow(pivot,aspect="auto",text_auto=True,labels=dict(x="Congress",y="Policy mechanism",color="Actions"))
        fig.update_layout(height=390,margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)

with right:
    st.subheader("🧭 Evidence explorer")
    choices=relevant.sort_values(["China Exclusion Index","Consensus Confidence","Evidence Count"],ascending=False)
    if choices.empty:
        st.info("No relevant evidence in the current filters.")
    else:
        labels=choices.apply(lambda r:f"{r['Bill']} | {r['Congress']} | {r['China Exclusion Index']:.0f}/100 | {str(r['Title'])[:42]}",axis=1).tolist()
        selected=st.selectbox("Choose a Congressional action",labels)
        row=choices.iloc[labels.index(selected)]
        st.markdown(
            f"""<div class="evidence-card">
            <b>{row['Bill']} — {row['Title']}</b><br>
            Congress: {row['Congress']}<br>
            Classification: {row['Consensus Classification']}<br>
            Exclusion Index: <b>{row['China Exclusion Index']:.0f}/100</b><br>
            Consensus confidence: {row['Consensus Confidence']:.0f}%<br>
            RAG agreement: {row['Run Agreement']}<br>
            Evidence extracts: {int(row['Evidence Count'])}
            </div>""",
            unsafe_allow_html=True
        )
        st.write("**Detected NLP themes**")
        st.write(row.get("NLP Themes","") or "No restrictive-China theme matched.")

        quotes=split_evidence(row.get("Evidence Quotes",""))
        whys=split_evidence(row.get("Evidence Why It Matters",""))
        for i,q in enumerate(quotes[:6]):
            why=whys[i] if i<len(whys) else ""
            st.info(f"Evidence {i+1}: {q}\n\nWhy it matters: {why}")

        url=str(row.get("Source URL",""))
        if url and url.lower()!="nan":
            st.link_button("🔗 Open source",url)

# ---------------- Visual row ----------------
st.divider()
c1,c2=st.columns(2)
with c1:
    st.subheader("🔥 Strongest exclusion mechanisms")
    tc=theme_counts(relevant)
    if not tc.empty:
        fig=px.bar(tc.sort_values("Actions"),x="Actions",y="Theme",orientation="h",color="Actions")
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=10,b=10),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)

with c2:
    st.subheader("☁️ Evidence language landscape")
    phrases=top_evidence_phrases(relevant,30)
    if not phrases.empty:
        phrases["Evidence"]="Evidence phrases"
        fig=px.treemap(phrases,path=["Evidence","Phrase"],values="Importance",color="Importance")
        fig.update_layout(height=420,margin=dict(l=5,r=5,t=5,b=5),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)

st.divider()
c3,c4=st.columns(2)
with c3:
    st.subheader("🎯 Risk / opportunity mix")
    if len(relevant):
        mix=relevant["Consensus Classification"].value_counts().reset_index()
        mix.columns=["Classification","Actions"]
        fig=px.pie(mix,names="Classification",values="Actions",hole=.55)
        fig.update_layout(height=360,margin=dict(l=5,r=5,t=5,b=5))
        st.plotly_chart(fig,use_container_width=True)

with c4:
    st.subheader("🔍 Confidence vs exclusion pressure")
    if len(relevant):
        fig=px.scatter(
            relevant,x="Consensus Confidence",y="China Exclusion Index",
            size="Evidence Count",color="Consensus Classification",
            hover_name="Bill",hover_data=["Congress","Title","NLP Themes"],
            size_max=42
        )
        fig.update_layout(height=360,margin=dict(l=5,r=5,t=5,b=5))
        st.plotly_chart(fig,use_container_width=True)

# ---------------- AI evidence synthesis ----------------
st.divider()
st.subheader("✨ OpenAI Evidence Brief")
key=os.getenv("OPENAI_API_KEY","").strip()
if not key:
    try:
        key=str(st.secrets.get("OPENAI_API_KEY","")).strip()
    except Exception:
        key=""

ai1,ai2=st.columns([1,1])
with ai1:
    model=st.selectbox("Model",["gpt-5.4-mini","gpt-5.4-nano"],index=0)
with ai2:
    max_actions=st.slider("Evidence-rich actions sent to AI",5,20,12)

question=st.text_area(
    "Ask an evidence-based question",
    value="Based only on the selected Congressional evidence, is the U.S. policy direction increasingly excluding or reducing dependence on China, and what does this mean for DAQO's China-export, Mexico-manufacture and U.S.-manufacture strategies?",
    height=100
)

if not key:
    st.caption("Add OPENAI_API_KEY to .env locally or Streamlit Secrets online to enable AI synthesis.")

if st.button("🧠 Generate evidence brief",type="primary",disabled=not bool(key)):
    from openai import OpenAI
    context=evidence_context(relevant,max_actions=max_actions)
    if not context:
        st.warning("No evidence is available under the current filters.")
    else:
        prompt=f"""You are a policy research analyst. Answer the question using ONLY the evidence supplied below.

QUESTION:
{question}

RULES:
- Do not invent facts outside the supplied evidence.
- Distinguish observed evidence from interpretation.
- Cite bill numbers in every major finding.
- Focus on trends, mechanisms, and implications for DAQO.
- Discuss China exclusion/restriction, Mexico-manufacture implications, and U.S.-manufacture implications when supported.
- Explicitly mention uncertainty and disagreement where relevant.
- Use a short executive summary, 4-7 key findings, strategic implications, and a final evidence-strength assessment.

EVIDENCE:
{context}
"""
        try:
            client=OpenAI(api_key=key)
            response=client.responses.create(model=model,input=prompt,max_output_tokens=1400)
            st.markdown(response.output_text)
        except Exception as exc:
            st.error(f"OpenAI analysis failed: {exc}")

# ---------------- Table + export ----------------
st.divider()
st.subheader("📋 Merged evidence intelligence")
cols=[
    "Bill","Title","Congress","Relevant","Consensus Classification",
    "China Exclusion Index","Exclusion Band","Consensus Confidence","Run Agreement",
    "Evidence Count","NLP Themes","Matched Evidence Terms","Source URL"
]
st.dataframe(
    view[[c for c in cols if c in view.columns]].sort_values(
        ["China Exclusion Index","Consensus Confidence"],ascending=False
    ),
    use_container_width=True,hide_index=True
)

st.download_button(
    "⬇️ Download filtered merged evidence (CSV)",
    data=view.to_csv(index=False).encode("utf-8-sig"),
    file_name="DAQO_merged_evidence_intelligence.csv",
    mime="text/csv"
)
