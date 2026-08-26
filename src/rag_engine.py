import json, re, requests
from bs4 import BeautifulSoup

def safe(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s

def fetch_url_text(url, timeout=35):
    """Retrieve readable Congressional information from the GovTrack URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script","style","nav","footer","header","noscript","svg","form"]):
        tag.decompose()
    text = "\n".join(
        re.sub(r"\s+", " ", x).strip()
        for x in soup.get_text("\n").splitlines()
        if x.strip()
    )
    if len(text) < 350:
        raise ValueError("Too little readable text returned by GovTrack")
    return text

def chunks(t, size=1900, overlap=250):
    out=[]; start=0
    while start < len(t):
        end=min(start+size, len(t))
        p=t[start:end].strip()
        if len(p) >= 150:
            out.append(p)
        if end >= len(t):
            break
        start=max(0, end-overlap)
    return out

def build_document_store(df, m, selected_source="govtrack", progress_callback=None):
    docs=[]; log=[]; checked=loaded=errors=0; total=len(df)
    configs={
        "govtrack":("govtrack_col","GovTrack Bill Page"),
        "congress":("congress_col","Congress.gov Full Bill Text"),
    }
    column_key,source_name=configs.get(selected_source,configs["govtrack"])
    source_col=m.get(column_key)
    for pos,(_,row) in enumerate(df.iterrows(),1):
        bill=safe(row.get(m.get("bill_col"))) if m.get("bill_col") else ""
        title=safe(row.get(m.get("title_col"))) if m.get("title_col") else ""
        if progress_callback: progress_callback(pos-1,total,f"Reading {bill} from {source_name}")
        url=safe(row.get(source_col)) if source_col else ""
        cc=[]; source_urls=[]
        if url:
            checked+=1
            try:
                text=fetch_url_text(url); loaded+=1; source_urls.append(url)
                for p in chunks(text): cc.append({"source":source_name,"url":url,"text":p})
                log.append({"Bill":bill,"Source":source_name,"Result":"Loaded","URL":url})
            except Exception as ex:
                errors+=1
                log.append({"Bill":bill,"Source":source_name,"Result":"Source Retrieval Failed","Detail":str(ex)[:180],"URL":url})
        else:
            errors+=1
            log.append({"Bill":bill,"Source":source_name,"Result":"Missing source URL","Detail":"Selected source column is empty.","URL":""})
        if cc:
            docs.append({
                "bill_number":bill,"title":title,
                "status":safe(row.get(m.get("status_col"))) if m.get("status_col") else "",
                "country":safe(row.get(m.get("country_col"))) if m.get("country_col") else "",
                "bill_type":safe(row.get(m.get("type_col"))) if m.get("type_col") else "",
                "congress":safe(row.get(m.get("congress_session_col"))) if m.get("congress_session_col") else "",
                "evidence_source":source_name,"source_urls":source_urls,"chunks":cc
            })
        if progress_callback: progress_callback(pos,total,f"Finished {bill}")
    return docs,{"bills_scanned":total,"links_checked":checked,"sources_loaded":loaded,
                 "source_errors":errors,"documents_created":len(docs),"selected_source":source_name},log

def candidates(doc,n=14):
    terms=["transformer","8504","electrical","grid","transmission","substation","china","chinese","prc",
           "mexico","usmca","tariff","import","forced labor","sanction","investment","cfius","ownership",
           "manufacturing","procurement","buy america"]
    a=[]
    for c in doc["chunks"]:
        a.append((sum(x in c["text"].lower() for x in terms),c))
    return [c for _,c in sorted(a,key=lambda z:z[0],reverse=True)[:n]]

SYSTEM="""You perform qualitative Congressional content analysis for DAQO's strategy for serving the U.S. market.
DAQO is a Chinese electrical-equipment manufacturer; main study product is transformers, HTS 8504. The dataset supplies Country: NEVER infer or replace it.
Scenarios are CHINA-EXPORT, MEXICO-MANUFACTURE, US-MANUFACTURE.
FIRST classify RELEVANT YES/NO. Relevance means meaningful evidence of existing/emerging risk/opportunity or Congressional sentiment/political pressure affecting DAQO's strategic environment. Enactment, binding force, passage, direct DAQO naming, direct transformer naming and immediate economic effect are NOT required. Non-binding and unsuccessful actions may be relevant. Political criticism alone with no meaningful economic/trade/investment/manufacturing/supply-chain/market connection is not relevant.
If relevant assign EXACTLY ONE principal classification:
COUNTRY RISK, COUNTRY OPPORTUNITY, INDUSTRY RISK, INDUSTRY OPPORTUNITY, INVESTMENT RISK, INVESTMENT OPPORTUNITY.
COUNTRY covers broader country trade/economic/supply-chain/geopolitical treatment; COUNTRY OPPORTUNITY includes favorable trade/USMCA/North-American location conditions.
INDUSTRY is ONLY direct transformer/8504, substations, switchgear, transmission/distribution, grid infrastructure, electrical equipment or directly relevant inputs/market. Grid buildout can be INDUSTRY OPPORTUNITY. 5G, EVs/batteries, semiconductors, pharma, aerospace and unrelated military equipment are NOT automatically DAQO industry, though broader country restrictions can be COUNTRY RISK.
INVESTMENT covers establishment, ownership, acquisition, financing or expansion of manufacturing; CFIUS/Chinese ownership restrictions can be risk, and nearshoring/manufacturing/FDI incentives can be opportunity.
Record policy_stage separately: Proposed, Passed one chamber, Passed Congress, Enacted, Implemented / Enforced, Non-binding resolution, Other.
Record directness: DIRECT, EMERGING, SENTIMENT / POLITICAL PRESSURE.
Select only meaningfully affected scenarios.
Use substantive/operative evidence; for non-binding actions substantive resolution/call language may support sentiment. Do not rely on title alone.
For every action write a concise 2-5 sentence analytical summary: what it does, why keyword screening may flag it, relevance/non-relevance, classification/mechanism, affected scenario. For relevant actions explain Congressional Action -> Policy/Economic Mechanism -> DAQO Risk/Opportunity -> Affected Scenario.
Return JSON keys: relevant, primary_classification, policy_stage, directness, affected_scenarios, mechanism, evidence (list of quote/source/why_it_matters objects), analytical_summary, confidence."""

def run_classification(documents,api_key,progress_callback=None):
    from openai import OpenAI
    client=OpenAI(api_key=api_key)
    out=[]; total=len(documents)

    for pos,d in enumerate(documents,1):
        if progress_callback:
            progress_callback(pos-1,total,f"Classifying {d['bill_number']}")

        ev="\n\n".join(
            f"[{i}] Source={c['source']}\n{c['text']}"
            for i,c in enumerate(candidates(d),1)
        )
        user=f"""Bill: {d['bill_number']}
Title: {d['title']}
Type: {d['bill_type']}
Status: {d['status']}
Country supplied by dataset: {d['country']}
Congress: {d['congress']}

DOCUMENT PASSAGES
{ev}

Return the required JSON classification."""

        try:
            r=client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=.05,
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":SYSTEM},
                    {"role":"user","content":user}
                ]
            )
            a=json.loads(r.choices[0].message.content)
        except Exception as e:
            a={
                "analysis_status":"FAILED",
                "error_type":type(e).__name__,
                "error_message":str(e),
                "relevant":"",
                "primary_classification":"",
                "policy_stage":"",
                "directness":"",
                "affected_scenarios":[],
                "mechanism":"",
                "evidence":[],
                "analytical_summary":"Analysis failed before a qualitative conclusion could be produced.",
                "confidence":""
            }
        else:
            a["analysis_status"]="SUCCESS"
            a["error_type"]=""
            a["error_message"]=""
            if str(a.get("relevant","NO")).upper() != "YES":
                a["relevant"]="NO"
                a["primary_classification"]="NOT RELEVANT"
                a["affected_scenarios"]=[]

        a.update({
            "bill_number":d["bill_number"],
            "title":d["title"],
            "country":d["country"],
            "source_urls":d.get("source_urls",[]),
            "evidence_source":d.get("evidence_source","")
        })
        out.append(a)

        if progress_callback:
            progress_callback(pos,total,f"Finished {d['bill_number']}")

    return out
