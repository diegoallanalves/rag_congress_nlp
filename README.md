# DAQO China Evidence Intelligence v10

This version merges the two DAQO RAG runs into one evidence-first consensus dataset.

## One data source
`data/merged_evidence_consensus.xlsx`

The two original run files are no longer required at runtime.

## Main features
- Evidence-first NLP analysis
- China Exclusion Index
- Trend chart by Congress
- Policy-mechanism heatmap
- Risk/opportunity mix
- Evidence phrase treemap
- Confidence vs exclusion-pressure bubble chart
- Evidence explorer beside the visualisations
- RAG run agreement and consensus confidence
- Optional OpenAI Evidence Brief using only the filtered evidence

## OpenAI
Local:
1. Copy `.env.example` to `.env`
2. Add `OPENAI_API_KEY=...`

Streamlit Cloud:
Add `OPENAI_API_KEY` under app Secrets.

The app does not send the entire dataset to OpenAI. It sends only the top evidence-rich actions in the current filtered view.

## Run
```powershell
pip install -r requirements.txt
streamlit run app.py
```
