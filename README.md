# DAQO China Strategic Exposure Intelligence — V10 Compatible

This NLP application is designed to consume the Excel exports produced by the simplified DAQO Congressional RAG V10 workflow.

## Data input

The app accepts **one or many V10 RAG Excel exports** and merges them automatically. Each workbook must contain at least:

- `All Results`
- `Evidence`

The standard V10 export also contains `Relevant`, `Not Relevant`, `Failed Analysis`, and `Methodology`.

## Main workflow

1. Upload one or more V10 RAG exports.
2. Validate the workbook structure.
3. Merge actions and evidence.
4. Filter by Congress and Country.
5. Run evidence-first NLP.
6. Explore trends, themes, evidence phrases, and Congressional evidence.
7. Optionally generate an OpenAI evidence brief.
8. Export the merged NLP intelligence workbook.

## NLP focus

The NLP layer primarily analyzes the **Evidence Quote** and **Why It Matters** text. Duplicate actions and duplicate evidence extracts are consolidated. Failed analysis remains separate from Not Relevant.

Visuals include:

- China exclusion pressure by Congress
- risk/opportunity mix by country
- restrictive-China policy mechanisms
- policy mechanism heatmap
- TF-IDF evidence language landscape
- exclusion signal distribution
- evidence explorer

## OpenAI

For local use, create `.env` from `.env.example`:

```text
OPENAI_API_KEY=your_key_here
```

For Streamlit Cloud, add `OPENAI_API_KEY` under app Secrets.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

`.env` is ignored and must never be committed.
