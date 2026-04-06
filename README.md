# 🧬 Clinical Trial Patient Matching Agent

[![CI](https://github.com/Yuvansh1/clinical-trial-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuvansh1/clinical-trial-agent/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-green)](https://github.com/langchain-ai/langgraph)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)](https://mlflow.org)

An **agentic AI system** that matches patients to clinical trials using:
- 🤖 **LangGraph** — 8-node stateful agent orchestration
- 🔍 **RAG (ChromaDB + sentence-transformers)** — semantic search over trial eligibility criteria
- 📐 **Rule-based eligibility screener** — age, lab values, hard exclusions
- 💬 **Gemini LLM** — clinical reasoning in plain English
- 📊 **MLflow** — full experiment tracking of every matching run
- 🚀 **FastAPI + Streamlit** — REST API + interactive dashboard

---

## 🧠 The Problem This Solves

Only ~5% of cancer patients enroll in clinical trials. A key barrier is the manual, time-consuming process of matching patients to trials based on complex eligibility criteria. This system automates that pipeline end-to-end using agentic AI.

---

## 🏗️ Architecture

```
clinical-trial-agent/
├── main.py                              # FastAPI REST API
├── streamlit_app.py                     # Interactive dashboard
├── src/
│   ├── utils/
│   │   ├── data_generator.py            # Synthetic trials + patient data
│   │   ├── vector_store.py              # ChromaDB embedding + semantic search
│   │   └── mlflow_tracker.py            # MLflow experiment logging
│   ├── models/
│   │   └── eligibility_screener.py      # Rule-based eligibility scoring
│   └── agents/
│       └── matching_agent.py            # ★ LangGraph 8-node agent
├── tests/
│   └── test_matching.py                 # 20 unit + integration tests
├── .github/workflows/ci.yml
├── Dockerfile
└── requirements.txt
```

---

## 🤖 LangGraph Agent — 8-Node Pipeline

```
START
  │
  ▼
[load_data]              Load trials CSV + patient profile
  │
  ▼
[index_trials]           Embed trial criteria → ChromaDB vector store
  │
  ▼
[retrieve_candidates]    Semantic search: top-3 trials for patient
  │
  ▼
[screen_eligibility]     Rule-based screen: age, labs, hard exclusions
                         → confidence score per trial (0–1)
  │
  ▼
[generate_reasoning]     Gemini LLM: clinical narrative for each match
  │
  ▼
[human_review]           ← Human-in-the-loop (CRC/physician sign-off)
  │
  ▼
[log_to_mlflow]          Log params, metrics, artifacts to MLflow
  │
  ▼
[finalize_report]        Assemble final patient-trial match report
  │
  ▼
 END
```

**LangGraph features used:**
- `StateGraph` + `TypedDict` state — clean shared state across all nodes
- `MemorySaver` checkpointer — runs resumable by `thread_id`
- Human-in-the-loop gate — CRC approval before finalizing
- Error isolation — each node handles its own exceptions

---

## 📊 Therapeutic Areas Covered

| Area | Trial | Key Criteria |
|---|---|---|
| Oncology | NSCLC (anti-PD1) | ECOG 0-2, no prior immunotherapy |
| Cardiology | Heart failure | LVEF < 40%, NYHA II-III |
| Neurology | Alzheimer's | MMSE 20-26, amyloid PET+ |
| Diabetes | Type 2 DM | HbA1c 7.5-10.5%, BMI 25-45 |
| Rare Disease | Gaucher type 1 | Enzyme confirmed, ERT naive |
| Immunology | Rheumatoid arthritis | DAS28 > 3.2, MTX inadequate |

---

## 🚀 Quickstart

```bash
git clone https://github.com/Yuvansh1/clinical-trial-agent.git
cd clinical-trial-agent

# Install (uv recommended)
uv venv --python 3.11
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Optional: Gemini for LLM reasoning
cp .env.example .env
# Add GEMINI_API_KEY to .env

# Run API
uvicorn main:app --reload --port 8000

# Run dashboard (separate terminal)
streamlit run streamlit_app.py

# Run MLflow UI (separate terminal)
mlflow ui --port 5000
```

---

## 🔌 API Usage

**Match a patient:**
```bash
# PowerShell
Invoke-RestMethod -Method POST -Uri http://localhost:8000/match `
  -ContentType "application/json" `
  -Body '{"patient_id": "PAT1000"}'
```

**View recent MLflow runs:**
```bash
Invoke-RestMethod http://localhost:8000/runs
```

**Swagger UI:** http://localhost:8000/docs

---

## 📊 MLflow Tracking

Every matching run logs:

| Category | What's Logged |
|---|---|
| **Params** | patient_id, trials searched, candidates retrieved |
| **Metrics** | similarity scores, match rate, avg confidence |
| **Artifacts** | patient profile, candidate trials, eligibility results, reasoning |
| **Tags** | agent version, embedding model |

View all runs: `mlflow ui` → http://localhost:5000

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src
```

Covers: data generation, age/lab extraction, exclusion screening, confidence scoring, vector store, and full agent pipeline.

---

## 🔮 Roadmap

- [ ] Real CTGOV API integration (clinicaltrials.gov)
- [ ] HL7 FHIR patient record ingestion
- [ ] Multi-patient batch processing
- [ ] Slack notification for human approval step
- [ ] Redis checkpointer for production state persistence

---

## 📄 License

MIT
