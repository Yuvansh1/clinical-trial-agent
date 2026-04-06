# 🧬 Clinical Trial Patient Matching Agent

[![CI](https://github.com/Yuvansh1/clinical-trial-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuvansh1/clinical-trial-agent/actions/workflows/ci.yml)
[![CD](https://github.com/Yuvansh1/clinical-trial-agent/actions/workflows/cd.yml/badge.svg)](https://github.com/Yuvansh1/clinical-trial-agent/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-green)](https://github.com/langchain-ai/langgraph)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/Yuvansh1/clinical-trial-agent/pkgs/container/clinical-trial-agent)

An **agentic AI system** that matches patients to clinical trials using:
- 🤖 **LangGraph** — 8-node stateful agent orchestration with human-in-the-loop
- 🔍 **RAG (ChromaDB + sentence-transformers)** — semantic search over trial eligibility criteria
- 📐 **Rule-based eligibility screener** — age, lab values, hard exclusions with confidence scoring
- 💬 **Gemini LLM** — clinical reasoning in plain English (falls back to rule-based without API key)
- 📊 **MLflow** — full experiment tracking of every matching run
- 🚀 **FastAPI + Streamlit** — REST API + interactive dashboard
- 🐳 **Docker + GHCR** — containerized and published to GitHub Container Registry via CD pipeline

---

## 🧠 The Problem This Solves

Only ~5% of cancer patients enroll in clinical trials. A key barrier is the manual, time-consuming process of matching patients to trials based on complex eligibility criteria. This system automates that pipeline end-to-end using agentic AI.

---

## 🏗️ Architecture

```
clinical-trial-agent/
├── main.py                              # FastAPI REST API
├── streamlit_app.py                     # Interactive Streamlit dashboard
├── src/
│   ├── agents/
│   │   └── matching_agent.py            # ★ LangGraph 8-node agent (core logic)
│   ├── models/
│   │   └── eligibility_screener.py      # Rule-based eligibility scoring
│   └── utils/
│       ├── data_generator.py            # Synthetic trials + patient data
│       ├── vector_store.py              # ChromaDB embedding + semantic search
│       └── mlflow_tracker.py            # MLflow experiment logging
├── tests/
│   └── test_matching.py                 # 20+ unit + integration tests
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                       # Lint + test + security scan
│   │   ├── cd.yml                       # Build + push Docker image to GHCR
│   │   └── pr.yml                       # Auto-label + coverage comment on PRs
│   └── labeler.yml                      # PR label rules
├── CLAUDE.md                            # Claude Code project context
├── CONTRIBUTING.md                      # Contribution guide
├── Dockerfile
└── requirements.txt
```

---

## 🤖 LangGraph Agent — 8-Node Pipeline

```
START
  │
  ▼
[load_data]              Load trials CSV + build patient profile query
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
- `StateGraph` + `TypedDict` — clean serializable state across all nodes
- `MemorySaver` checkpointer — runs resumable by `thread_id`
- Human-in-the-loop gate — CRC/physician approval before finalizing
- `_OBJECT_CACHE` pattern — non-serializable objects (DataFrames, numpy) stored outside state

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
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Optional: add Gemini API key for LLM reasoning
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here

# Terminal 1 — API
uvicorn main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run streamlit_app.py

# Terminal 3 — MLflow UI
mlflow ui --port 5000
```

---

## 🐳 Run with Docker

```bash
# Pull latest image from GHCR
docker pull ghcr.io/yuvansh1/clinical-trial-agent:latest

# Run
docker run -p 8000:8000 ghcr.io/yuvansh1/clinical-trial-agent:latest

# API docs available at:
# http://localhost:8000/docs
```

---

## 🔌 API Usage

**Match a patient to trials:**
```bash
# PowerShell
Invoke-RestMethod -Method POST -Uri http://localhost:8000/match `
  -ContentType "application/json" `
  -Body '{"patient_id": "PAT1000"}'

# curl
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "PAT1000"}'
```

**View recent MLflow runs:**
```bash
Invoke-RestMethod http://localhost:8000/runs
```

**Swagger UI:** http://localhost:8000/docs

---

## 📊 MLflow Tracking

Every agent run automatically logs:

| Category | What's Logged |
|---|---|
| **Params** | patient_id, trials searched, candidates retrieved |
| **Metrics** | similarity scores, match rate, avg confidence score |
| **Artifacts** | patient profile, candidate trials, eligibility results, LLM reasoning |
| **Tags** | agent version, embedding model name |

```bash
mlflow ui --port 5000
# Open http://localhost:5000 to view all runs
```

---

## ⚙️ CI/CD Pipeline

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push + PR | Lint (flake8), test (pytest + coverage), security scan (bandit) |
| `cd.yml` | Push to main (after CI passes) | Build Docker image, push to `ghcr.io` |
| `pr.yml` | Every PR | Auto-label by changed files, post coverage comment |

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src
```

Covers: data generation, age/lab extraction, exclusion screening,
confidence scoring, vector store indexing/search, and full agent pipeline.

---

## 🔮 Roadmap

- [ ] Real CTGOV API integration (clinicaltrials.gov)
- [ ] HL7 FHIR patient record ingestion
- [ ] Multi-patient batch processing
- [ ] Slack notification for human-in-the-loop approval
- [ ] Redis checkpointer for production state persistence

---

## 📄 License

MIT
