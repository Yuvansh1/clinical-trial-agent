# CLAUDE.md — Clinical Trial Patient Matching Agent

This file helps Claude Code understand the project structure, conventions, and workflows.

---

## Project Overview

An agentic AI pipeline that matches patients to clinical trials using:
- **LangGraph** for multi-step agent orchestration
- **ChromaDB + sentence-transformers** for RAG (semantic search over trial criteria)
- **Rule-based eligibility screener** for age, lab values, exclusion checks
- **MLflow** for experiment tracking
- **FastAPI** REST API + **Streamlit** dashboard

---

## Project Structure

```
clinical-trial-agent/
├── main.py                          # FastAPI app — entry point for API
├── streamlit_app.py                 # Streamlit dashboard
├── src/
│   ├── agents/
│   │   └── matching_agent.py        # ★ LangGraph 8-node agent (main logic)
│   ├── models/
│   │   └── eligibility_screener.py  # Rule-based patient-trial scoring
│   └── utils/
│       ├── data_generator.py        # Synthetic trials + patients
│       ├── vector_store.py          # ChromaDB wrapper
│       └── mlflow_tracker.py        # MLflow logging helpers
├── tests/
│   └── test_matching.py             # pytest test suite
├── data/                            # Auto-generated CSVs + ChromaDB (git-ignored)
├── mlruns/                          # MLflow runs (git-ignored)
├── CLAUDE.md                        # This file
├── .env.example                     # Environment variable template
└── requirements.txt
```

---

## How to Run

```bash
# Install
uv venv --python 3.11
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# API
uvicorn main:app --reload --port 8000

# Dashboard
streamlit run streamlit_app.py

# MLflow UI
mlflow ui --port 5000

# Tests
pytest tests/ -v
```

---

## Environment Variables

```
GEMINI_API_KEY=...     # Optional — enables LLM reasoning. Falls back to rule-based if absent.
```

---

## Key Conventions

- **State management**: LangGraph `AgentState` holds only JSON-serializable primitives. Non-serializable objects (DataFrames, numpy arrays) live in `_CACHE` dict keyed by `thread_id`.
- **numpy types**: Always wrap values going into state with `_to_python()` to avoid msgpack errors.
- **MLflow**: Every agent run auto-logs to `./mlruns`. View with `mlflow ui`.
- **ChromaDB**: Persisted at `./data/chroma_db`. Delete this folder to force re-indexing.
- **Tests**: Use `thread_id` like `"test-run-1"`, `"test-run-2"` etc. to avoid cache collisions between tests.

---

## Agent Node Map

| Node | File | Purpose |
|---|---|---|
| `load_data` | `matching_agent.py` | Load trials CSV + build patient query |
| `index_trials` | `matching_agent.py` → `vector_store.py` | Embed trials into ChromaDB |
| `retrieve_candidates` | `matching_agent.py` → `vector_store.py` | Semantic search top-3 trials |
| `screen_eligibility` | `matching_agent.py` → `eligibility_screener.py` | Score each candidate |
| `generate_reasoning` | `matching_agent.py` | LLM or rule-based narrative |
| `human_review` | `matching_agent.py` | Approval gate |
| `log_to_mlflow` | `matching_agent.py` → `mlflow_tracker.py` | Log run |
| `finalize_report` | `matching_agent.py` | Assemble output |

---

## Common Tasks for Claude Code

**Add a new trial to the dataset:**
→ Edit `TRIAL_TEMPLATES` list in `src/utils/data_generator.py`

**Add a new eligibility check (e.g. BMI):**
→ Edit `score_patient_trial()` in `src/models/eligibility_screener.py`

**Add a new agent node:**
→ Define function in `matching_agent.py`, register with `workflow.add_node()`, add edge

**Add a new API endpoint:**
→ Add route to `main.py`, add Pydantic model if needed

**Add a new test:**
→ Add to `tests/test_matching.py`, follow existing fixture pattern

---

## Do Not

- Do not store DataFrames or numpy arrays directly in `AgentState`
- Do not use `shuffle=True` in any data splits (time-series awareness)
- Do not hardcode API keys — always use `.env` + `python-dotenv`
- Do not delete `data/.gitkeep` — it keeps the data directory tracked in git
