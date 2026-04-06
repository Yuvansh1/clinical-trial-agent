# 📖 Project Explainer — Clinical Trial Patient Matching Agent
### What this project is, how it works, and why each piece exists

---

## 🧠 The Problem in Simple Terms

Imagine a cancer patient whose doctor thinks they might qualify for a new drug trial. To check, someone has to manually read through hundreds of clinical trials, each with 10-20 complex medical eligibility rules, and figure out which ones the patient qualifies for. This takes hours and is often skipped entirely — which is why only 5% of patients ever enroll in trials.

This project automates that process using AI.

---

## 🗂️ What the Project Does — Step by Step

When you run the agent for a patient, here is exactly what happens:

**Step 1 — Load Data**
The system loads a database of 6 clinical trials (each with real inclusion/exclusion criteria) and finds the patient's profile (age, diagnosis, lab results, medications, comorbidities).

**Step 2 — Index Trials**
Each trial's eligibility criteria is converted into a mathematical representation called an embedding using a model called `sentence-transformers`. These embeddings are stored in ChromaDB — a vector database — so they can be searched by meaning, not just keywords.

**Step 3 — Retrieve Candidates**
The patient's profile is also converted into an embedding and compared against all trial embeddings. The top 3 most similar trials are retrieved. This is called RAG (Retrieval Augmented Generation).

**Step 4 — Screen Eligibility**
Each candidate trial is checked against the patient using rules:
- Is the patient the right age?
- Do their lab values (HbA1c, LVEF, MMSE etc.) fall within the required range?
- Do they have any hard exclusion criteria (pregnant, HIV, prior organ transplant)?

Each trial gets a confidence score from 0 to 1.

**Step 5 — Generate Reasoning**
Gemini (Google's LLM) reads the screening results and writes a plain-English clinical summary explaining why each trial is or isn't a good match — like a doctor would explain it.

**Step 6 — Human Approval**
A human (Clinical Research Coordinator or physician) reviews and approves the results before they're finalized. This is called human-in-the-loop and is critical in healthcare AI.

**Step 7 — Log to MLflow**
Every single run is recorded in MLflow — an experiment tracking tool. It logs what patient was matched, how many trials were found, the confidence scores, and the full reasoning. This creates an audit trail.

**Step 8 — Final Report**
Everything is assembled into a structured report that can be returned via API or shown in the dashboard.

---

## 🧩 What Each Technology Does

| Technology | What it is | Why it's used here |
|---|---|---|
| **LangGraph** | Framework for building multi-step AI agents | Orchestrates all 8 steps in order, manages state, enables human-in-the-loop |
| **ChromaDB** | Vector database | Stores trial embeddings for fast semantic search |
| **sentence-transformers** | NLP embedding model | Converts text into numbers that capture meaning |
| **MLflow** | Experiment tracking tool | Logs every run for auditability and comparison |
| **FastAPI** | Python web framework | Exposes the agent as a REST API |
| **Streamlit** | Python dashboard framework | Provides a visual UI to run and view results |
| **Gemini** | Google's LLM | Generates clinical reasoning in natural language |
| **Docker** | Containerization | Packages the whole app so anyone can run it anywhere |
| **GitHub Actions** | CI/CD automation | Runs tests and builds Docker image automatically on every code push |

---

## 🤖 What is LangGraph and Why Does it Matter?

LangGraph is a library for building **agentic AI** — AI that doesn't just answer one question but takes multiple steps autonomously to complete a task.

Think of it like this:

- **Normal AI (ChatGPT style):** You ask a question → it answers → done
- **Agentic AI (LangGraph style):** You give a goal → it plans → executes step 1 → uses the result in step 2 → decides what to do in step 3 → loops or branches if needed → delivers a final result

In this project the agent:
1. Decides which trials to retrieve (based on semantic similarity)
2. Decides which trials pass eligibility (based on rules + labs)
3. Decides what to write in the reasoning (based on all prior results)
4. Waits for human approval before finalizing

This is real agentic behavior — not just one LLM call.

---

## 📊 What is MLflow and Why Does it Matter?

MLflow is like a logbook for every AI/ML experiment you run.

Every time the agent matches a patient, MLflow records:
- **What went in:** patient ID, how many trials were searched
- **What came out:** how many matches, confidence scores, match rate
- **The full reasoning:** saved as a text file you can read later
- **Metadata:** which version of the agent ran, which embedding model was used

In pharma, this is critical because every clinical decision needs to be auditable — you need to prove what the AI did and why.

You can see all runs visually by running `mlflow ui` and opening `http://localhost:5000`.

---

## ⚙️ What is CI/CD and Why Does it Matter?

CI/CD stands for Continuous Integration / Continuous Deployment.

**CI (ci.yml) — runs every time you push code:**
1. Checks code style (flake8) — no messy code
2. Runs all tests (pytest) — nothing is broken
3. Scans for security issues (bandit) — no vulnerabilities

If any of these fail, the pipeline stops and tells you what broke.

**CD (cd.yml) — runs after CI passes on main branch:**
1. Builds your app into a Docker image
2. Pushes it to GitHub Container Registry (ghcr.io)
3. Anyone can now run your app with: `docker pull ghcr.io/yuvansh1/clinical-trial-agent:latest`

**PR checks (pr.yml) — runs on every pull request:**
1. Auto-labels the PR (e.g. "agent", "tests", "api") based on which files changed
2. Posts a test coverage table as a comment on the PR

This is the same workflow used at real pharma tech companies.

---

## 🗄️ What is RAG?

RAG stands for Retrieval Augmented Generation. It is a technique where instead of asking an LLM to answer from memory, you first retrieve relevant documents and then ask the LLM to reason over them.

In this project:
1. **Retrieval** — ChromaDB finds the most relevant trials for the patient
2. **Augmented** — the patient profile + trial criteria are combined into a prompt
3. **Generation** — Gemini generates clinical reasoning based on that specific context

This is better than just asking Gemini "what trials does this patient qualify for?" because Gemini would hallucinate trials that don't exist. RAG grounds the answer in your actual trial database.

---

## 🏥 Why This is Relevant to Pharma

Every component maps to a real pharma/healthcare use case:

- **Trial matching** → Clinical Operations, Site Management
- **Eligibility screening** → Regulatory Affairs, Protocol Compliance
- **MLflow audit trail** → GxP compliance, 21 CFR Part 11
- **Human-in-the-loop** → Physician oversight requirement in AI-assisted clinical decisions
- **Docker + CI/CD** → Validated software deployment in regulated environments

---

## 💬 How to Explain This in an Interview

**Simple version (30 seconds):**
> "I built an AI agent that automatically matches patients to clinical trials. It uses semantic search to find relevant trials from a database, scores each one against the patient's lab values and medical history, then uses an LLM to explain the matches in clinical language — all tracked in MLflow with a human approval step before finalizing."

**Technical version (2 minutes):**
> "The core is a LangGraph agent with 8 nodes. It starts by embedding clinical trial eligibility criteria into ChromaDB using sentence-transformers. When a patient comes in, their profile is embedded and a cosine similarity search retrieves the top candidate trials. Each candidate goes through a rule-based screener that checks age ranges, lab values like HbA1c or LVEF, and hard exclusion criteria — producing a 0-1 confidence score. The Gemini LLM then generates clinical reasoning over the screened results. Every run is logged to MLflow with full params, metrics, and artifacts. The whole thing is exposed via FastAPI, containerized with Docker, and deployed via GitHub Actions CI/CD to GitHub Container Registry."
