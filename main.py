"""
Clinical Trial Patient Matching API
"""

import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.agents.matching_agent import run_agent
from src.utils.mlflow_tracker import get_recent_runs

app = FastAPI(
    title="Clinical Trial Patient Matching API",
    description="LangGraph Agentic AI + RAG + MLflow for pharma clinical trial matching",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_results: dict = {}


class MatchRequest(BaseModel):
    patient_id: str = "PAT1000"
    thread_id: Optional[str] = None


@app.get("/")
def root():
    return {
        "app": "Clinical Trial Matching API",
        "docs": "/docs",
        "endpoints": ["/match", "/result/{thread_id}", "/runs", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match_patient(req: MatchRequest):
    """
    Run the full LangGraph matching agent for a patient.
    Returns matched trials with confidence scores and clinical reasoning.
    """
    thread_id = req.thread_id or f"ct-{uuid.uuid4().hex[:8]}"
    report = run_agent(patient_id=req.patient_id, thread_id=thread_id)
    _results[thread_id] = report
    return {"thread_id": thread_id, "status": report.get("status"), "report": report}


@app.get("/result/{thread_id}")
def get_result(thread_id: str):
    result = _results.get(thread_id)
    if not result:
        raise HTTPException(404, f"No result for thread '{thread_id}'.")
    return result


@app.get("/runs")
def recent_runs(n: int = 10):
    """Fetch recent matching runs from MLflow."""
    return {"runs": get_recent_runs(n=n)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
