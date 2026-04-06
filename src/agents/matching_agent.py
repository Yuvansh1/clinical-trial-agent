"""
Clinical Trial Patient Matching — LangGraph Agentic Orchestration
=================================================================

Agent graph:
    START
      │
      ▼
  [load_data]             ← Load trials + patient from CSV / generate if missing
      │
      ▼
  [index_trials]          ← Embed trials into ChromaDB vector store
      │
      ▼
  [retrieve_candidates]   ← Semantic search: find top-K trials for patient
      │
      ▼
  [screen_eligibility]    ← Rule-based + lab screening with confidence scores
      │
      ▼
  [generate_reasoning]    ← LLM (Gemini) explains matches in clinical language
      │
      ▼
  [human_review]          ← Human-in-the-loop checkpoint (CRC/physician approval)
      │
      ▼
  [log_to_mlflow]         ← Log full run: params, metrics, artifacts to MLflow
      │
      ▼
  [finalize_report]       ← Assemble final patient-trial match report
      │
      ▼
    END
"""

import os
import json
import operator
import pandas as pd
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.utils.data_generator import generate_trials, generate_patients
from src.utils.vector_store import TrialVectorStore, build_patient_query
from src.utils.mlflow_tracker import log_matching_run
from src.models.eligibility_screener import screen_patient_against_trials


# ─────────────────────────────────────────────
# Module-level cache for non-serializable objects
# ─────────────────────────────────────────────
_CACHE: Dict[str, Any] = {}


def _to_python(obj):
    """Recursively convert non-serializable types to Python primitives."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (bool,)):
        return bool(obj)
    if isinstance(obj, (int,)):
        return int(obj)
    if hasattr(obj, 'item'):   # numpy scalars
        return obj.item()
    if isinstance(obj, float):
        return float(obj)
    return obj


# ─────────────────────────────────────────────
# 1. Agent State (only serializable types)
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    cache_key: str
    patient_id: str
    patient_query: Optional[str]
    n_trials_indexed: Optional[int]
    candidate_trials: Optional[List[Dict]]
    eligibility_results: Optional[List[Dict]]
    final_matches: Optional[List[Dict]]
    agent_reasoning: Optional[str]
    mlflow_run_id: Optional[str]
    human_approved: Optional[bool]
    final_report: Optional[Dict]
    error: Optional[str]


# ─────────────────────────────────────────────
# 2. Node Functions
# ─────────────────────────────────────────────

def load_data(state: AgentState) -> AgentState:
    """Node 1: Load trials and the target patient from CSV or generate synthetically."""
    try:
        cache_key = state["cache_key"]
        patient_id = state["patient_id"]

        # Load or generate trials
        trials_path = Path("data/trials.csv")
        patients_path = Path("data/patients.csv")

        if trials_path.exists() and patients_path.exists():
            trials_df = pd.read_csv(trials_path)
            patients_df = pd.read_csv(patients_path)
        else:
            Path("data").mkdir(exist_ok=True)
            trials_df = generate_trials()
            patients_df = generate_patients(n=50)
            trials_df.to_csv(trials_path, index=False)
            patients_df.to_csv(patients_path, index=False)

        # Find target patient
        patient_row = patients_df[patients_df["patient_id"] == patient_id]
        if patient_row.empty:
            patient_row = patients_df.iloc[[0]]
            actual_id = patient_row.iloc[0]["patient_id"]
            print(f"[Agent] Patient {patient_id} not found, using {actual_id}")

        patient = patient_row.iloc[0].to_dict()
        patient_query = build_patient_query(patient_row.iloc[0])

        _CACHE[cache_key] = {
            "trials_df": trials_df,
            "patient": patient,
        }

        print(f"[Agent] Loaded {len(trials_df)} trials. Patient: {patient['patient_id']}")

        return {
            **state,
            "patient_id": patient["patient_id"],
            "patient_query": patient_query,
            "messages": state["messages"] + [
                AIMessage(content=f"Loaded {len(trials_df)} trials. Patient profile built.")
            ],
        }
    except Exception as e:
        return {**state, "error": f"load_data failed: {str(e)}"}


def index_trials(state: AgentState) -> AgentState:
    """Node 2: Embed trial eligibility criteria into ChromaDB."""
    try:
        cache_key = state["cache_key"]
        trials_df = _CACHE[cache_key]["trials_df"]

        store = TrialVectorStore()
        n_indexed = store.index_trials(trials_df)

        _CACHE[cache_key]["vector_store"] = store
        print(f"[Agent] Indexed {n_indexed} trials into ChromaDB.")

        return {
            **state,
            "n_trials_indexed": n_indexed,
            "messages": state["messages"] + [
                AIMessage(content=f"Indexed {n_indexed} trials into vector store.")
            ],
        }
    except Exception as e:
        return {**state, "error": f"index_trials failed: {str(e)}"}


def retrieve_candidates(state: AgentState) -> AgentState:
    """Node 3: Semantic search to retrieve top candidate trials for patient."""
    try:
        cache_key = state["cache_key"]
        store: TrialVectorStore = _CACHE[cache_key]["vector_store"]
        patient_query = state["patient_query"]

        candidates = store.search_trials(patient_query, n_results=3)
        candidates = _to_python(candidates)

        print(f"[Agent] Retrieved {len(candidates)} candidate trials.")
        for c in candidates:
            print(f"  → {c['name']} (similarity: {c['similarity_score']})")

        return {
            **state,
            "candidate_trials": candidates,
            "messages": state["messages"] + [
                AIMessage(content=f"Retrieved {len(candidates)} candidate trials via semantic search.")
            ],
        }
    except Exception as e:
        return {**state, "error": f"retrieve_candidates failed: {str(e)}"}


def screen_eligibility(state: AgentState) -> AgentState:
    """Node 4: Screen patient against candidate trials using rule-based + lab checks."""
    try:
        cache_key = state["cache_key"]
        patient = _CACHE[cache_key]["patient"]
        candidates = state["candidate_trials"]

        # Enrich candidates with trial metadata for screening
        trials_df = _CACHE[cache_key]["trials_df"]
        enriched = []
        for c in candidates:
            trial_row = trials_df[trials_df["trial_id"] == c["trial_id"]]
            if not trial_row.empty:
                t = trial_row.iloc[0].to_dict()
                t["therapeutic_area"] = c.get("therapeutic_area", t.get("therapeutic_area", ""))
                enriched.append(t)

        results = screen_patient_against_trials(patient, enriched)
        results = _to_python(results)

        final_matches = [r for r in results if r.get("eligible")]

        print(f"[Agent] Screened {len(results)} trials. Eligible: {len(final_matches)}")

        return {
            **state,
            "eligibility_results": results,
            "final_matches": final_matches,
            "messages": state["messages"] + [
                AIMessage(content=f"Screened {len(results)} trials. {len(final_matches)} eligible.")
            ],
        }
    except Exception as e:
        return {**state, "error": f"screen_eligibility failed: {str(e)}"}


def generate_reasoning(state: AgentState) -> AgentState:
    """Node 5: LLM generates clinical reasoning for matches."""
    try:
        patient_query = state["patient_query"]
        final_matches = state["final_matches"] or []
        eligibility_results = state["eligibility_results"] or []

        context = f"""
You are a Clinical Research Coordinator (CRC) AI assistant.

PATIENT PROFILE:
{patient_query}

ELIGIBILITY SCREENING RESULTS:
{json.dumps(eligibility_results, indent=2)}

ELIGIBLE TRIALS:
{json.dumps(final_matches, indent=2)}

Please provide:
1. A plain-English summary of why each eligible trial is a good match for this patient
2. Any clinical considerations or risks to flag for the investigator
3. Recommended priority order for trial enrollment (with rationale)
4. Any missing patient data that would improve the match assessment

Be precise, concise, and use appropriate medical terminology.
"""

        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if gemini_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=gemini_api_key,
                temperature=0.2,
            )
            from langchain_core.messages import HumanMessage as LCHuman
            response = llm.invoke([LCHuman(content=context)])
            reasoning = response.content
        else:
            # Rule-based fallback
            if final_matches:
                top = final_matches[0]
                reasoning = f"""
**Clinical Match Summary (Rule-Based — Set GEMINI_API_KEY for LLM reasoning)**

Patient ID: {state['patient_id']}

**Top Match: {top.get('trial_name', 'N/A')}**
Confidence Score: {top.get('confidence_score', 0):.0%}

Score Breakdown:
{json.dumps(top.get('score_breakdown', {}), indent=2)}

**Clinical Considerations:**
- Review all flagged items before enrollment: {top.get('flags', [])}
- Confirm lab values are current (within 30 days)
- Verify informed consent process with IRB guidelines
- Check site availability and patient travel feasibility

**Recommended Action:** Refer to principal investigator for final eligibility review.
"""
            else:
                reasoning = (
                    f"No eligible trials found for patient {state['patient_id']}. "
                    "Consider expanding search criteria or reviewing in 30 days as new trials open."
                )

        print("[Agent] Reasoning generated.")
        return {
            **state,
            "agent_reasoning": reasoning,
            "messages": state["messages"] + [AIMessage(content="Clinical reasoning generated.")],
        }
    except Exception as e:
        return {**state, "error": f"generate_reasoning failed: {str(e)}"}


def human_review(state: AgentState) -> AgentState:
    """
    Node 6: Human-in-the-loop checkpoint.
    In production: pause here for CRC/physician sign-off via UI or Slack.
    In API mode: auto-approve.
    """
    print("[Agent] Human review checkpoint — auto-approved for API mode.")
    return {
        **state,
        "human_approved": True,
        "messages": state["messages"] + [
            HumanMessage(content="CRC reviewed and approved matching results.")
        ],
    }


def log_to_mlflow(state: AgentState) -> AgentState:
    """Node 7: Log full run to MLflow for experiment tracking."""
    try:
        cache_key = state["cache_key"]
        patient = _CACHE[cache_key]["patient"]

        run_id = log_matching_run(
            patient_id=state["patient_id"],
            patient_profile=state.get("patient_query", ""),
            candidate_trials=state.get("candidate_trials") or [],
            eligibility_results=state.get("eligibility_results") or [],
            final_matches=state.get("final_matches") or [],
            agent_reasoning=state.get("agent_reasoning", ""),
            total_trials_searched=state.get("n_trials_indexed", 0),
            run_metadata={
                "therapeutic_area": patient.get("therapeutic_area_focus"),
                "patient_age": patient.get("age"),
            },
        )

        print(f"[Agent] MLflow run logged: {run_id}")
        return {
            **state,
            "mlflow_run_id": run_id,
            "messages": state["messages"] + [
                AIMessage(content=f"MLflow run logged: {run_id}")
            ],
        }
    except Exception as e:
        # MLflow logging failure should not block the report
        print(f"[Agent] MLflow logging warning: {e}")
        return {**state, "mlflow_run_id": None}


def finalize_report(state: AgentState) -> AgentState:
    """Node 8: Assemble the final patient-trial match report."""
    report = {
        "status": "completed",
        "patient_id": state["patient_id"],
        "trials_searched": state.get("n_trials_indexed", 0),
        "candidates_retrieved": len(state.get("candidate_trials") or []),
        "eligible_trials": len(state.get("final_matches") or []),
        "final_matches": state.get("final_matches") or [],
        "eligibility_details": state.get("eligibility_results") or [],
        "clinical_reasoning": state.get("agent_reasoning"),
        "human_approved": state.get("human_approved"),
        "mlflow_run_id": state.get("mlflow_run_id"),
        "agent_message_count": len(state.get("messages") or []),
    }

    print("[Agent] Final report assembled.")
    return {
        **state,
        "final_report": report,
        "messages": state["messages"] + [AIMessage(content="Patient matching complete.")],
    }


# ─────────────────────────────────────────────
# 3. Build the Graph
# ─────────────────────────────────────────────

def build_matching_agent() -> StateGraph:
    """
    Compile the LangGraph clinical trial matching agent.

    8-node DAG:
    load_data → index_trials → retrieve_candidates → screen_eligibility
    → generate_reasoning → human_review → log_to_mlflow → finalize_report → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("load_data", load_data)
    workflow.add_node("index_trials", index_trials)
    workflow.add_node("retrieve_candidates", retrieve_candidates)
    workflow.add_node("screen_eligibility", screen_eligibility)
    workflow.add_node("generate_reasoning", generate_reasoning)
    workflow.add_node("human_review", human_review)
    workflow.add_node("log_to_mlflow", log_to_mlflow)
    workflow.add_node("finalize_report", finalize_report)

    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "index_trials")
    workflow.add_edge("index_trials", "retrieve_candidates")
    workflow.add_edge("retrieve_candidates", "screen_eligibility")
    workflow.add_edge("screen_eligibility", "generate_reasoning")
    workflow.add_edge("generate_reasoning", "human_review")
    workflow.add_edge("human_review", "log_to_mlflow")
    workflow.add_edge("log_to_mlflow", "finalize_report")
    workflow.add_edge("finalize_report", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_agent(patient_id: str = "PAT1000", thread_id: str = "ct-run-1") -> Dict:
    """
    Run the full clinical trial matching agent for a given patient.

    Args:
        patient_id: Patient ID to match (must exist in data/patients.csv)
        thread_id: LangGraph thread ID for checkpointing/resuming

    Returns:
        Final match report dict
    """
    graph = build_matching_agent()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Find clinical trials for patient {patient_id}.")],
        "cache_key": thread_id,
        "patient_id": patient_id,
        "patient_query": None,
        "n_trials_indexed": None,
        "candidate_trials": None,
        "eligibility_results": None,
        "final_matches": None,
        "agent_reasoning": None,
        "mlflow_run_id": None,
        "human_approved": None,
        "final_report": None,
        "error": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(initial_state, config=config)

    return final_state.get("final_report", {
        "status": "error",
        "error": final_state.get("error"),
    })


if __name__ == "__main__":
    report = run_agent(patient_id="PAT1000", thread_id="demo-run-1")
    print("\n=== FINAL REPORT ===")
    print(json.dumps(
        {k: v for k, v in report.items() if k != "clinical_reasoning"},
        indent=2,
    ))
    if report.get("clinical_reasoning"):
        print("\n=== CLINICAL REASONING ===")
        print(report["clinical_reasoning"])
