"""
MLflow Tracker: Logs every patient matching run as an MLflow experiment.
Tracks inputs, match scores, reasoning, and final recommendations.
"""

import json
import mlflow
import mlflow.pyfunc
from datetime import datetime
from typing import Dict, List, Any


EXPERIMENT_NAME = "clinical-trial-patient-matching"


def setup_mlflow(tracking_uri: str = "./mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_matching_run(
    patient_id: str,
    patient_profile: str,
    candidate_trials: List[Dict],
    eligibility_results: List[Dict],
    final_matches: List[Dict],
    agent_reasoning: str,
    total_trials_searched: int,
    run_metadata: Dict[str, Any] = None,
) -> str:
    """
    Log a complete patient-trial matching run to MLflow.

    Returns the MLflow run_id.
    """
    setup_mlflow()

    run_name = f"match_{patient_id}_{datetime.now().strftime('%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:

        # --- Params ---
        mlflow.log_param("patient_id", patient_id)
        mlflow.log_param("total_trials_in_db", total_trials_searched)
        mlflow.log_param("candidates_retrieved", len(candidate_trials))
        mlflow.log_param("eligible_matches", len(final_matches))
        if run_metadata:
            for k, v in run_metadata.items():
                mlflow.log_param(k, v)

        # --- Metrics ---
        if candidate_trials:
            scores = [t.get("similarity_score", 0) for t in candidate_trials]
            avg_sim = sum(scores) / len(candidate_trials)
            max_sim = max(t.get("similarity_score", 0) for t in candidate_trials)
            mlflow.log_metric("avg_similarity_score", round(avg_sim, 4))
            mlflow.log_metric("max_similarity_score", round(max_sim, 4))

        mlflow.log_metric("match_rate", len(final_matches) / max(total_trials_searched, 1))
        mlflow.log_metric("n_eligible_trials", len(final_matches))

        if final_matches:
            avg_confidence = sum(
                m.get("confidence_score", 0) for m in final_matches
            ) / len(final_matches)
            mlflow.log_metric("avg_match_confidence", round(avg_confidence, 4))

        # --- Artifacts ---
        # Patient profile
        with open("_patient_profile.txt", "w") as f:
            f.write(patient_profile)
        mlflow.log_artifact("_patient_profile.txt", "inputs")

        # Candidate trials
        with open("_candidate_trials.json", "w") as f:
            json.dump(candidate_trials, f, indent=2)
        mlflow.log_artifact("_candidate_trials.json", "retrieval")

        # Eligibility screening results
        with open("_eligibility_results.json", "w") as f:
            json.dump(eligibility_results, f, indent=2)
        mlflow.log_artifact("_eligibility_results.json", "screening")

        # Final matches + recommendations
        with open("_final_matches.json", "w") as f:
            json.dump(final_matches, f, indent=2)
        mlflow.log_artifact("_final_matches.json", "output")

        # Agent reasoning / LLM narrative
        with open("_agent_reasoning.txt", "w") as f:
            f.write(agent_reasoning)
        mlflow.log_artifact("_agent_reasoning.txt", "reasoning")

        # Tags
        mlflow.set_tag("agent_version", "langgraph-v1")
        mlflow.set_tag("embedding_model", "all-MiniLM-L6-v2")

        return run.info.run_id


def get_recent_runs(n: int = 10) -> List[Dict]:
    """Fetch recent matching runs from MLflow."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=n,
    )

    return [
        {
            "run_id": r.info.run_id,
            "patient_id": r.data.params.get("patient_id"),
            "eligible_matches": r.data.metrics.get("n_eligible_trials"),
            "avg_confidence": r.data.metrics.get("avg_match_confidence"),
            "start_time": r.info.start_time,
        }
        for r in runs
    ]
