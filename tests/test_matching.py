"""
Test suite: Clinical Trial Patient Matching Agent
"""

import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock

from src.utils.data_generator import generate_trials, generate_patients
from src.models.eligibility_screener import (
    extract_age_range,
    check_hard_exclusions,
    score_patient_trial,
    screen_patient_against_trials,
)


# ─── Fixtures ─────────────────────────────────

@pytest.fixture(scope="module")
def trials_df():
    return generate_trials()

@pytest.fixture(scope="module")
def patients_df():
    return generate_patients(n=20)

@pytest.fixture(scope="module")
def sample_patient(patients_df):
    return patients_df.iloc[0].to_dict()

@pytest.fixture(scope="module")
def sample_trial(trials_df):
    return trials_df.iloc[0].to_dict()


# ─── Data Tests ───────────────────────────────

class TestDataGenerator:
    def test_trials_shape(self, trials_df):
        assert len(trials_df) == 6
        assert "trial_id" in trials_df.columns
        assert "inclusion_criteria" in trials_df.columns
        assert "exclusion_criteria" in trials_df.columns

    def test_patients_shape(self, patients_df):
        assert len(patients_df) == 20
        assert "patient_id" in patients_df.columns
        assert "age" in patients_df.columns

    def test_trial_ids_unique(self, trials_df):
        assert trials_df["trial_id"].nunique() == len(trials_df)

    def test_patient_ids_unique(self, patients_df):
        assert patients_df["patient_id"].nunique() == len(patients_df)

    def test_labs_parseable(self, patients_df):
        for _, row in patients_df.iterrows():
            labs = json.loads(row["labs"])
            assert isinstance(labs, dict)


# ─── Eligibility Screener Tests ───────────────

class TestEligibilityScreener:
    def test_age_range_extraction(self):
        text = "Age 18-75 years"
        lo, hi = extract_age_range(text)
        assert lo == 18
        assert hi == 75

    def test_age_range_and_above(self):
        text = "Age 4 and above"
        lo, hi = extract_age_range(text)
        assert lo == 4
        assert hi == 120

    def test_no_hard_exclusions_clean_patient(self):
        patient = {
            "comorbidities": "Hypertension",
            "all_conditions": "NSCLC",
            "current_medications": "Aspirin",
            "pregnant": False,
        }
        exclusion = "Active autoimmune disease | Prior organ transplant | Active brain metastases"
        triggered = check_hard_exclusions(patient, exclusion)
        assert len(triggered) == 0

    def test_hard_exclusion_pregnant(self):
        patient = {
            "comorbidities": "",
            "all_conditions": "NSCLC",
            "current_medications": "",
            "pregnant": True,
        }
        exclusion = "Pregnant or breastfeeding"
        triggered = check_hard_exclusions(patient, exclusion)
        assert "pregnant" in triggered

    def test_score_patient_trial_returns_dict(self, sample_patient, sample_trial):
        result = score_patient_trial(sample_patient, sample_trial)
        assert "confidence_score" in result
        assert "eligible" in result
        assert "score_breakdown" in result
        assert "flags" in result

    def test_confidence_score_range(self, sample_patient, sample_trial):
        result = score_patient_trial(sample_patient, sample_trial)
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_screen_multiple_trials(self, sample_patient, trials_df):
        trials = trials_df.to_dict("records")
        results = screen_patient_against_trials(sample_patient, trials)
        assert len(results) == len(trials)
        # Sorted by confidence descending
        scores = [r["confidence_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_age_fail_sets_flag(self):
        patient = {
            "age": 90,
            "therapeutic_area_focus": "Oncology",
            "comorbidities": "",
            "all_conditions": "NSCLC",
            "current_medications": "",
            "labs": "{}",
            "pregnant": False,
        }
        trial = {
            "trial_id": "TEST001",
            "name": "Test Trial",
            "therapeutic_area": "Oncology",
            "inclusion_criteria": "Age 18-75",
            "exclusion_criteria": "",
        }
        result = score_patient_trial(patient, trial)
        assert any("Age" in f for f in result["flags"])


# ─── Agent Tests ──────────────────────────────

class TestMatchingAgent:
    def test_agent_imports(self):
        from src.agents.matching_agent import build_matching_agent, run_agent
        assert callable(run_agent)

    def test_agent_graph_builds(self):
        from src.agents.matching_agent import build_matching_agent
        graph = build_matching_agent()
        assert graph is not None

    def test_full_agent_run(self):
        from src.agents.matching_agent import run_agent
        report = run_agent(patient_id="PAT1000", thread_id="test-run-1")
        assert report.get("status") == "completed"
        assert "patient_id" in report
        assert "final_matches" in report
        assert "eligibility_details" in report
        assert "clinical_reasoning" in report
        assert report.get("human_approved") is True

    def test_agent_metrics_sensible(self):
        from src.agents.matching_agent import run_agent
        report = run_agent(patient_id="PAT1001", thread_id="test-run-2")
        assert report.get("trials_searched", 0) > 0
        assert report.get("candidates_retrieved", 0) >= 0
        assert isinstance(report.get("final_matches"), list)


# ─── Vector Store Tests ───────────────────────

class TestVectorStore:
    def test_build_patient_query(self, patients_df):
        from src.utils.vector_store import build_patient_query
        patient = patients_df.iloc[0]
        query = build_patient_query(patient)
        assert "Age" in query
        assert len(query) > 50

    def test_vector_store_indexes_and_searches(self, trials_df):
        from src.utils.vector_store import TrialVectorStore
        store = TrialVectorStore(persist_path="./data/test_chroma")
        n = store.index_trials(trials_df)
        assert n == len(trials_df)
        results = store.search_trials("cancer patient oncology lung", n_results=2)
        assert len(results) >= 1
        assert "similarity_score" in results[0]
        assert 0 <= results[0]["similarity_score"] <= 1
