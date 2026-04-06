"""
Eligibility Screener: Rule-based + LLM screening of patients against trial criteria.
Assigns a confidence score to each patient-trial pair.
"""

import json
import re
from typing import Dict, List, Tuple


HARD_EXCLUSION_KEYWORDS = [
    "pregnant", "breastfeeding", "hiv", "active tb",
    "prior organ transplant", "prior gene therapy",
    "hypersensitivity",
]


def extract_age_range(criteria_text: str) -> Tuple[int, int]:
    """Extract age range from inclusion criteria text."""
    match = re.search(r"[Aa]ge\s+(\d+)[\s\-–]+(\d+)", criteria_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"[Aa]ge\s+(\d+)\s+and above", criteria_text)
    if match:
        return int(match.group(1)), 120
    return 0, 120


def check_hard_exclusions(patient: Dict, exclusion_criteria: str) -> List[str]:
    """
    Check if patient hits any hard exclusion criteria.
    Returns list of triggered exclusions (empty = none triggered).
    """
    triggered = []
    patient_text = " ".join([
        str(patient.get("comorbidities", "")),
        str(patient.get("all_conditions", "")),
        str(patient.get("current_medications", "")),
        "pregnant" if patient.get("pregnant") else "",
    ]).lower()

    for keyword in HARD_EXCLUSION_KEYWORDS:
        if keyword in patient_text and keyword in exclusion_criteria.lower():
            triggered.append(keyword)

    return triggered


def score_patient_trial(patient: Dict, trial: Dict) -> Dict:
    """
    Score a patient-trial pair with a confidence score (0-1).

    Scoring components:
        - Age eligibility (0.25)
        - Therapeutic area alignment (0.25)
        - No hard exclusions (0.30)
        - Lab value compatibility (0.20)

    Returns dict with score, breakdown, and pass/fail flags.
    """
    score = 0.0
    breakdown = {}
    flags = []

    inclusion = trial.get("inclusion_criteria", "")
    exclusion = trial.get("exclusion_criteria", "")

    # 1. Age check (0.25)
    min_age, max_age = extract_age_range(inclusion)
    age = patient.get("age", 0)
    if min_age <= age <= max_age:
        score += 0.25
        breakdown["age"] = f"PASS ({age} within {min_age}-{max_age})"
    else:
        breakdown["age"] = f"FAIL ({age} outside {min_age}-{max_age})"
        flags.append(f"Age {age} outside range {min_age}-{max_age}")

    # 2. Therapeutic area alignment (0.25)
    patient_area = patient.get("therapeutic_area_focus", "").lower()
    trial_area = trial.get("therapeutic_area", "").lower()
    if patient_area and trial_area and (patient_area in trial_area or trial_area in patient_area):
        score += 0.25
        breakdown["therapeutic_area"] = f"PASS ({trial_area})"
    else:
        breakdown["therapeutic_area"] = f"PARTIAL ({patient_area} vs {trial_area})"
        score += 0.10  # partial credit

    # 3. Hard exclusion check (0.30)
    exclusions_triggered = check_hard_exclusions(patient, exclusion)
    if not exclusions_triggered:
        score += 0.30
        breakdown["exclusion_screen"] = "PASS (no hard exclusions triggered)"
    else:
        breakdown["exclusion_screen"] = f"FAIL: {', '.join(exclusions_triggered)}"
        flags.append(f"Hard exclusions: {', '.join(exclusions_triggered)}")

    # 4. Lab compatibility (0.20)
    try:
        labs = json.loads(patient.get("labs", "{}"))
        lab_issues = []

        # Generic lab checks based on trial area
        trial_area = trial.get("therapeutic_area", "")
        if trial_area == "Cardiology":
            egfr = labs.get("eGFR", 100)
            if egfr < 20:
                lab_issues.append(f"eGFR {egfr} < 20 mL/min")
            sbp = labs.get("systolic_bp", 120)
            if sbp < 90:
                lab_issues.append(f"Systolic BP {sbp} < 90 mmHg")

        elif trial_area == "Diabetes":
            hba1c = labs.get("HbA1c", 8.0)
            bmi = labs.get("BMI", 30)
            if not (7.5 <= hba1c <= 10.5):
                lab_issues.append(f"HbA1c {hba1c} outside 7.5-10.5%")
            if not (25 <= bmi <= 45):
                lab_issues.append(f"BMI {bmi} outside 25-45")

        elif trial_area == "Oncology":
            ecog = labs.get("ECOG", 1)
            if ecog > 2:
                lab_issues.append(f"ECOG {ecog} > 2")

        elif trial_area == "Neurology":
            mmse = labs.get("MMSE", 25)
            amyloid = labs.get("amyloid_PET", "negative")
            if not (20 <= mmse <= 26):
                lab_issues.append(f"MMSE {mmse} outside 20-26")
            if amyloid == "negative":
                lab_issues.append("Amyloid PET negative")

        elif trial_area == "Immunology":
            lymphs = labs.get("lymphocytes", 1000)
            if lymphs < 500:
                lab_issues.append(f"Lymphocytes {lymphs} < 500")
            das28 = labs.get("DAS28", 4.0)
            if das28 <= 3.2:
                lab_issues.append(f"DAS28 {das28} <= 3.2 (not moderate-severe)")

        if not lab_issues:
            score += 0.20
            breakdown["labs"] = "PASS"
        else:
            breakdown["labs"] = f"CONCERNS: {'; '.join(lab_issues)}"
            flags.append(f"Lab issues: {'; '.join(lab_issues)}")

    except Exception:
        score += 0.10
        breakdown["labs"] = "PARTIAL (could not parse)"

    return {
        "trial_id": trial.get("trial_id"),
        "trial_name": trial.get("name"),
        "confidence_score": round(min(score, 1.0), 3),
        "eligible": score >= 0.55 and not any(
            "FAIL" in str(v) and "exclusion" in k
            for k, v in breakdown.items()
        ),
        "score_breakdown": breakdown,
        "flags": flags,
    }


def screen_patient_against_trials(patient: Dict, candidate_trials: List[Dict]) -> List[Dict]:
    """Screen a patient against all candidate trials and return scored results."""
    results = []
    for trial in candidate_trials:
        result = score_patient_trial(patient, trial)
        results.append(result)

    return sorted(results, key=lambda x: x["confidence_score"], reverse=True)
