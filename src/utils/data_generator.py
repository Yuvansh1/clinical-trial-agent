"""
Synthetic Clinical Trial + Patient Data Generator
Generates realistic trials with eligibility criteria and patient profiles.
"""

import random
import json
import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path

fake = Faker()
random.seed(42)
np.random.seed(42)


THERAPEUTIC_AREAS = [
    "Oncology", "Cardiology", "Neurology",
    "Diabetes", "Rare Disease", "Immunology"
]

PHASES = ["Phase I", "Phase II", "Phase III", "Phase IV"]

TRIAL_TEMPLATES = [
    {
        "name": "APEX-001: Anti-PD1 for Advanced NSCLC",
        "therapeutic_area": "Oncology",
        "phase": "Phase III",
        "inclusion": [
            "Age 18-75",
            "Diagnosed with non-small cell lung cancer (NSCLC) stage III or IV",
            "ECOG performance status 0-2",
            "No prior anti-PD1 or anti-PD-L1 therapy",
            "Adequate organ function (creatinine < 1.5x ULN, ALT/AST < 3x ULN)",
        ],
        "exclusion": [
            "Active autoimmune disease",
            "Prior organ transplant",
            "Active brain metastases",
            "Pregnant or breastfeeding",
            "HIV positive",
        ],
        "primary_endpoint": "Overall survival at 24 months",
        "drug": "Pembrolizumab + Carboplatin",
    },
    {
        "name": "CARDIO-202: SGLT2 Inhibitor for Heart Failure",
        "therapeutic_area": "Cardiology",
        "phase": "Phase III",
        "inclusion": [
            "Age 40-80",
            "Diagnosed with heart failure with reduced ejection fraction (HFrEF)",
            "LVEF < 40%",
            "NYHA Class II-III",
            "On stable guideline-directed medical therapy for at least 3 months",
        ],
        "exclusion": [
            "Type 1 diabetes",
            "eGFR < 20 mL/min",
            "Systolic blood pressure < 90 mmHg",
            "Recent MI within 3 months",
            "Active urinary tract infection",
        ],
        "primary_endpoint": "Composite of CV death and worsening heart failure",
        "drug": "Dapagliflozin 10mg",
    },
    {
        "name": "NEURO-305: Gene Therapy for Early Alzheimer's",
        "therapeutic_area": "Neurology",
        "phase": "Phase II",
        "inclusion": [
            "Age 55-80",
            "Early-stage Alzheimer's disease confirmed by amyloid PET",
            "MMSE score 20-26",
            "Stable on current medications for at least 2 months",
            "Caregiver/study partner available",
        ],
        "exclusion": [
            "MRI contraindications",
            "Severe psychiatric disorder",
            "Active cancer within 5 years",
            "Prior gene therapy",
            "Significant cerebrovascular disease",
        ],
        "primary_endpoint": "Change in ADAS-Cog score at 52 weeks",
        "drug": "AAV9-APOE2 vector",
    },
    {
        "name": "DIAB-410: Dual GLP-1/GIP Agonist for T2DM",
        "therapeutic_area": "Diabetes",
        "phase": "Phase III",
        "inclusion": [
            "Age 25-70",
            "Type 2 diabetes diagnosis for at least 1 year",
            "HbA1c 7.5%-10.5%",
            "BMI 25-45 kg/m²",
            "On metformin monotherapy or diet/exercise alone",
        ],
        "exclusion": [
            "Type 1 diabetes or LADA",
            "eGFR < 30 mL/min",
            "History of pancreatitis",
            "Personal or family history of MTC",
            "Gastroparesis",
        ],
        "primary_endpoint": "HbA1c reduction at 26 weeks",
        "drug": "Tirzepatide 10mg weekly",
    },
    {
        "name": "RARE-101: Enzyme Replacement for Gaucher Type 1",
        "therapeutic_area": "Rare Disease",
        "phase": "Phase II",
        "inclusion": [
            "Age 4 and above",
            "Confirmed Gaucher disease type 1 by enzyme assay or genotyping",
            "Symptomatic with splenomegaly or thrombocytopenia",
            "Naive to enzyme replacement therapy",
        ],
        "exclusion": [
            "Gaucher disease type 2 or 3 (neuronopathic)",
            "Active hepatitis B or C",
            "Severe pulmonary hypertension",
            "Hypersensitivity to imiglucerase",
        ],
        "primary_endpoint": "Reduction in spleen volume at 12 months",
        "drug": "Velaglucerase alfa",
    },
    {
        "name": "IMMUNO-520: JAK Inhibitor for Moderate-Severe RA",
        "therapeutic_area": "Immunology",
        "phase": "Phase III",
        "inclusion": [
            "Age 18-70",
            "Rheumatoid arthritis diagnosis per ACR/EULAR criteria",
            "Moderate to severe disease activity (DAS28 > 3.2)",
            "Inadequate response to methotrexate",
            "At least 6 swollen and 6 tender joints",
        ],
        "exclusion": [
            "Active or latent TB without treatment",
            "Absolute lymphocyte count < 500 cells/mm³",
            "Active serious infection",
            "History of DVT or PE",
            "Prior JAK inhibitor use",
        ],
        "primary_endpoint": "ACR20 response at week 12",
        "drug": "Upadacitinib 15mg daily",
    },
]


def generate_trials() -> pd.DataFrame:
    """Generate clinical trial records."""
    trials = []
    for i, t in enumerate(TRIAL_TEMPLATES):
        trials.append({
            "trial_id": f"NCT{2024000 + i:07d}",
            "name": t["name"],
            "therapeutic_area": t["therapeutic_area"],
            "phase": t["phase"],
            "drug": t["drug"],
            "primary_endpoint": t["primary_endpoint"],
            "inclusion_criteria": " | ".join(t["inclusion"]),
            "exclusion_criteria": " | ".join(t["exclusion"]),
            "status": "Recruiting",
            "sites": random.randint(10, 120),
            "target_enrollment": random.randint(100, 1200),
        })
    return pd.DataFrame(trials)


def generate_patients(n: int = 50) -> pd.DataFrame:
    """Generate synthetic patient profiles."""
    therapeutic_areas = THERAPEUTIC_AREAS
    patients = []

    for i in range(n):
        area = random.choice(therapeutic_areas)
        age = random.randint(20, 82)

        # Build condition list based on therapeutic area
        conditions = []
        labs = {}

        if area == "Oncology":
            conditions = random.sample([
                "Non-small cell lung cancer stage III",
                "Non-small cell lung cancer stage IV",
                "Small cell lung cancer",
                "Breast cancer",
                "Colorectal cancer",
            ], k=random.randint(1, 2))
            labs = {"creatinine": round(random.uniform(0.6, 2.2), 2),
                    "ALT": random.randint(15, 120),
                    "ECOG": random.randint(0, 3)}

        elif area == "Cardiology":
            conditions = random.sample([
                "Heart failure with reduced ejection fraction",
                "Hypertension",
                "Atrial fibrillation",
                "Coronary artery disease",
            ], k=random.randint(1, 2))
            labs = {"LVEF": random.randint(20, 55),
                    "eGFR": random.randint(15, 90),
                    "NYHA_class": random.randint(1, 4),
                    "systolic_bp": random.randint(80, 160)}

        elif area == "Neurology":
            conditions = random.sample([
                "Early Alzheimer's disease",
                "Mild cognitive impairment",
                "Parkinson's disease",
                "Multiple sclerosis",
            ], k=1)
            labs = {"MMSE": random.randint(15, 30),
                    "amyloid_PET": random.choice(["positive", "negative"])}

        elif area == "Diabetes":
            conditions = ["Type 2 diabetes mellitus"]
            labs = {"HbA1c": round(random.uniform(6.5, 12.0), 1),
                    "BMI": round(random.uniform(22, 48), 1),
                    "eGFR": random.randint(25, 95)}

        elif area == "Rare Disease":
            conditions = [random.choice([
                "Gaucher disease type 1",
                "Fabry disease",
                "Pompe disease",
            ])]
            labs = {"glucocerebrosidase_activity": round(random.uniform(0.5, 8.0), 2)}

        elif area == "Immunology":
            conditions = ["Rheumatoid arthritis"]
            labs = {"DAS28": round(random.uniform(2.0, 7.5), 1),
                    "swollen_joints": random.randint(0, 28),
                    "tender_joints": random.randint(0, 28),
                    "lymphocytes": random.randint(300, 2000)}

        comorbidities = random.sample([
            "Hypertension", "Type 2 diabetes", "Hypothyroidism",
            "Chronic kidney disease", "Depression", "Osteoarthritis",
            "Autoimmune disease", "HIV", "Active TB", "History of DVT",
        ], k=random.randint(0, 3))

        medications = random.sample([
            "Metformin", "Lisinopril", "Atorvastatin", "Levothyroxine",
            "Methotrexate", "Prednisone", "Insulin glargine", "Aspirin",
        ], k=random.randint(1, 4))

        patients.append({
            "patient_id": f"PAT{1000 + i}",
            "age": age,
            "sex": random.choice(["Male", "Female"]),
            "primary_condition": conditions[0] if conditions else "Unknown",
            "all_conditions": " | ".join(conditions),
            "comorbidities": " | ".join(comorbidities),
            "current_medications": " | ".join(medications),
            "labs": json.dumps(labs),
            "therapeutic_area_focus": area,
            "prior_trial_participation": random.choice([True, False]),
            "pregnant": random.choice([True, False]) if random.random() < 0.1 else False,
        })

    return pd.DataFrame(patients)


def save_data(base_path: str = "data") -> tuple:
    Path(base_path).mkdir(exist_ok=True)
    trials = generate_trials()
    patients = generate_patients(n=50)
    trials.to_csv(f"{base_path}/trials.csv", index=False)
    patients.to_csv(f"{base_path}/patients.csv", index=False)
    print(f"Saved {len(trials)} trials and {len(patients)} patients.")
    return trials, patients


if __name__ == "__main__":
    save_data()
