"""
Vector Store: Embeds clinical trial eligibility criteria into ChromaDB
for semantic similarity search against patient profiles.
"""

import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./data/chroma_db"


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def build_trial_documents(trials_df: pd.DataFrame) -> list[dict]:
    """
    Convert each trial into a rich text document for embedding.
    Combines name, area, criteria into a single searchable string.
    """
    docs = []
    for _, row in trials_df.iterrows():
        text = f"""
Trial: {row['name']}
Therapeutic Area: {row['therapeutic_area']}
Phase: {row['phase']}
Drug: {row['drug']}
Inclusion Criteria: {row['inclusion_criteria']}
Exclusion Criteria: {row['exclusion_criteria']}
Primary Endpoint: {row['primary_endpoint']}
        """.strip()

        docs.append({
            "id": row["trial_id"],
            "document": text,
            "metadata": {
                "trial_id": row["trial_id"],
                "name": row["name"],
                "therapeutic_area": row["therapeutic_area"],
                "phase": row["phase"],
                "drug": row["drug"],
            }
        })
    return docs


def build_patient_query(patient: pd.Series) -> str:
    """Convert patient record to a natural language query for vector search."""
    labs = json.loads(patient.get("labs", "{}"))
    lab_str = ", ".join([f"{k}: {v}" for k, v in labs.items()])

    return f"""
Patient profile:
Age: {patient['age']}, Sex: {patient['sex']}
Primary condition: {patient['primary_condition']}
All conditions: {patient['all_conditions']}
Comorbidities: {patient['comorbidities']}
Current medications: {patient['current_medications']}
Lab values: {lab_str}
Pregnant: {patient.get('pregnant', False)}
Prior trial participation: {patient.get('prior_trial_participation', False)}
    """.strip()


class TrialVectorStore:
    def __init__(self, persist_path: str = CHROMA_PATH):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        self.ef = get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name="clinical_trials",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def index_trials(self, trials_df: pd.DataFrame) -> int:
        """Embed and index all trials. Returns number indexed."""
        docs = build_trial_documents(trials_df)

        # Clear existing
        existing = self.collection.get()
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        self.collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["document"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
        )
        print(f"[VectorStore] Indexed {len(docs)} trials.")
        return len(docs)

    def search_trials(self, patient_query: str, n_results: int = 3) -> list[dict]:
        """
        Semantic search: find top-N trials most similar to patient profile.
        Returns list of {trial_id, name, therapeutic_area, score, document}.
        """
        results = self.collection.query(
            query_texts=[patient_query],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "trial_id": results["ids"][0][i],
                "name": results["metadatas"][0][i]["name"],
                "therapeutic_area": results["metadatas"][0][i]["therapeutic_area"],
                "phase": results["metadatas"][0][i]["phase"],
                "drug": results["metadatas"][0][i]["drug"],
                "similarity_score": round(1 - results["distances"][0][i], 4),
                "document": results["documents"][0][i],
            })

        return sorted(matches, key=lambda x: x["similarity_score"], reverse=True)

    def count(self) -> int:
        return self.collection.count()
