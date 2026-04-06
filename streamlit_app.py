"""
Clinical Trial Matching — Streamlit Dashboard
"""

import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Clinical Trial Matcher", page_icon="🧬", layout="wide")
st.title("🧬 Clinical Trial Patient Matching Agent")
st.caption("Powered by LangGraph · RAG (ChromaDB) · MLflow · Gemini")

with st.sidebar:
    st.header("⚙️ Run Agent")
    patient_id = st.text_input("Patient ID", value="PAT1000")

    if st.button("🤖 Match Patient to Trials", use_container_width=True):
        with st.spinner("Agent running... (load → embed → retrieve → screen → reason → log)"):
            resp = requests.post(f"{API_URL}/match", json={"patient_id": patient_id})
            if resp.ok:
                st.session_state["report"] = resp.json()["report"]
                st.session_state["thread_id"] = resp.json()["thread_id"]
                st.success("Matching complete!")
            else:
                st.error(f"Error: {resp.text}")

    st.divider()
    if st.button("📊 View MLflow UI", use_container_width=True):
        st.info("Run `mlflow ui` in terminal then open http://localhost:5000")

tab1, tab2, tab3 = st.tabs(["🎯 Match Results", "🔬 Eligibility Details", "📋 Agent Log"])

with tab1:
    if "report" in st.session_state:
        report = st.session_state["report"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patient", report.get("patient_id"))
        col2.metric("Trials Searched", report.get("trials_searched"))
        col3.metric("Candidates Retrieved", report.get("candidates_retrieved"))
        col4.metric("Eligible Matches", report.get("eligible_trials"))

        st.divider()

        matches = report.get("final_matches", [])
        if matches:
            st.subheader("✅ Eligible Trials")
            for m in matches:
                with st.expander(f"**{m.get('trial_name')}** — Confidence: {m.get('confidence_score', 0):.0%}"):
                    st.progress(m.get("confidence_score", 0))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Score Breakdown**")
                        for k, v in m.get("score_breakdown", {}).items():
                            color = "🟢" if "PASS" in str(v) else "🔴"
                            st.write(f"{color} **{k}**: {v}")
                    with col2:
                        flags = m.get("flags", [])
                        if flags:
                            st.markdown("**⚠️ Flags**")
                            for f in flags:
                                st.warning(f)
        else:
            st.info("No eligible trials found for this patient.")

        st.divider()
        st.subheader("🧠 Clinical Reasoning")
        st.markdown(report.get("clinical_reasoning", "No reasoning available."))

        if report.get("mlflow_run_id"):
            st.caption(f"MLflow Run ID: `{report['mlflow_run_id']}`")
    else:
        st.info("Run the agent from the sidebar to see match results.")

with tab2:
    if "report" in st.session_state:
        details = st.session_state["report"].get("eligibility_details", [])
        if details:
            st.subheader("Full Eligibility Screening Results")
            df = pd.DataFrame([{
                "Trial": d.get("trial_name"),
                "Confidence": f"{d.get('confidence_score', 0):.0%}",
                "Eligible": "✅" if d.get("eligible") else "❌",
                "Flags": "; ".join(d.get("flags", [])) or "None",
            } for d in details])
            st.dataframe(df, use_container_width=True)

            st.subheader("Raw Screening JSON")
            st.json(details)
    else:
        st.info("Run the agent first.")

with tab3:
    if "report" in st.session_state:
        report = st.session_state["report"]
        st.metric("Agent Nodes Executed", report.get("agent_message_count", 0))
        st.metric("Thread ID", st.session_state.get("thread_id", "—"))
        st.metric("Human Approved", "✅" if report.get("human_approved") else "❌")
        st.subheader("Full Report JSON")
        st.json({k: v for k, v in report.items() if k != "clinical_reasoning"})
    else:
        st.info("Run the agent first.")
