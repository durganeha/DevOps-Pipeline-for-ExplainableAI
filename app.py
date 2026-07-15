"""
app.py

Streamlit UI for the DevOps Pipeline for Explainable AI (loan approval domain).

Tabs:
    1. Fairness Audit    - live loan prediction + Disparate Impact status
    2. Explainability     - SHAP-based explanation of a prediction
    3. Drift Monitoring    - CUSUM bias-drift simulation
    4. Pipeline History    - latest CI fairness_report.json, if present

Run with:
    streamlit run app.py
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_processed_data, FEATURE_COLS, PROTECTED_COL, LABEL_COL
from src.fairness_metrics import disparate_impact
from src.explainability import (
    MODEL_PATH,
    load_model_bundle,
    get_shap_explainer,
    explain_single_prediction_text,
    plot_shap_summary,
)

st.set_page_config(page_title="Fairness-Aware Loan Approval Pipeline", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model_bundle():
    if not MODEL_PATH.exists():
        from src.train_model import train_and_save
        train_and_save()
    return load_model_bundle()


@st.cache_data
def get_background_data(_bundle):
    df = load_processed_data()
    return df[_bundle["feature_names"]].sample(min(200, len(df)), random_state=1)


bundle = get_model_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
background_data = get_background_data(bundle)


# ---------------------------------------------------------------------------
# Sidebar: applicant input form (shared across tabs)
# ---------------------------------------------------------------------------

st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=1500, step=100)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=1, value=140, step=1)
loan_term = st.sidebar.selectbox("Loan Term (days)", [360, 180, 300, 120, 240, 60])
credit_history = st.sidebar.selectbox("Credit History Meets Guidelines", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_row = pd.DataFrame([{
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1.0 if credit_history == "Yes" else 0.0,
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area],
}])[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_audit, tab_explain, tab_drift, tab_history = st.tabs(
    ["🔍 Fairness Audit", "🧠 Explainability", "📉 Drift Monitoring", "🗂️ Pipeline History"]
)

# --- Tab 1: Fairness Audit --------------------------------------------------
with tab_audit:
    st.title("Fairness Audit")
    st.caption("Automated DevOps Pipeline for Explainable AI — Loan Approval Model")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Live Prediction")
        applicant_scaled = scaler.transform(applicant_row)
        prediction = model.predict(applicant_scaled)[0]
        probability = model.predict_proba(applicant_scaled)[0][1]

        if prediction == 1:
            st.success(f"✅ Loan Approved (confidence: {probability:.1%})")
        else:
            st.error(f"❌ Loan Rejected (confidence: {1 - probability:.1%})")

    with col2:
        st.subheader("Model Fairness Status")
        test_di = bundle.get("test_disparate_impact")
        test_acc = bundle.get("test_accuracy")

        threshold = 0.8
        if test_di is not None:
            if test_di >= threshold:
                st.success(f"Disparate Impact (Gender): {test_di:.3f} — PASSES fairness gate (≥ {threshold})")
            else:
                st.error(f"Disparate Impact (Gender): {test_di:.3f} — FAILS fairness gate (< {threshold})")
        if test_acc is not None:
            st.metric("Model Accuracy (holdout)", f"{test_acc:.1%}")

    st.divider()
    st.subheader("Batch Audit")
    st.caption("Disparate Impact computed across the full processed dataset, by Gender.")
    df_all = load_processed_data()
    X_all_scaled = scaler.transform(df_all[FEATURE_COLS])
    preds_all = model.predict(X_all_scaled)
    di_all = disparate_impact(preds_all, df_all[PROTECTED_COL].to_numpy())
    st.metric("Dataset-wide Disparate Impact (Gender)", f"{di_all:.3f}")


# --- Tab 2: Explainability ---------------------------------------------------
with tab_explain:
    st.title("Explainability")
    st.caption("SHAP-based explanations — why did the model decide what it decided?")

    st.subheader("Why this applicant's prediction?")
    with st.spinner("Computing SHAP explanation..."):
        explainer = get_shap_explainer(bundle, background_data)
        explanation_text = explain_single_prediction_text(
            bundle, applicant_row, background_data=background_data, top_n=5
        )
    st.code(explanation_text, language=None)

    st.divider()
    st.subheader("Global Feature Importance")
    st.caption("Average impact of each feature across a sample of applicants.")
    with st.spinner("Generating SHAP summary plot..."):
        plot_path = plot_shap_summary(bundle, background_data, out_path="shap_summary.png")
    st.image(plot_path, use_container_width=True)


# --- Tab 3: Drift Monitoring --------------------------------------------------
with tab_drift:
    st.title("Drift Monitoring")
    st.caption(
        "Simulates loan applications arriving in batches over time and uses CUSUM to detect "
        "when the DEPLOYED model's fairness drifts — without the model itself changing."
    )

    if st.button("Run drift simulation"):
        with st.spinner("Simulating batches and running CUSUM..."):
            result = subprocess.run(
                [sys.executable, "scripts/cusum_drift_detection.py"],
                capture_output=True, text=True,
            )
        st.text(result.stdout)
        if result.returncode != 0:
            st.error(result.stderr)
        plot_file = Path("cusum_drift_plot.png")
        if plot_file.exists():
            st.image(str(plot_file), use_container_width=True)
    else:
        st.info("Click the button above to simulate incoming data batches and monitor for bias drift.")


# --- Tab 4: Pipeline History --------------------------------------------------
with tab_history:
    st.title("Pipeline History")
    st.caption("Latest automated fairness-gate result from CI (or a local run).")

    report_path = Path("fairness_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.json(report)
        if report.get("passed"):
            st.success("Latest run: PASSED the fairness gate")
        else:
            st.error("Latest run: FAILED the fairness gate")
    else:
        st.info(
            "No fairness_report.json found yet. Run `python scripts/fairness_gate.py` "
            "locally, or check the Actions tab on GitHub after a push."
        )

    st.divider()
    st.caption(
        "In CI, this report is generated automatically by .github/workflows/fairness-gate.yml "
        "on every push and uploaded as a downloadable artifact."
    )