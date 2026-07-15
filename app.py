"""
app.py

Streamlit UI for the DevOps Pipeline for Explainable AI (loan approval domain).

Tabs:
    1. Fairness Audit     - live loan prediction + Disparate Impact status
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

st.set_page_config(
    page_title="Fairness-Aware Loan Approval Pipeline",
    page_icon="🧭",
    layout="wide",
)

FAIRNESS_THRESHOLD = 0.8

# -----------------------------------------------------------------------------
# Design system: fonts, colors, and component CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --navy: #0B3D5C;
    --teal: #0E7C86;
    --teal-light: #5FD3C4;
    --bg: #F4F9F9;
    --card-bg: #FFFFFF;
    --ink: #122B33;
    --muted: #5C7078;
    --good: #1B8A5A;
    --good-bg: #E5F5EC;
    --bad: #C4453F;
    --bad-bg: #FBE9E8;
    --border: #DCE8E8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--navy); letter-spacing: -0.01em; }

/* Top banner */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.6rem; margin-bottom: 1.2rem;
    background: linear-gradient(120deg, var(--navy) 0%, var(--teal) 100%);
    border-radius: 14px;
    color: white;
}
.app-header h1 { color: white !important; font-size: 1.5rem; margin: 0; }
.app-header p { color: #DCF3F0; margin: 0.15rem 0 0 0; font-size: 0.92rem; }
.app-header .badge {
    background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.35);
    padding: 0.3rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
}

/* KPI cards */
.kpi-row { display: flex; gap: 0.9rem; margin-bottom: 1.3rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 200px; background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.2rem;
}
.kpi-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600; color: var(--navy); }
.kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }

/* Status pill */
.pill { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600; }
.pill-good { background: var(--good-bg); color: var(--good); }
.pill-bad { background: var(--bad-bg); color: var(--bad); }

/* Fairness gauge (signature element) */
.gauge-wrap { margin: 0.4rem 0 0.2rem 0; }
.gauge-track { position: relative; height: 34px; background: #E7EFEF; border-radius: 8px; overflow: visible; }
.gauge-fill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 8px 0 0 8px; transition: width 0.4s ease; }
.gauge-fill-good { background: linear-gradient(90deg, var(--teal-light), var(--teal)); }
.gauge-fill-bad { background: linear-gradient(90deg, #F0928D, var(--bad)); }
.gauge-marker { position: absolute; top: -6px; width: 2px; height: 46px; background: var(--navy); }
.gauge-marker-label {
    position: absolute; top: -26px; transform: translateX(-50%);
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--navy); font-weight: 600;
    white-space: nowrap;
}
.gauge-scale { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); margin-top: 0.3rem; font-family: 'JetBrains Mono', monospace; }

/* Section card */
.section-card {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}

/* Sidebar section labels */
.sidebar-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin: 0.8rem 0 0.2rem 0; font-weight: 600; }

/* Feature contribution row (explainability) */
.contrib-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.45rem 0; border-bottom: 1px solid var(--border); }
.contrib-name { flex: 0 0 160px; font-size: 0.85rem; color: var(--ink); font-weight: 500; }
.contrib-bar-track { flex: 1; height: 10px; background: #E7EFEF; border-radius: 5px; position: relative; }
.contrib-bar-fill { position: absolute; top: 0; height: 100%; border-radius: 5px; }
.contrib-val { flex: 0 0 70px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def fairness_gauge_html(di_score: float, threshold: float = FAIRNESS_THRESHOLD, scale_max: float = 1.3) -> str:
    """Renders the signature Fairness Gauge: a horizontal bar showing DI
    against the legal threshold marker."""
    fill_pct = max(0, min(di_score / scale_max, 1.0)) * 100
    marker_pct = (threshold / scale_max) * 100
    fill_class = "gauge-fill-good" if di_score >= threshold else "gauge-fill-bad"
    return f"""
    <div class="gauge-wrap">
        <div class="gauge-track">
            <div class="gauge-fill {fill_class}" style="width:{fill_pct:.1f}%;"></div>
            <div class="gauge-marker" style="left:{marker_pct:.1f}%;"></div>
            <div class="gauge-marker-label" style="left:{marker_pct:.1f}%;">{threshold:.2f} threshold</div>
        </div>
        <div class="gauge-scale"><span>0.0</span><span>{scale_max/2:.2f}</span><span>{scale_max:.2f}</span></div>
    </div>
    """


def contrib_bar_html(name: str, value: float, raw, max_abs: float) -> str:
    color = "#1B8A5A" if value > 0 else "#C4453F"
    pct = min(abs(value) / max_abs, 1.0) * 100 if max_abs > 0 else 0
    if value > 0:
        left_pos, bar_width = 50, pct / 2
    else:
        left_pos, bar_width = 50 - pct / 2, pct / 2
    return f"""
    <div class="contrib-row">
        <div class="contrib-name">{name}</div>
        <div class="contrib-bar-track">
            <div class="contrib-bar-fill" style="left:{left_pos:.1f}%; width:{bar_width:.1f}%; background:{color};"></div>
        </div>
        <div class="contrib-val" style="color:{color};">{value:+.3f}</div>
    </div>
    """


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
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <div>
        <h1>Fairness-Aware Loan Approval Pipeline</h1>
        <p>DevOps Pipeline for Explainable AI &middot; Automated fairness gate, SHAP explainability, CUSUM drift monitoring</p>
    </div>
    <div class="badge">Gender &middot; Disparate Impact &ge; 0.80</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: applicant input form (shared across tabs)
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Applicant Details")

st.sidebar.markdown('<div class="sidebar-label">Personal</div>', unsafe_allow_html=True)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

st.sidebar.markdown('<div class="sidebar-label">Financial</div>', unsafe_allow_html=True)
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=1500, step=100)
credit_history = st.sidebar.selectbox("Credit History Meets Guidelines", ["Yes", "No"])

st.sidebar.markdown('<div class="sidebar-label">Loan Terms</div>', unsafe_allow_html=True)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=1, value=140, step=1)
loan_term = st.sidebar.selectbox("Loan Term (days)", [360, 180, 300, 120, 240, 60])
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
# KPI summary row (always visible)
# ---------------------------------------------------------------------------

test_di = bundle.get("test_disparate_impact")
test_acc = bundle.get("test_accuracy")
di_pass = test_di is not None and test_di >= FAIRNESS_THRESHOLD

kpi_html = f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Model Accuracy (holdout)</div>
        <div class="kpi-value">{test_acc:.1%}</div>
        <div class="kpi-sub">Logistic Regression, scaled features</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Disparate Impact (Gender)</div>
        <div class="kpi-value">{test_di:.3f}</div>
        <div class="kpi-sub"><span class="pill {'pill-good' if di_pass else 'pill-bad'}">{'PASSES' if di_pass else 'FAILS'} fairness gate</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Protected Attribute</div>
        <div class="kpi-value" style="font-size:1.3rem;">Gender</div>
        <div class="kpi-sub">Legal threshold: DI &ge; 0.80</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_audit, tab_explain, tab_drift, tab_history = st.tabs(
    ["🔍  Fairness Audit", "💡  Explainability", "📊  Drift Monitoring", "📁  Pipeline History"]
)

# --- Tab 1: Fairness Audit --------------------------------------------------
with tab_audit:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Live Prediction")
        applicant_scaled = scaler.transform(applicant_row)
        prediction = model.predict(applicant_scaled)[0]
        probability = model.predict_proba(applicant_scaled)[0][1]

        if prediction == 1:
            st.success(f"Loan Approved — confidence {probability:.1%}")
        else:
            st.error(f"Loan Rejected — confidence {1 - probability:.1%}")
        st.caption("Based on the applicant details entered in the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Model Fairness Status")
        st.markdown(fairness_gauge_html(test_di, FAIRNESS_THRESHOLD), unsafe_allow_html=True)
        status_text = "above" if di_pass else "below"
        st.caption(f"Disparate Impact of {test_di:.3f} is {status_text} the 0.80 legal threshold (the \"80% rule\").")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Batch Audit")
    st.caption("Disparate Impact computed across the full processed dataset, by Gender.")
    df_all = load_processed_data()
    X_all_scaled = scaler.transform(df_all[FEATURE_COLS])
    preds_all = model.predict(X_all_scaled)
    di_all = disparate_impact(preds_all, df_all[PROTECTED_COL].to_numpy())
    st.markdown(fairness_gauge_html(di_all, FAIRNESS_THRESHOLD), unsafe_allow_html=True)
    st.caption(f"Dataset-wide Disparate Impact (Gender): **{di_all:.3f}** across {len(df_all)} applicants.")
    st.markdown('</div>', unsafe_allow_html=True)


# --- Tab 2: Explainability ---------------------------------------------------
with tab_explain:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Why this applicant's prediction?")
    st.caption("SHAP contribution of each feature toward this specific applicant's outcome.")

    with st.spinner("Computing SHAP explanation..."):
        explainer = get_shap_explainer(bundle, background_data)
        shap_values = explainer.shap_values(scaler.transform(applicant_row[bundle["feature_names"]]))
        row_shap = shap_values[0] if shap_values.ndim > 1 else shap_values
        feature_names = bundle["feature_names"]
        contributions = sorted(
            zip(feature_names, row_shap, applicant_row.iloc[0][feature_names].values),
            key=lambda x: abs(x[1]), reverse=True,
        )
        max_abs = max(abs(v) for _, v, _ in contributions) or 1.0

    rows_html = "".join(contrib_bar_html(name, val, raw, max_abs) for name, val, raw in contributions)
    st.markdown(rows_html, unsafe_allow_html=True)
    st.caption("Bars extend right (green) if a feature increased approval likelihood, left (red) if it decreased it.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Global Feature Importance")
    st.caption("Average impact of each feature across a sample of applicants.")
    with st.spinner("Generating SHAP summary plot..."):
        plot_path = plot_shap_summary(bundle, background_data, out_path="shap_summary.png")
    st.image(plot_path, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- Tab 3: Drift Monitoring --------------------------------------------------
with tab_drift:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Drift Monitoring")
    st.caption(
        "Simulates loan applications arriving in batches over time and uses CUSUM to detect "
        "when the deployed model's fairness drifts — without the model itself changing."
    )

    if st.button("Run drift simulation", type="primary"):
        with st.spinner("Simulating batches and running CUSUM..."):
            result = subprocess.run(
                [sys.executable, "scripts/cusum_drift_detection.py"],
                capture_output=True, text=True,
            )
        st.code(result.stdout, language=None)
        if result.returncode != 0:
            st.error(result.stderr)
        plot_file = Path("cusum_drift_plot.png")
        if plot_file.exists():
            st.image(str(plot_file), use_container_width=True)
    else:
        st.info("Click the button above to simulate incoming data batches and monitor for bias drift.")
    st.markdown('</div>', unsafe_allow_html=True)


# --- Tab 4: Pipeline History --------------------------------------------------
with tab_history:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Pipeline History")
    st.caption("Latest automated fairness-gate result from CI (or a local run).")

    report_path = Path("fairness_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        passed = report.get("passed")
        st.markdown(
            f'<span class="pill {"pill-good" if passed else "pill-bad"}">'
            f'Latest run: {"PASSED" if passed else "FAILED"} the fairness gate</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.json(report)
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
    st.markdown('</div>', unsafe_allow_html=True)