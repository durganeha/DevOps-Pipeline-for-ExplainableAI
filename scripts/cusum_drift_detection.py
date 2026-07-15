"""
cusum_drift_detection.py

Simulates loan applications arriving in batches over time and uses CUSUM
(Cumulative Sum) to detect when the deployed model's fairness (Disparate
Impact) drifts below an acceptable level — WITHOUT the model itself ever
changing. This models a real production scenario: a fixed, already-approved
model whose predictions become unfair as the real-world applicant population
shifts (e.g. an economic downturn that disproportionately affects one group's
income/credit history).

Usage:
    python scripts/cusum_drift_detection.py

Outputs:
    - Per-batch DI scores + CUSUM statistic printed to console
    - cusum_drift_plot.png showing both series and the detected alarm point
"""

import sys
from pathlib import Path

# Defensive fix: Windows consoles often default to cp1252, which can't
# encode certain unicode characters and crashes print(). Force UTF-8 output
# regardless of platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe for CI environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import FEATURE_COLS, PROTECTED_COL
from src.fairness_metrics import disparate_impact


def generate_batch(n_samples: int, covariate_gap: float, seed: int):
    """
    Generates one batch of loan applications for a point in time.
    `covariate_gap` widens income/credit-history gaps between genders as it
    increases, simulating real-world drift AFTER the model has been deployed
    and validated (the model itself is never touched here).
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    gender = rng.choice([1, 0], size=n, p=[0.8, 0.2])  # 1 = Male, 0 = Female
    married = rng.choice([1, 0], size=n, p=[0.65, 0.35])
    dependents = rng.choice([0, 1, 2, 3], size=n, p=[0.55, 0.2, 0.15, 0.1])
    education = rng.choice([1, 0], size=n, p=[0.78, 0.22])
    self_employed = rng.choice([1, 0], size=n, p=[0.15, 0.85])

    applicant_income = rng.normal(5400, 3000, n) - covariate_gap * 2200 * (gender == 0)
    coapplicant_income = rng.normal(1600, 2000, n).clip(0, None)
    loan_amount = rng.normal(140, 60, n).clip(9, None)
    loan_term = rng.choice([360, 180, 300, 120, 240, 60], size=n, p=[0.75, 0.08, 0.06, 0.04, 0.04, 0.03])

    # Credit history is the strongest legitimate predictor in this dataset —
    # widening the gap here is what really drives the drift.
    credit_hist_prob = np.clip(0.84 - covariate_gap * 0.28 * (gender == 0), 0.05, 0.95)
    credit_history = (rng.random(n) < credit_hist_prob).astype(float)

    property_area = rng.choice([2, 1, 0], size=n, p=[0.38, 0.38, 0.24])
    applicant_income = applicant_income.clip(150, None)

    import pandas as pd
    return pd.DataFrame({
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area,
    })


def train_baseline_model():
    """Trains a fixed model on 'clean' (no covariate gap) data, representing
    the model that already passed fairness_gate.py at deploy time."""
    train_batch = generate_batch(3000, covariate_gap=0.0, seed=0)

    # Ground truth for training only — approval depends on legitimate factors,
    # matching the relationship baked into src/data_loader.py's synthetic data.
    logit = (
        2.8 * train_batch["Credit_History"]
        + 0.00008 * train_batch["ApplicantIncome"]
        + 0.00008 * train_batch["CoapplicantIncome"]
        - 0.004 * train_batch["LoanAmount"]
        - 1.6
    )
    prob = 1 / (1 + np.exp(-logit))
    rng = np.random.default_rng(1)
    y_train = (rng.random(len(train_batch)) < prob).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_batch[FEATURE_COLS])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def run_cusum(di_scores: list, target: float, k: float, h: float):
    """
    One-sided (lower) CUSUM for detecting a downward shift in DI.
    S_t = max(0, S_{t-1} + (target - DI_t) - k); alarm when S_t > h.
    """
    S = 0.0
    cusum_values = []
    alarm_batch = None
    for t, di in enumerate(di_scores):
        S = max(0.0, S + (target - di) - k)
        cusum_values.append(S)
        if alarm_batch is None and S > h:
            alarm_batch = t
    return cusum_values, alarm_batch


def main():
    n_batches = 12
    batch_size = 800
    # First few batches: world looks like training data (no gap). Later
    # batches: a real-world covariate gap gradually opens up.
    covariate_gap_schedule = [0.0] * 4 + list(np.linspace(0.15, 1.6, n_batches - 4))

    model, scaler = train_baseline_model()

    di_scores = []
    for i in range(n_batches):
        batch = generate_batch(batch_size, covariate_gap=covariate_gap_schedule[i], seed=100 + i)
        X_scaled = scaler.transform(batch[FEATURE_COLS])
        preds = model.predict(X_scaled)
        di = disparate_impact(preds.astype(int), batch[PROTECTED_COL].to_numpy())
        di_scores.append(di)
        print(f"Batch {i+1:2d}: Disparate Impact = {di:.3f}")

    target_di = 0.9
    k = 0.05
    h = 0.5
    cusum_values, alarm_batch = run_cusum(di_scores, target_di, k, h)

    print("\n" + "=" * 50)
    print("CUSUM DRIFT DETECTION SUMMARY")
    print("=" * 50)
    for i, (di, s) in enumerate(zip(di_scores, cusum_values)):
        flag = "  <-- ALARM" if alarm_batch == i else ""
        print(f"Batch {i+1:2d} | DI = {di:.3f} | CUSUM = {s:.3f}{flag}")

    if alarm_batch is not None:
        print(f"\nDRIFT DETECTED at batch {alarm_batch + 1} "
              f"(detection latency: {alarm_batch + 1} batches after monitoring started)")
    else:
        print("\nNo drift detected within the monitored window.")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(range(1, n_batches + 1), di_scores, marker="o", label="Disparate Impact")
    axes[0].axhline(0.8, color="red", linestyle="--", label="Legal threshold (0.8)")
    axes[0].set_ylabel("Disparate Impact")
    axes[0].set_title("Disparate Impact per Batch (Loan Approval Model)")
    axes[0].legend()

    axes[1].plot(range(1, n_batches + 1), cusum_values, marker="o", color="orange", label="CUSUM statistic")
    axes[1].axhline(h, color="red", linestyle="--", label=f"Alarm threshold (h={h})")
    if alarm_batch is not None:
        axes[1].axvline(alarm_batch + 1, color="red", linestyle=":", label="Drift detected")
    axes[1].set_xlabel("Batch (simulated time)")
    axes[1].set_ylabel("CUSUM statistic")
    axes[1].set_title("CUSUM Bias-Drift Monitoring")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cusum_drift_plot.png", dpi=150)
    print("\nPlot saved to cusum_drift_plot.png")


if __name__ == "__main__":
    main()