"""
train_model.py

Trains the baseline loan-approval classifier and saves it to
models/baseline_model.pkl. This is the model that:
  - the Streamlit app (app.py) loads for live predictions + SHAP explanations
  - scripts/fairness_gate.py evaluates in CI
  - scripts/cusum_drift_detection.py treats as the "already deployed" model
    being monitored for drift

Usage:
    python src/train_model.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow running as a script

from src.data_loader import load_processed_data, FEATURE_COLS, LABEL_COL, PROTECTED_COL
from src.fairness_metrics import disparate_impact

MODEL_PATH = Path("models/baseline_model.pkl")


def train_and_save():
    df = load_processed_data()

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    y_pred = model.predict(X_test_scaled)
    di_score = disparate_impact(y_pred, X_test[PROTECTED_COL].to_numpy())

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Test Disparate Impact (Gender): {di_score:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": FEATURE_COLS,
            "protected_col": PROTECTED_COL,
            "label_col": LABEL_COL,
            "test_accuracy": test_acc,
            "test_disparate_impact": di_score,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model bundle to {MODEL_PATH}")

    return model, di_score


if __name__ == "__main__":
    train_and_save()