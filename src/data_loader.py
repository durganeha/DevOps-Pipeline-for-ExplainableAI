"""
data_loader.py

Loads and preprocesses the Loan Prediction dataset
(Analytics Vidhya / Kaggle "Loan Prediction Problem Dataset").

Expected raw columns (standard schema for this dataset):
    Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
    Credit_History, Property_Area, Loan_Status

Where to get the real dataset:
    Kaggle: search "Loan Prediction Problem Dataset" (by ninzaami), or
    Analytics Vidhya's "Loan Prediction Practice Problem".
    Download train_ctrl.csv / train.csv, rename to loan_data.csv, and place
    it at: data/raw/loan_data.csv

If that file isn't present, this module falls back to a synthetic dataset
with the SAME column names and types, so the rest of the pipeline (model
training, fairness gate, CI) can be built and tested before you've downloaded
the real data. Swap in the real file whenever you're ready — no other code
needs to change.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/loan_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/loan_data_clean.csv")

PROTECTED_COL = "Gender"          # 1 = Male (privileged), 0 = Female (unprivileged)
LABEL_COL = "Loan_Status"         # 1 = approved (favorable), 0 = rejected

FEATURE_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
]


def _generate_synthetic_fallback(n_samples: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic stand-in matching the real dataset's schema, so the pipeline
    is fully runnable before the real CSV is downloaded. Mirrors the real
    dataset's rough shape: Credit_History is the dominant predictor of
    approval (as it is in the real data), with a mild, realistic gender
    disparity baked in for the fairness gate to have something to catch.
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.8, 0.2])
    married = rng.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.55, 0.2, 0.15, 0.1])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], size=n_samples, p=[0.15, 0.85])
    applicant_income = rng.normal(5400, 3000, n_samples).clip(150, None).round()
    coapplicant_income = rng.normal(1600, 2000, n_samples).clip(0, None).round()
    loan_amount = rng.normal(140, 60, n_samples).clip(9, None).round()
    loan_term = rng.choice([360, 180, 300, 120, 240, 60], size=n_samples, p=[0.75, 0.08, 0.06, 0.04, 0.04, 0.03])
    credit_history = rng.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_samples, p=[0.38, 0.38, 0.24])

    # Approval driven mainly by credit history + income (legitimate), plus a
    # mild illegitimate gender effect so the fairness gate has a real signal.
    logit = (
        2.8 * credit_history
        + 0.00008 * applicant_income
        + 0.00008 * coapplicant_income
        - 0.004 * loan_amount
        - 1.6
    )
    logit += np.where(gender == "Male", 0.35, 0.0)

    prob = 1 / (1 + np.exp(-logit))
    loan_status = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "Loan_ID": [f"LP{1000+i}" for i in range(n_samples)],
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    })


def load_raw_data() -> pd.DataFrame:
    """Loads the real dataset if present, otherwise generates the synthetic fallback."""
    if RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Loaded real dataset from {RAW_DATA_PATH} ({len(df)} rows)")
    else:
        df = _generate_synthetic_fallback()
        print(
            f"WARNING: {RAW_DATA_PATH} not found — using synthetic fallback data "
            f"({len(df)} rows). Download the real dataset and place it there when ready."
        )
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and encodes the raw loan dataset into a fully numeric DataFrame
    ready for model training. Handles the missing values and text labels
    that are typical of the real Kaggle/Analytics Vidhya version of this
    dataset.
    """
    df = df.copy()

    # Standardize Loan_Status to 0/1 whether it comes in as Y/N text (as in
    # the real dataset) or already numeric (as in the synthetic fallback).
    # Using replace() instead of a dtype check for the condition, since
    # newer pandas versions can infer string columns as dtype "str" rather
    # than "object", which would silently skip a dtype-gated map().
    df[LABEL_COL] = df[LABEL_COL].replace({"Y": 1, "N": 0})
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    # Fill missing values (the real dataset has several NaNs in these columns)
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    # Encode categoricals to numeric
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
    df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)
    df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
    df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
    df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})

    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    cols = ["Loan_ID"] + FEATURE_COLS + [LABEL_COL] if "Loan_ID" in df.columns else FEATURE_COLS + [LABEL_COL]
    return df[cols]


def load_processed_data(force_reprocess: bool = False) -> pd.DataFrame:
    """
    Main entry point used by train_model.py and the fairness gate. Loads the
    cached processed CSV if it exists, otherwise builds it from raw data.
    """
    if PROCESSED_DATA_PATH.exists() and not force_reprocess:
        return pd.read_csv(PROCESSED_DATA_PATH)

    raw_df = load_raw_data()
    processed_df = preprocess(raw_df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH} ({len(processed_df)} rows)")
    return processed_df


if __name__ == "__main__":
    df = load_processed_data(force_reprocess=True)
    print(df.head())
    print("\nApproval rate by Gender:")
    print(df.groupby("Gender")[LABEL_COL].mean())