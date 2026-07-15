"""
tests/test_data_loader.py

Unit tests for src/data_loader.py's preprocessing logic.
Run with:
    pytest tests/test_data_loader.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import preprocess, LABEL_COL, FEATURE_COLS


def _raw_row(**overrides):
    """Builds one raw-schema row (as it would come from the real Kaggle CSV), with overrides."""
    row = {
        "Loan_ID": "LP9999",
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
        "Loan_Status": "Y",
    }
    row.update(overrides)
    return row


def test_preprocess_encodes_categoricals_correctly():
    df = pd.DataFrame([_raw_row()])
    out = preprocess(df)
    assert out.loc[0, "Gender"] == 1        # Male -> 1
    assert out.loc[0, "Married"] == 1       # Yes -> 1
    assert out.loc[0, "Education"] == 1     # Graduate -> 1
    assert out.loc[0, "Self_Employed"] == 0  # No -> 0
    assert out.loc[0, "Property_Area"] == 2  # Urban -> 2
    assert out.loc[0, LABEL_COL] == 1        # Y -> 1


def test_preprocess_handles_female_and_rejected():
    df = pd.DataFrame([_raw_row(Gender="Female", Loan_Status="N")])
    out = preprocess(df)
    assert out.loc[0, "Gender"] == 0
    assert out.loc[0, LABEL_COL] == 0


def test_preprocess_handles_dependents_3plus():
    df = pd.DataFrame([_raw_row(Dependents="3+")])
    out = preprocess(df)
    assert out.loc[0, "Dependents"] == 3


def test_preprocess_fills_missing_gender_with_mode():
    """Missing Gender values should be imputed with the column mode, not dropped."""
    rows = [_raw_row(Gender="Male") for _ in range(3)] + [_raw_row(Gender=None)]
    df = pd.DataFrame(rows)
    out = preprocess(df)
    assert out["Gender"].isna().sum() == 0
    assert len(out) == 4  # no rows dropped due to missing Gender


def test_preprocess_drops_rows_with_missing_label():
    """Rows with a missing/unparseable Loan_Status should be dropped (can't train on them)."""
    rows = [_raw_row(), _raw_row(Loan_Status=None)]
    df = pd.DataFrame(rows)
    out = preprocess(df)
    assert len(out) == 1


def test_preprocess_output_contains_all_feature_columns():
    df = pd.DataFrame([_raw_row()])
    out = preprocess(df)
    for col in FEATURE_COLS:
        assert col in out.columns