"""
fairness_metrics.py

Shared fairness-metric functions used by scripts/fairness_gate.py,
scripts/cusum_drift_detection.py, and app.py — kept in one place so the
UI and the CI check are always computing the SAME thing.
"""

import numpy as np


def disparate_impact(y_pred: np.ndarray, protected: np.ndarray, privileged_value=1) -> float:
    """
    DI = P(favorable outcome | unprivileged) / P(favorable outcome | privileged)
    Favorable outcome here = loan approved (y_pred == 1).
    DI close to 1.0 = fair. DI < 0.8 is the common "80% rule" legal threshold
    used in US employment/housing/credit discrimination law.
    """
    y_pred = np.asarray(y_pred)
    protected = np.asarray(protected)

    unpriv_mask = protected != privileged_value
    priv_mask = protected == privileged_value

    if unpriv_mask.sum() == 0 or priv_mask.sum() == 0:
        raise ValueError("Both privileged and unprivileged groups must be present to compute DI.")

    unpriv_rate = np.mean(y_pred[unpriv_mask] == 1)
    priv_rate = np.mean(y_pred[priv_mask] == 1)

    return 0.0 if priv_rate == 0 else float(unpriv_rate / priv_rate)


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, protected: np.ndarray, privileged_value=1) -> float:
    """
    Difference in True Positive Rate (TPR) between privileged and
    unprivileged groups, restricted to individuals who actually qualify
    (y_true == 1). Close to 0 = fair. Positive = privileged group favored.
    """
    y_true, y_pred, protected = np.asarray(y_true), np.asarray(y_pred), np.asarray(protected)

    unpriv_qualified = (protected != privileged_value) & (y_true == 1)
    priv_qualified = (protected == privileged_value) & (y_true == 1)

    if unpriv_qualified.sum() == 0 or priv_qualified.sum() == 0:
        return float("nan")

    unpriv_tpr = np.mean(y_pred[unpriv_qualified] == 1)
    priv_tpr = np.mean(y_pred[priv_qualified] == 1)

    return float(priv_tpr - unpriv_tpr)


def statistical_parity_difference(y_pred: np.ndarray, protected: np.ndarray, privileged_value=1) -> float:
    """
    Difference in approval rate between privileged and unprivileged groups
    (unlike DI, this is a difference, not a ratio). Close to 0 = fair.
    """
    y_pred, protected = np.asarray(y_pred), np.asarray(protected)
    unpriv_rate = np.mean(y_pred[protected != privileged_value] == 1)
    priv_rate = np.mean(y_pred[protected == privileged_value] == 1)
    return float(priv_rate - unpriv_rate)