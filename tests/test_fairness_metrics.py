"""
tests/test_fairness_metrics.py

Unit tests for src/fairness_metrics.py. Run with:
    pytest tests/test_fairness_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fairness_metrics import (
    disparate_impact,
    equal_opportunity_difference,
    statistical_parity_difference,
)


def test_disparate_impact_perfectly_fair():
    """Equal approval rates across groups should give DI == 1.0."""
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert disparate_impact(y_pred, protected) == pytest.approx(1.0)


def test_disparate_impact_maximally_unfair():
    """Privileged group all approved, unprivileged all rejected -> DI == 0.0."""
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert disparate_impact(y_pred, protected) == pytest.approx(0.0)


def test_disparate_impact_partial_bias():
    """Unprivileged group approved half as often as privileged -> DI == 0.5."""
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])  # priv: 4/4, unpriv: 2/4... wait, adjust below
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # priv (protected==1): indices 0-3, all approved (4/4 = 1.0)
    # unpriv (protected==0): indices 4-11, y_pred = [1,1,1,1,0,0,0,0] -> 4/8 = 0.5
    di = disparate_impact(y_pred, protected)
    assert di == pytest.approx(0.5)


def test_disparate_impact_raises_when_group_missing():
    """Should raise if one of the two groups isn't present at all."""
    y_pred = np.array([1, 1, 0, 0])
    protected = np.array([1, 1, 1, 1])  # no unprivileged group present
    with pytest.raises(ValueError):
        disparate_impact(y_pred, protected)


def test_equal_opportunity_difference_fair():
    """Equal TPR among qualified individuals in both groups -> ~0 difference."""
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])  # 2/4 TPR in each group
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert equal_opportunity_difference(y_true, y_pred, protected) == pytest.approx(0.0)


def test_equal_opportunity_difference_biased():
    """Privileged group has higher TPR among qualified individuals -> positive difference."""
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 1])  # priv: 4/4 TPR, unpriv: 1/4 TPR
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    diff = equal_opportunity_difference(y_true, y_pred, protected)
    assert diff > 0


def test_statistical_parity_difference_fair():
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert statistical_parity_difference(y_pred, protected) == pytest.approx(0.0)


def test_statistical_parity_difference_biased():
    """Privileged group approved more often -> positive statistical parity difference."""
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    diff = statistical_parity_difference(y_pred, protected)
    assert diff > 0