"""
tests/test_drift_detection.py

Unit tests for the CUSUM logic in scripts/cusum_drift_detection.py.
Tests the CUSUM algorithm itself in isolation (not the full data-generation
pipeline), so these run fast and don't depend on model training.

Run with:
    pytest tests/test_drift_detection.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.cusum_drift_detection import run_cusum


def test_cusum_no_alarm_when_stable():
    """DI scores hovering right around the target should never trigger an alarm."""
    di_scores = [0.9, 0.91, 0.89, 0.9, 0.92, 0.88, 0.9, 0.91]
    cusum_values, alarm_batch = run_cusum(di_scores, target=0.9, k=0.05, h=0.5)
    assert alarm_batch is None
    assert all(v >= 0 for v in cusum_values)


def test_cusum_detects_sustained_drop():
    """A sustained, large drop in DI should eventually trigger an alarm."""
    di_scores = [0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3]
    cusum_values, alarm_batch = run_cusum(di_scores, target=0.9, k=0.05, h=0.5)
    assert alarm_batch is not None
    assert alarm_batch >= 3  # can't alarm before the drop starts


def test_cusum_ignores_small_noise_within_slack():
    """Small fluctuations smaller than the slack (k) should not accumulate into an alarm."""
    di_scores = [0.9, 0.87, 0.9, 0.88, 0.91, 0.89, 0.9, 0.88] * 3
    cusum_values, alarm_batch = run_cusum(di_scores, target=0.9, k=0.05, h=0.5)
    assert alarm_batch is None


def test_cusum_resets_after_recovery():
    """CUSUM statistic should reset toward 0 once DI recovers back to target,
    since it's a max(0, ...) accumulator, not a permanent running total."""
    di_scores = [0.9, 0.4, 0.4, 0.95, 0.95, 0.95, 0.95, 0.95]
    cusum_values, _ = run_cusum(di_scores, target=0.9, k=0.05, h=100.0)  # h huge so no alarm interferes
    # after several recovered batches, CUSUM should have decayed back down
    assert cusum_values[-1] < cusum_values[2]


def test_cusum_earlier_alarm_with_lower_threshold():
    """A lower alarm threshold (h) should trigger at the same time or earlier
    than a higher one, for the same drift pattern."""
    di_scores = [0.9, 0.9, 0.6, 0.5, 0.4, 0.3, 0.2]
    _, alarm_low_h = run_cusum(di_scores, target=0.9, k=0.05, h=0.3)
    _, alarm_high_h = run_cusum(di_scores, target=0.9, k=0.05, h=1.0)
    assert alarm_low_h is not None
    assert alarm_high_h is None or alarm_low_h <= alarm_high_h