import numpy as np

from ztb.utils.data.outlier_detection import calculate_z_score_single


def test_calculate_z_score_single_basic():
    history = np.array([0.01, 0.012, 0.01, 0.011, 0.013])
    value = 0.04
    z = calculate_z_score_single(value, history)
    assert isinstance(z, float)
    assert z > 2.0


def test_calculate_z_score_single_empty_history():
    z = calculate_z_score_single(0.05, np.array([]))
    assert z == 0.0


def test_calculate_z_score_single_small_std():
    history = np.array([0.01] * 10)
    z = calculate_z_score_single(0.02, history, min_std=1e-5)
    # small std triggers min_std fallback -> returns 0.0
    assert z == 0.0
    # Test MAD method
    history2 = np.array([0.01, 0.02, 0.015, 0.02, 0.03])
    z_mad = calculate_z_score_single(0.05, history2, method="mad")
    assert isinstance(z_mad, float)
    assert z_mad > 0.0
