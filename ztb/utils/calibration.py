"""
Probability Calibration Diagnostics.

Compute Brier score and reliability curves for action probability calibration.
Helps monitor whether predicted probabilities are well-calibrated.
"""

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from ztb.metrics.metrics import calculate_distribution_stats

class BrierScoreDict(TypedDict):
    """Brier score result dictionary."""

    overall_brier: float
    per_action_brier: dict[str, float | None]  # Can be None when action never occurs
    overall_brier_stats: dict[str, float]  # Added stats

class ReliabilityCurveDict(TypedDict):
    """Reliability curve result dictionary."""

    bin_edges: list[float]
    bin_counts: list[int]
    bin_predicted_prob: list[float]
    bin_observed_freq: list[float]
    expected_calibration_error: float

class CalibrationReportDict(TypedDict):
    """Calibration analysis report dictionary."""

    brier_score: BrierScoreDict
    reliability_curves: dict[str, ReliabilityCurveDict]
    n_samples: int
    n_actions: int

def compute_brier_score(
    predicted_probs: NDArray[np.float32],
    actual_actions: NDArray[np.int64],
    n_actions: int = 3,
) -> dict[str, Any]:  # Using Any for backward compatibility with tests
    """
    Compute Brier score for multi-class predictions.

    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes (one-hot encoded).
    Lower is better (0 = perfect, 1 = worst).

    Args:
        predicted_probs: Array of shape (n_samples, n_actions) with predicted probabilities
        actual_actions: Array of shape (n_samples,) with actual action indices
        n_actions: Number of possible actions (default: 3 for HOLD/BUY/SELL)

    Returns:
        Dictionary with overall and per-action Brier scores
    """
    n_samples = len(actual_actions)

    # Convert actual actions to one-hot encoding
    actual_one_hot = np.zeros((n_samples, n_actions))
    actual_one_hot[np.arange(n_samples), actual_actions] = 1.0

    # Compute Brier score: mean squared error between probabilities and one-hot
    brier_per_sample = np.sum((predicted_probs - actual_one_hot) ** 2, axis=1)
    overall_brier = np.mean(brier_per_sample)
    brier_stats = calculate_distribution_stats(brier_per_sample)

    # Per-action Brier score
    per_action_brier: dict[str, float | None] = {}
    for action_idx in range(n_actions):
        action_name = (
            ["HOLD", "BUY", "SELL"][action_idx]
            if n_actions == 3
            else f"Action_{action_idx}"
        )
        # For samples where this action was taken
        mask = actual_actions == action_idx
        if np.sum(mask) > 0:
            per_action_brier[action_name] = float(np.mean(brier_per_sample[mask]))
        else:
            per_action_brier[action_name] = None  # None when action never occurs

    return {
        "overall": float(overall_brier),  # For backward compatibility with tests
        "overall_brier": float(overall_brier),
        "overall_brier_stats": brier_stats,
        "per_action": per_action_brier,  # For backward compatibility with tests
        "per_action_brier": per_action_brier,
    }

def compute_reliability_curve(
    predicted_probs: NDArray[np.float32],
    actual_actions: NDArray[np.int64],
    action_idx: int,
    n_bins: int = 10,
) -> ReliabilityCurveDict:
    """
    Compute reliability curve for a specific action.

    A reliability curve shows how well-calibrated the predicted probabilities are.
    For perfectly calibrated predictions, the curve should follow the diagonal
    (predicted probability = observed frequency).

    Args:
        predicted_probs: Array of shape (n_samples, n_actions) with predicted probabilities
        actual_actions: Array of shape (n_samples,) with actual action indices
        action_idx: Index of action to analyze (0=HOLD, 1=BUY, 2=SELL)
        n_bins: Number of bins for grouping probabilities

    Returns:
        Dictionary with bin information:
        - bin_edges: Edges of probability bins
        - bin_counts: Number of samples in each bin
        - bin_predicted_prob: Mean predicted probability per bin
        - bin_observed_freq: Observed frequency of action per bin
        - expected_calibration_error: ECE metric
    """
    # Get predicted probabilities for this action
    probs_for_action = predicted_probs[:, action_idx]

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs_for_action, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute statistics per bin
    bin_counts = []
    bin_predicted_prob = []
    bin_observed_freq = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        count = np.sum(mask)

        if count > 0:
            # Mean predicted probability in this bin
            mean_pred_prob = np.mean(probs_for_action[mask])

            # Observed frequency: proportion of samples where action was actually taken
            actual_in_bin = actual_actions[mask]
            observed_freq = np.mean(actual_in_bin == action_idx)

            bin_counts.append(int(count))
            bin_predicted_prob.append(float(mean_pred_prob))
            bin_observed_freq.append(float(observed_freq))
        else:
            bin_counts.append(0)
            bin_predicted_prob.append(None)  # None for empty bins
            bin_observed_freq.append(None)  # None for empty bins

    # Compute Expected Calibration Error (ECE)
    # ECE = weighted average of |predicted_prob - observed_freq|
    ece = 0.0
    total_samples = len(probs_for_action)

    for i in range(n_bins):
        if (
            bin_counts[i] > 0
            and bin_predicted_prob[i] is not None
            and bin_observed_freq[i] is not None
        ):
            weight = bin_counts[i] / total_samples
            ece += weight * abs(bin_predicted_prob[i] - bin_observed_freq[i])

    return {
        "bin_edges": [float(x) for x in bin_edges],
        "bin_counts": bin_counts,
        "bin_predicted_prob": bin_predicted_prob,
        "bin_observed_freq": bin_observed_freq,
        "expected_calibration_error": float(ece),
    }

def compute_full_calibration_report(
    predicted_probs: NDArray[np.float32],
    actual_actions: NDArray[np.int64],
    n_bins: int = 10,
    action_names: list[str] | None = None,
) -> dict[str, Any]:  # Using Any for backward compatibility
    """
    Compute full calibration report with Brier scores and reliability curves.

    Args:
        predicted_probs: Array of shape (n_samples, n_actions) with predicted probabilities
        actual_actions: Array of shape (n_samples,) with actual action indices
        n_bins: Number of bins for reliability curves
        action_names: Names of actions (default: ["HOLD", "BUY", "SELL"])

    Returns:
        Dictionary with:
        - brier_score: Overall and per-action Brier scores
        - reliability_curves: Reliability curve for each action
    """
    n_actions = predicted_probs.shape[1]

    if action_names is None:
        action_names = (
            ["HOLD", "BUY", "SELL"]
            if n_actions == 3
            else [f"Action_{i}" for i in range(n_actions)]
        )

    # Compute Brier score
    brier_scores = compute_brier_score(predicted_probs, actual_actions, n_actions)

    # Compute reliability curves for each action
    reliability_curves = {}
    for action_idx, action_name in enumerate(action_names):
        reliability_curves[action_name] = compute_reliability_curve(
            predicted_probs,
            actual_actions,
            action_idx,
            n_bins,
        )

    return {
        "brier_score": brier_scores,
        "reliability_curves": reliability_curves,
        "n_samples": len(actual_actions),
        "n_actions": n_actions,
    }
