from typing import Optional, TypedDict


class CalibrationStats(TypedDict):
    """Statistics for a specific regime/action bin."""

    p_win_lcb: float
    p_win_mean: float
    avg_win: float
    avg_loss: float
    n_eff: float


class CalibrationStatsBundle(TypedDict):
    """Bundle of stats including L1 and Fallback."""

    l1: CalibrationStats
    fallback: CalibrationStats
    n_min: float


class FusedSignal(TypedDict):
    """Signal fused from RL and Patterns."""

    rl_action: float
    regime: str
    pattern_score: Optional[float]


class GateResult(TypedDict):
    """Result of CalibrationGate evaluation."""

    should_enter: bool
    ev: float
    ev_l1: float
    ev_fb: float
    lambda_val: float
    cost: float
    stats: CalibrationStats
    stats_fallback: CalibrationStats
