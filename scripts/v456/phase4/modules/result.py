"""Walk-Forward Result: 分析結果の集約"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class WindowPerformance:
    """ウィンドウ単位の性能"""
    window_id: int
    in_sample_reward: float = 0.0
    out_of_sample_reward: float = 0.0
    val_roi: float = 0.0
    test_roi: float = 0.0
    val_final_balance: float = 0.0
    test_final_balance: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trades: int = 0


@dataclass
class WalkForwardResult:
    """Walk-Forward分析全体結果"""
    windows: List = field(default_factory=list)
    performances: List[WindowPerformance] = field(default_factory=list)
    average_val_roi: float = 0.0
    average_test_roi: float = 0.0
    test_roi_std: float = 0.0
    average_sharpe: float = 0.0
    sharpe_consistency: float = 0.0  # test window間でのSharpe相関
    average_win_rate: float = 0.0
    overfitting_ratio: float = 0.0  # 訓練 vs テスト性能比
