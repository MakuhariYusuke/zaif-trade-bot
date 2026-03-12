"""
Walk-Forward 分析結果クラス
"""

from dataclasses import dataclass, field

from .types import TimeSeriesWindow, WindowPerformance

@dataclass
class WalkForwardResult:
    """Walk-Forward分析全体結果
    
    複数のウィンドウでの訓練・評価の集計結果。
    過学習の有無を判定するための統計量を含みます。
    
    Attributes:
        windows: ウィンドウリスト
        performances: 各ウィンドウの性能結果
        average_val_roi: 検証 ROI の平均
        average_test_roi: テスト ROI の平均
        test_roi_std: テスト ROI の標準偏差
        average_sharpe: Sharpe 比の平均
        sharpe_consistency: Sharpe 比の一貫性
        average_win_rate: 勝率の平均
        overfitting_ratio: 過学習比率の平均
    """
    windows: list[TimeSeriesWindow] = field(default_factory=list)
    performances: list[WindowPerformance] = field(default_factory=list)
    average_val_roi: float = 0.0
    average_test_roi: float = 0.0
    test_roi_std: float = 0.0
    average_sharpe: float = 0.0
    sharpe_consistency: float = 0.0
    average_win_rate: float = 0.0
    overfitting_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    @property
    def num_windows(self) -> int:
        """ウィンドウ数"""
        return len(self.windows)

    @property
    def roi_improvement(self) -> float:
        """平均テスト ROI の改善度
        
        Returns:
            float: average_test_roi / average_val_roi
        """
        if self.average_val_roi == 0.0:
            return 0.0
        return self.average_test_roi / self.average_val_roi

    def is_robust_model(self, threshold: float = 0.20) -> bool:
        """モデルの堅牢性を判定
        
        Doc08基準: ROI ≥ 1.05 / PF ≥ 1.05 / Sharpe ≥ 0.5 / Overfitting ≤ 0.20
        
        Args:
            threshold: 過学習比率の閾値（デフォルト: 0.20 = 20%）
        
        Returns:
            bool: True if all conditions met
        """
        # 過学習チェック
        if self.overfitting_ratio > threshold:
            return False
        
        # 基本性能チェック (Doc08基準)
        if self.average_test_roi < 1.05:
            return False  # ROI ≥ 1.05
        
        if self.average_sharpe < 0.5:
            return False  # Sharpe ≥ 0.5
        
        # Profit Factorチェック
        if self.profit_factor < 1.05:
            return False  # PF ≥ 1.05
        
        return True
