"""
Walk-Forward 分析用の型定義

型安全性を確保するための TypedDict と dataclass を集約管理
"""
from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from scripts.v457.backtest_v457 import BacktestReporter

class TimeSeriesWindow(NamedTuple):
    """時系列分割ウィンドウの定義
    
    一つの Walk-Forward ウィンドウを表現するための構造体。
    時系列データの特定の区間（訓練・検証・テスト）を指定します。
    
    Attributes:
        window_id: ウィンドウ識別子（0から開始）
        train_start: 訓練期間の開始インデックス（inclusive）
        train_end: 訓練期間の終了インデックス（exclusive）
        val_start: 検証期間の開始インデックス（inclusive）
        val_end: 検証期間の終了インデックス（exclusive）
        test_start: テスト期間の開始インデックス（inclusive）
        test_end: テスト期間の終了インデックス（exclusive）
    
    Example:
        >>> window = TimeSeriesWindow(
        ...     window_id=0,
        ...     train_start=0, train_end=500,
        ...     val_start=500, val_end=650,
        ...     test_start=650, test_end=800
        ... )
        >>> df_train = df.iloc[window.train_start:window.train_end]
    """
    window_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    def __post_init__(self) -> None:
        """バリデーション（強化版）"""
        # インデックス範囲チェック
        if self.train_end <= self.train_start:
            raise ValueError(
                f"train_end must be > train_start: "
                f"train_end={self.train_end}, train_start={self.train_start}"
            )
        if self.val_end <= self.val_start:
            raise ValueError(
                f"val_end must be > val_start: "
                f"val_end={self.val_end}, val_start={self.val_start}"
            )
        if self.test_end <= self.test_start:
            raise ValueError(
                f"test_end must be > test_start: "
                f"test_end={self.test_end}, test_start={self.test_start}"
            )
        
        # 期間の重複チェック
        if not (self.train_end <= self.val_start):
            raise ValueError(
                f"Ranges must not overlap: train_end={self.train_end} > val_start={self.val_start}. "
                f"Train and Val periods overlap!"
            )
        if not (self.val_end <= self.test_start):
            raise ValueError(
                f"Ranges must not overlap: val_end={self.val_end} > test_start={self.test_start}. "
                f"Val and Test periods overlap!"
            )
        
        # インデックスの単調増加チェック
        if not (self.train_start < self.train_end <= self.val_start < self.val_end <= self.test_start < self.test_end):
            raise ValueError(
                f"Indices must be strictly increasing: "
                f"train [{self.train_start}:{self.train_end}] → "
                f"val [{self.val_start}:{self.val_end}] → "
                f"test [{self.test_start}:{self.test_end}]"
            )
        
        # 最小サイズチェック（各セグメント最低1サンプル）
        if self.train_size < 1:
            raise ValueError(f"Train size must be >= 1, got {self.train_size}")
        if self.val_size < 1:
            raise ValueError(f"Val size must be >= 1, got {self.val_size}")
        if self.test_size < 1:
            raise ValueError(f"Test size must be >= 1, got {self.test_size}")
        
        # 訓練期間は検証・テスト期間より大きい必要がある
        if self.train_size < max(self.val_size, self.test_size):
            import warnings
            warnings.warn(
                f"Window {self.window_id}: Training size ({self.train_size}) "
                f"is smaller than val/test size. This may cause overfitting."
            )

    @property
    def train_size(self) -> int:
        """訓練データサイズ"""
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        """検証データサイズ"""
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        """テストデータサイズ"""
        return self.test_end - self.test_start

    @property
    def total_size(self) -> int:
        """全体サイズ"""
        return self.test_end - self.train_start

@dataclass
class WindowPerformance:
    """ウィンドウ単位の性能メトリクス
    
    一つの Walk-Forward ウィンドウで訓練したモデルの性能を記録します。
    検証期間（in-sample）とテスト期間（out-of-sample）の両方の結果を含みます。
    
    Attributes:
        window_id: ウィンドウ識別子
        in_sample_reward: 訓練期間での累積報酬（RL環境から）
        out_of_sample_reward: テスト期間での累積報酬（RL環境から）
        val_roi: 検証期間の ROI（Return on Investment）
            範囲: -1.0 ～ +∞、例: 0.05 = +5%
        test_roi: テスト期間の ROI（未来データに対する性能）
            過学習検出用に val_roi と比較
        val_final_balance: 検証期間の最終残高（JPY）
        test_final_balance: テスト期間の最終残高（JPY）
        sharpe_ratio: Sharpe 比（リスク調整リターン）
            計算: (mean_return - rf_rate) / std_return
            目安: 0.5-1.0=可、1.0-2.0=良、>2.0=優秀
        max_drawdown: 最大ドローダウン
            範囲: -1.0 ～ 0.0、例: -0.15 = -15%の下落
        win_rate: 勝率（0.0 ～ 1.0）
            例: 0.65 = 65%のトレードが利益
        trades: 実行されたトレード数
    
    Example:
        >>> perf = WindowPerformance(
        ...     window_id=0,
        ...     val_roi=0.0523,
        ...     test_roi=0.0419,
        ...     sharpe_ratio=1.25,
        ...     max_drawdown=-0.082,
        ...     win_rate=0.65,
        ...     trades=42
        ... )
        >>> overfitting = abs(perf.val_roi - perf.test_roi) / abs(perf.val_roi)
        >>> print(f"Overfitting ratio: {overfitting:.2%}")  # 19.9%
    """
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
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    val_reporter: BacktestReporter | None = None
    test_reporter: BacktestReporter | None = None

    @property
    def overfitting_ratio(self) -> float:
        """過学習比率を計算
        
        Returns:
            float: |val_roi - test_roi| / |val_roi|
                0.0-0.20 = 過学習なし（堅牢）
                0.20-0.50 = 中程度の過学習
                >0.50 = 深刻な過学習
        """
        if self.val_roi == 0.0:
            return 0.0
        return abs(self.val_roi - self.test_roi) / abs(self.val_roi)

    @property
    def roi_difference(self) -> float:
        """検証 ROI とテスト ROI の差分
        
        Returns:
            float: val_roi - test_roi
                正の値 = テストセットで性能低下
                負の値 = テストセットで性能向上
        """
        return self.val_roi - self.test_roi

    def validate(self) -> None:
        """パフォーマンス指標の妥当性を検証
        
        Raises:
            ValueError: 不正な値が検出された場合
        """
        # ROI のバリデーション（-1.0 以上である必要がある）
        if self.val_roi < -1.0:
            raise ValueError(
                f"Window {self.window_id}: val_roi={self.val_roi} < -1.0 is invalid. "
                f"ROI must be >= -1.0 (100% loss is the worst case)"
            )
        if self.test_roi < -1.0:
            raise ValueError(
                f"Window {self.window_id}: test_roi={self.test_roi} < -1.0 is invalid"
            )
        
        # Sharpe比のバリデーション（通常は -5 ～ 5 の範囲）
        if abs(self.sharpe_ratio) > 10.0:
            import warnings
            warnings.warn(
                f"Window {self.window_id}: sharpe_ratio={self.sharpe_ratio} is unusually large. "
                f"This may indicate insufficient data or extreme volatility."
            )
        
        # Max Drawdown のバリデーション（-1.0 ～ 0.0 の範囲）
        if not (-1.0 <= self.max_drawdown <= 0.0):
            raise ValueError(
                f"Window {self.window_id}: max_drawdown={self.max_drawdown} must be in [-1.0, 0.0]"
            )
        
        # Win Rate のバリデーション（0.0 ～ 1.0 の範囲）
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError(
                f"Window {self.window_id}: win_rate={self.win_rate} must be in [0.0, 1.0]"
            )
        
        # トレード数のバリデーション（非負）
        if self.trades < 0:
            raise ValueError(
                f"Window {self.window_id}: trades={self.trades} must be >= 0"
            )
        
        # 最終残高のバリデーション（非負、かつ初期残高 >= 0 を前提）
        if self.val_final_balance < 0.0:
            import warnings
            warnings.warn(
                f"Window {self.window_id}: val_final_balance={self.val_final_balance} is negative. "
                f"This means the account is in deficit."
            )
        if self.test_final_balance < 0.0:
            import warnings
            warnings.warn(
                f"Window {self.window_id}: test_final_balance={self.test_final_balance} is negative"
            )

@dataclass
class WalkForwardResult:
    """Walk-Forward 分析全体の集計結果
    
    複数のウィンドウでの訓練・評価の集計結果。
    過学習の有無を判定するための統計量を含みます。
    
    Attributes:
        windows: ウィンドウリスト
        performances: 各ウィンドウの性能結果
        reporters: 各ウィンドウのBacktestReporter
        average_val_roi: 検証 ROI の平均
        average_test_roi: テスト ROI の平均
        test_roi_std: テスト ROI の標準偏差
            低い値 = 複数ウィンドウで安定した性能
        average_sharpe: Sharpe 比の平均
        sharpe_consistency: Sharpe 比の一貫性
            計算: 1 - (sharpe_std / sharpe_mean)
            値が高い = 安定して良い性能
        average_win_rate: 勝率の平均
        overfitting_ratio: 過学習比率の平均
            <0.20 = 堅牢なモデル ✅
            0.20-0.50 = 注視が必要 ⚠️
            >0.50 = 却下推奨 ❌
    
    Example:
        >>> result = WalkForwardResult(
        ...     windows=windows,
        ...     performances=performances,
        ...     reporters=reporters,
        ...     average_test_roi=0.0419,
        ...     overfitting_ratio=0.199
        ... )
        >>> if result.overfitting_ratio < 0.20:
        ...     print("✅ Model is robust!")
    """
    windows: list[TimeSeriesWindow] = field(default_factory=list)
    performances: list[WindowPerformance] = field(default_factory=list)
    reporters: list["BacktestReporter"] = field(default_factory=list)
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
        """モデルが堅牢かどうかを判定
        
        Args:
            threshold: 過学習比率の閾値（デフォルト: 0.20）
            
        Returns:
            bool: 堅牢ならTrue
        """
        # Restore robustness criteria: ROI > 5%, Sharpe > 1.0, PF > 1.2
        return (
            self.overfitting_ratio < threshold
            and self.average_test_roi > 0.05
            and self.average_sharpe > 1.0
            and self.profit_factor > 1.2
        )

# 型別名（type aliases）
WindowList = list[TimeSeriesWindow]
PerformanceList = list[WindowPerformance]
MetricsDict = dict[str, Any]
