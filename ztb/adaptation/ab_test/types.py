"""
Type definitions for A/B Testing Framework
メモリ効率と処理時間を考慮したストリーミング処理対応
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

# ストリーミングデータ処理のための型定義
SampleData = dict[str, Any]  # 柔軟なサンプルデータ型

# 型エイリアス
SampleDataType = dict[str, Any]

class StatisticalTest(Enum):
    """統計検定タイプ"""

    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"

class ABTestStatus(Enum):
    """A/Bテストステータス"""

    CREATED = "created"  # 作成済み
    RUNNING = "running"  # 実行中
    PAUSED = "paused"  # 一時停止
    COMPLETED = "completed"  # 完了
    CANCELLED = "cancelled"  # キャンセル
    FAILED = "failed"  # 失敗

class ABTestResult(Enum):
    """A/Bテスト結果"""

    INCONCLUSIVE = "inconclusive"  # 結論不明
    WINNER_A = "winner_a"  # Aが勝者
    WINNER_B = "winner_b"  # Bが勝者
    TIE = "tie"  # 同等

@dataclass
class ABTestVariant:
    """A/Bテストのバリアント（AまたはB）"""

    variant_id: str
    model_path: str
    model_version: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class StatisticalResult:
    """統計分析結果"""

    test_type: StatisticalTest
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    sample_size_a: int
    sample_size_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StreamingStatistics:
    """ストリーミング統計計算クラス（メモリ効率重視）"""

    variant_id: str
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # For Welford's online algorithm
    min_val: float = float("inf")
    max_val: float = float("-inf")

    def add_sample(self, sample: SampleDataType) -> None:
        """サンプルを追加して統計を更新（Welfordのオンラインアルゴリズム）"""
        value = sample.get("value", sample.get("reward", 0.0))

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def get_variance(self) -> float:
        """分散を計算"""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def get_std(self) -> float:
        """標準偏差を計算"""
        return self.get_variance() ** 0.5

    def reset(self) -> None:
        """統計をリセット"""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")

@dataclass
class RiskAssessment:
    """リスク評価結果"""

    overall_risk: str  # "low", "medium", "high"
    sample_size_risk: str = "low"
    statistical_risk: str = "low"
    performance_risk: str = "low"
    regression_risk: str = "low"
    risk_factors: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

@dataclass
class ABTestMetrics:
    """A/Bテストメトリクス（ストリーミング計算対応）"""

    variant_id: str
    sample_count: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0

@dataclass
class ABTestConfiguration:
    """A/Bテスト設定"""

    test_id: str
    name: str
    description: str

    # テスト対象
    variant_a: ABTestVariant
    variant_b: ABTestVariant

    # テスト条件
    target_metric: str = "rmse"  # 評価指標
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    minimum_sample_size: int = 1000
    maximum_sample_size: int = 10000
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.1

    # 時間制御
    max_duration_hours: int = 24
    check_interval_minutes: int = 15

    # リスク管理
    traffic_percentage: float = 10.0  # テスト対象トラフィックの割合
    max_regression_threshold: float = 0.05  # 許容最大回帰率

    # パフォーマンス最適化
    batch_size: int = 100  # バッチ処理サイズ
    enable_compression: bool = True  # データ圧縮有効化
    enable_parallel_processing: bool = True  # 並列処理有効化

    # メタデータ
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

@dataclass
class ABTestState:
    """A/Bテスト状態（実行時情報）"""

    test_id: str
    status: ABTestStatus
    start_time: datetime | None = None
    end_time: datetime | None = None
    current_sample_count: int = 0

    # メトリクス（ストリーミング更新）
    metrics_a: ABTestMetrics = field(default_factory=lambda: ABTestMetrics("A"))
    metrics_b: ABTestMetrics = field(default_factory=lambda: ABTestMetrics("B"))

    # 統計結果
    latest_statistical_result: StatisticalResult | None = None

    # リスク管理
    regression_detected: bool = False
    early_stop_triggered: bool = False

    # パフォーマンス監視
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

@dataclass
class ABTestResultSummary:
    """A/Bテスト結果サマリー"""

    test_id: str
    result: ABTestResult
    winner_variant_id: str | None
    confidence_level: float
    statistical_result: StatisticalResult
    risk_assessment: dict[str, Any]
    recommendations: list[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ABTestReport:
    """A/Bテストレポート"""

    test_id: str
    configuration: ABTestConfiguration
    state: ABTestState
    result_summary: ABTestResultSummary | None
    performance_metrics: dict[str, Any]
    risk_metrics: dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)

# コールバック型定義（メモリ効率的な処理のため）
SampleProcessorCallback = Callable[[SampleDataType], None]
TestCompletionCallback = Callable[[ABTestResultSummary], None]
RiskAlertCallback = Callable[[str, dict[str, Any]], None]

# ストリーミングデータ処理のためのイテレータ型
SampleData = dict[str, Any]  # 柔軟なサンプルデータ型
