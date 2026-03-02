"""
Configuration management for A/B Testing Framework
処理時間短縮・メモリ効率を考慮した最適化設定
"""

from dataclasses import dataclass, field

from .types import ABTestConfiguration, ABTestVariant, StatisticalTest

@dataclass
class ABTestPerformanceConfig:
    """パフォーマンス最適化設定"""

    # メモリ管理
    max_memory_mb: int = 1024
    compression_threshold_mb: int = 100
    cleanup_interval_seconds: int = 300

    # 処理最適化
    batch_size: int = 1000
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # ストリーミング処理
    stream_buffer_size: int = 10000
    enable_prefetching: bool = True
    prefetch_buffer_size: int = 1000

    # 早期停止
    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_samples: int = 500

@dataclass
class ABTestRiskConfig:
    """リスク管理設定"""

    # 回帰検知
    max_regression_rate: float = 0.05
    regression_detection_window: int = 100
    regression_alert_threshold: float = 0.03

    # トラフィック制御
    max_traffic_percentage: float = 25.0
    traffic_ramp_up_steps: int = 5
    traffic_ramp_up_interval_minutes: int = 30

    # フォールバック
    enable_automatic_rollback: bool = True
    rollback_cooldown_hours: int = 6
    rollback_trigger_threshold: float = 0.1

@dataclass
class ABTestStatisticalConfig:
    """統計分析設定"""

    # 検定設定
    default_test: StatisticalTest = StatisticalTest.T_TEST
    confidence_level: float = 0.95
    alpha: float = 0.05

    # サンプルサイズ
    min_sample_size: int = 500
    max_sample_size: int = 50000
    sample_size_check_interval: int = 100

    # 効果量
    min_effect_size: float = 0.1
    effect_size_method: str = "cohen_d"

    # 信頼区間
    confidence_interval_method: str = "bootstrap"
    bootstrap_iterations: int = 1000

@dataclass
class ABTestConfig:
    """A/Bテスト全体設定"""

    # 基本設定
    enabled: bool = True
    max_concurrent_tests: int = 3
    default_test_duration_hours: int = 24

    # パフォーマンス設定
    performance: ABTestPerformanceConfig = field(
        default_factory=ABTestPerformanceConfig
    )

    # リスク管理設定
    risk: ABTestRiskConfig = field(default_factory=ABTestRiskConfig)

    # 統計分析設定
    statistics: ABTestStatisticalConfig = field(default_factory=ABTestStatisticalConfig)

    # 通知設定
    notifications_enabled: bool = True
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "high_regression_rate": 0.1,
            "low_confidence": 0.8,
            "insufficient_samples": 0.5,
        }
    )

    # ログ設定
    log_level: str = "INFO"
    enable_detailed_logging: bool = False
    log_retention_days: int = 30

    def __post_init__(self):
        """設定の検証と初期化"""
        if self.max_concurrent_tests < 1:
            raise ValueError("max_concurrent_tests must be at least 1")

        if not (0 < self.performance.max_memory_mb <= 8192):
            raise ValueError("max_memory_mb must be between 1 and 8192")

        if not (0.0 <= self.risk.max_regression_rate <= 1.0):
            raise ValueError("max_regression_rate must be between 0.0 and 1.0")

        if not (0.0 < self.statistics.alpha <= 0.5):
            raise ValueError("alpha must be between 0.0 and 0.5")

    @classmethod
    def create_default_test_config(
        cls,
        test_name: str,
        variant_a: ABTestVariant,
        variant_b: ABTestVariant,
        target_metric: str = "rmse",
    ) -> ABTestConfiguration:
        """デフォルトのテスト設定を作成（処理時間短縮・メモリ効率最適化）"""
        return ABTestConfiguration(
            test_id=f"ab_test_{test_name}_{int(__import__('time').time())}",
            name=test_name,
            description=f"A/B test for {test_name}",
            variant_a=variant_a,
            variant_b=variant_b,
            target_metric=target_metric,
            statistical_test=StatisticalTest.T_TEST,
            minimum_sample_size=1000,
            maximum_sample_size=10000,
            confidence_level=0.95,
            minimum_effect_size=0.1,
            max_duration_hours=24,
            check_interval_minutes=15,
            traffic_percentage=10.0,
            max_regression_threshold=0.05,
            batch_size=500,  # メモリ効率のため中程度のバッチサイズ
            enable_compression=True,
            enable_parallel_processing=True,
        )

    @classmethod
    def create_performance_optimized_config(cls) -> "ABTestConfig":
        """パフォーマンス最適化された設定を作成"""
        config = cls()

        # メモリ使用量を削減
        config.performance.max_memory_mb = 512
        config.performance.batch_size = 2000
        config.performance.stream_buffer_size = 5000

        # 並列処理を有効化
        config.performance.max_workers = 8
        config.performance.enable_caching = True

        # 早期停止を有効化
        config.performance.enable_early_stopping = True
        config.performance.early_stopping_patience = 3

        return config

    @classmethod
    def create_memory_conservative_config(cls) -> "ABTestConfig":
        """メモリ節約型の設定を作成"""
        config = cls()

        # メモリ使用量を大幅に削減
        config.performance.max_memory_mb = 256
        config.performance.batch_size = 500
        config.performance.stream_buffer_size = 2000
        config.performance.prefetch_buffer_size = 500

        # 圧縮を有効化
        config.performance.compression_threshold_mb = 50

        # 並列処理を制限
        config.performance.max_workers = 2

        # 統計分析を最適化
        config.statistics.max_sample_size = 25000
        config.statistics.bootstrap_iterations = 500

        return config

    def get_optimized_config_for_environment(self) -> "ABTestConfig":
        """環境に応じた最適化設定を取得"""
        try:
            import psutil

            memory_gb = psutil.virtual_memory().total / (1024**3)

            if memory_gb < 4:  # 4GB未満
                return self.create_memory_conservative_config()
            elif memory_gb < 8:  # 8GB未満
                config = self.create_performance_optimized_config()
                config.performance.max_memory_mb = 512
                return config
            else:  # 8GB以上
                return self.create_performance_optimized_config()

        except ImportError:
            # psutilが利用できない場合は標準設定
            return self
