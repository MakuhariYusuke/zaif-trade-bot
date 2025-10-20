"""
Configuration for Concept Drift Detection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .drift_types import DriftThresholds


@dataclass
class ConceptDriftConfig:
    """コンセプトドリフト検知の設定"""

    # 基本設定
    enabled: bool = True
    detection_interval_seconds: int = 300  # 5分ごと
    max_history_size: int = 1000

    # Kolmogorov-Smirnov設定
    ks_test_significance_level: float = 0.05
    enable_ks_test: bool = True

    # ADWIN設定
    adwin_delta: float = 0.002
    enable_adwin: bool = True

    # DDM設定
    ddm_min_samples: int = 30
    enable_ddm: bool = True

    # EDDM設定
    eddm_window_size: int = 30
    enable_eddm: bool = True

    # 検知閾値
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

    # ウィンドウサイズ設定
    window_sizes: Dict[str, int] = field(
        default_factory=lambda: {
            "short": 100,
            "medium": 1000,
            "long": 5000,
        }  # 短期ウィンドウ  # 中期ウィンドウ  # 長期ウィンドウ
    )

    # 特徴量設定
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None

    # アラート設定
    alert_enabled: bool = True
    alert_severity_threshold: str = "medium"

    # ログ設定
    log_level: str = "INFO"
    log_drift_events: bool = True

    # パフォーマンス設定
    max_memory_mb: int = 100
    parallel_detection: bool = True

    # カスタム設定
    custom_params: Dict[str, Any] = field(default_factory=dict)
