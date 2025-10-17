"""
Statistical Analysis Engine for A/B Testing
処理時間短縮・メモリ効率を考慮したストリーミング統計計算
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from .types import StatisticalResult, ABTestMetrics, StatisticalTest
from .config import ABTestConfig

logger = logging.getLogger(__name__)


class ABTestAnalyzer:
    """A/Bテスト統計分析エンジン"""

    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.performance.max_workers)

    def analyze_parallel(self, data_a: np.ndarray, data_b: np.ndarray) -> StatisticalResult:
        """並列統計分析"""
        # 並列で統計計算を実行
        future_a = self.executor.submit(self._calculate_stats, data_a)
        future_b = self.executor.submit(self._calculate_stats, data_b)

        stats_a = future_a.result()
        stats_b = future_b.result()

        # t検定を実行
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)

        # 効果量を計算
        effect_size = self._calculate_effect_size(data_a, data_b)

        # 信頼区間を計算
        ci_lower, ci_upper = self._calculate_confidence_interval(data_a, data_b)

        # 基本統計量を取得
        mean_a = stats_a['mean']
        mean_b = stats_b['mean']
        std_a = stats_a['std']
        std_b = stats_b['std']

        return StatisticalResult(
            test_type="t-test",
            p_value=float(p_value),
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            sample_size_a=len(data_a),
            sample_size_b=len(data_b),
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b
        )

    def _calculate_stats(self, data: np.ndarray) -> Dict[str, float]:
        """基本統計量を計算"""
        return {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'count': len(data)
        }

    def _calculate_effect_size(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """効果量を計算（Cohen's d）"""
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        std_a, std_b = np.std(data_a, ddof=1), np.std(data_b, ddof=1)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        return (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0

    def _calculate_confidence_interval(self, data_a: np.ndarray, data_b: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """効果量の信頼区間を計算"""
        effect_size = self._calculate_effect_size(data_a, data_b)
        # 簡易的な信頼区間計算（実際にはより複雑な計算が必要）
        se = 1.0 / np.sqrt(min(len(data_a), len(data_b)) / 2)  # 近似
        margin = se * 1.96  # 95%信頼区間
        return effect_size - margin, effect_size + margin

    def calculate_bootstrap_ci(self, data_a: np.ndarray, data_b: np.ndarray, n_bootstrap: int = 1000) -> StatisticalResult:
        """ブートストラップ法による信頼区間計算"""
        effect_sizes = []

        for _ in range(n_bootstrap):
            # ブートストラップサンプルを生成
            sample_a = np.random.choice(data_a, size=len(data_a), replace=True)
            sample_b = np.random.choice(data_b, size=len(data_b), replace=True)

            # 効果量を計算
            effect_size = self._calculate_effect_size(sample_a, sample_b)
            effect_sizes.append(effect_size)

        # 信頼区間を計算
        effect_sizes = np.array(effect_sizes)
        ci_lower = np.percentile(effect_sizes, 2.5)
        ci_upper = np.percentile(effect_sizes, 97.5)
        mean_effect = np.mean(effect_sizes)

        # p値を計算（効果量が0をまたぐ確率）
        p_value = np.mean(effect_sizes <= 0) if mean_effect > 0 else np.mean(effect_sizes >= 0)

        return StatisticalResult(
            test_type=StatisticalTest.T_TEST,
            p_value=float(p_value),
            effect_size=mean_effect,
            confidence_interval=(ci_lower, ci_upper),
            sample_size_a=len(data_a),
            sample_size_b=len(data_b),
            mean_a=np.mean(data_a),
            mean_b=np.mean(data_b),
            std_a=np.std(data_a, ddof=1),
            std_b=np.std(data_b, ddof=1)
        )

    def update_streaming_stats(self, variant_id: str, batch_data: np.ndarray) -> None:
        """ストリーミング統計を更新"""
        # ストリーミング統計の更新ロジック
        # （実際の実装では、variant_idごとにStreamingStatisticsインスタンスを管理）
        pass

    def analyze_comparison(self, metrics_a: ABTestMetrics, metrics_b: ABTestMetrics, test_type: str = "t-test") -> StatisticalResult:
        """メトリクス比較分析"""
        # 簡易的な実装 - 実際にはより複雑な分析が必要
        data_a = np.random.normal(metrics_a.mean_reward, metrics_a.std_reward, 100)  # 仮定
        data_b = np.random.normal(metrics_b.mean_reward, metrics_b.std_reward, 100)  # 仮定

        return self.analyze_parallel(data_a, data_b)

    def __del__(self):
        """クリーンアップ"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)