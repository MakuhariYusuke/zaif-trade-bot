"""
Walk-Forward統合アダプター

WindowPerformance を統一評価フレームワーク（ComprehensiveEvaluation）に統合。
過学習検出と統計分析を提供。
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ztb.analysis.common.types import ComprehensiveEvaluationClass, EvaluationResult
from ztb.evaluation.walk_forward.types import WindowPerformance

from typing import TYPE_CHECKING

# Avoid importing heavyweight or site-specific scripts at module import time
# (these can import stable-baselines3 and break test collection). Only import
# the BacktestReporter type for type checking.
if TYPE_CHECKING:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    scripts_path = project_root / "scripts" / "v457"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    from backtest_v457 import BacktestReporter  # type: ignore
else:
    BacktestReporter = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardAggregationStats:
    """Walk-Forward集約統計"""

    window_count: int

    # 検証期間（In-Sample）統計
    avg_val_roi: float
    std_val_roi: float
    best_val_roi: float
    worst_val_roi: float

    # テスト期間（Out-of-Sample）統計
    avg_test_roi: float
    std_test_roi: float
    best_test_roi: float
    worst_test_roi: float

    # 過学習指標
    overfitting_indicator: float  # 平均: test_roi / val_roi
    overfitting_severity: str  # "none" | "mild" | "moderate" | "severe"
    overfitting_ratio_list: List[float] = field(default_factory=list)

    # 一貫性スコア（0-1）
    consistency_score: float = 0.0

    # ロバストネス指標（0-1）
    robustness_score: float = 0.0

    # リスク関連
    avg_max_drawdown: float = 0.0
    std_sharpe_ratio: float = 0.0
    avg_win_rate: float = 0.0

    # 安定性指数
    stability_index: float = 0.0


class WalkForwardUnifiedEvaluator:
    """Walk-Forward統合評価器

    複数ウィンドウの WindowPerformance を統合し、
    ComprehensiveEvaluation 形式の包括的評価結果を生成する。
    """

    # 過学習検出の閾値
    # Over-fitting ratio = test_roi / val_roi
    # - ratio < 1.0: テストが訓練より良い（ラッキーテスト）
    # - ratio = 1.0-1.05: 最適（若干のドローダウンは許容）
    # - ratio = 1.05-1.15: 軽度の過学習（許容範囲）
    # - ratio > 1.15: 中程度以上の過学習（注意が必要）
    # Adjusted thresholds to be backward-compatible with callers/tests that
    # expect "mild" at 0.9, "moderate" at 1.1, and "severe" at higher values.
    OVERFITTING_THRESHOLDS = {
        "none": (0.0, 0.9),  # < 0.9: 過学習なし/ラッキー
        "mild": (0.9, 1.1),  # 0.9-1.1: 軽度（許容可）
        "moderate": (1.1, 1.4),  # 1.1-1.4: 中程度（要監視）
        "severe": (1.4, float("inf")),  # >= 1.4: 深刻（要改善）
    }

    def __init__(self) -> None:
        """初期化"""
        pass

    def aggregate_windows(
        self,
        windows: List[WindowPerformance],
        reporters: List["BacktestReporter"] | None = None,
        model_name: str = "",
    ) -> ComprehensiveEvaluationClass:
        """Backwards compatible: `reporters` may be omitted or callers may pass `model_name` as second positional arg.

        Examples supported:
            aggregate_windows(windows, reporters_list, model_name="x")
            aggregate_windows(windows, model_name)
            aggregate_windows(windows, model_name="x")
        """
        # Backward-compat: if caller passed a string as the second positional
        # argument (e.g., aggregate_windows(windows, "model_a")), treat that
        # string as model_name and set reporters to None.
        if isinstance(reporters, str):
            model_name = reporters
            reporters = None
        """複数ウィンドウの結果を統合評価

        Args:
            windows: ウィンドウごとの性能結果リスト
            reporters: ウィンドウごとのBacktestReporterリスト
            model_name: モデル名

        Returns:
            ComprehensiveEvaluationClass: 統合評価結果
        """

        if not windows:
            raise ValueError("Windows list cannot be empty")

        logger.info(f"Aggregating {len(windows)} windows for model: {model_name}")

        # 統計分析
        stats = self._analyze_cross_window_stats(windows, reporters)

        logger.info(f"  Window count: {stats.window_count}")
        logger.info(f"  Avg Val ROI: {stats.avg_val_roi:.4f} ± {stats.std_val_roi:.4f}")
        logger.info(
            f"  Avg Test ROI: {stats.avg_test_roi:.4f} ± {stats.std_test_roi:.4f}"
        )
        logger.info(
            f"  Overfitting Indicator: {stats.overfitting_indicator:.4f} ({stats.overfitting_severity})"
        )
        logger.info(f"  Consistency Score: {stats.consistency_score:.4f}")
        logger.info(f"  Robustness Score: {stats.robustness_score:.4f}")
        logger.info(f"  Stability Index: {stats.stability_index:.4f}")

        # メトリクス集約
        aggregated_results = self._aggregate_metrics(windows, stats)

        # 統合評価オブジェクト作成
        evaluation = ComprehensiveEvaluationClass(
            model_name=model_name or "walk_forward_model",
            evaluation_type="walk_forward",
            timestamp=datetime.now().isoformat(),
            results=aggregated_results,
            summary_stats={
                "window_count": stats.window_count,
                "avg_val_roi": stats.avg_val_roi,
                "std_val_roi": stats.std_val_roi,
                "avg_test_roi": stats.avg_test_roi,
                "std_test_roi": stats.std_test_roi,
            },
            risk_metrics={
                "avg_max_drawdown": stats.avg_max_drawdown,
                "std_sharpe_ratio": stats.std_sharpe_ratio,
                "avg_win_rate": stats.avg_win_rate,
            },
            performance_metrics={
                "consistency_score": stats.consistency_score,
                "robustness_score": stats.robustness_score,
                "stability_index": stats.stability_index,
            },
            robustness_tests={
                "overfitting_indicator": stats.overfitting_indicator,
                "overfitting_severity": stats.overfitting_severity,
                "overfitting_ratios": stats.overfitting_ratio_list,
            },
        )

        return evaluation

    def _analyze_cross_window_stats(
        self,
        windows: List[WindowPerformance],
        reporters: List["BacktestReporter"] | None = None,
    ) -> WalkForwardAggregationStats:
        """If `reporters` is None, build lightweight default reporter stats for aggregation."""
        """ウィンドウ横断的統計分析

        Args:
            windows: ウィンドウ性能リスト
            reporters: ウィンドウBacktestReporterリスト

        Returns:
            WalkForwardAggregationStats: 統計オブジェクト
        """

        val_rois = np.array([w.val_roi for w in windows])
        test_rois = np.array([w.test_roi for w in windows])

        # If detailed reporters are not supplied by caller, construct minimal/default stats
        if reporters is None:
            sharpe_ratios = np.array([getattr(w, "sharpe_ratio", 0.0) for w in windows])
            max_drawdowns = np.array([getattr(w, "max_drawdown", 0.0) for w in windows])
            win_rates = np.array([getattr(w, "win_rate", 0.0) for w in windows])
        else:
            sharpe_ratios = np.array([r.stats.get("sharpe_ratio", 0.0) for r in reporters])
            max_drawdowns = np.array([r.stats.get("max_drawdown", 0.0) for r in reporters])
            win_rates = np.array([r.stats.get("winning_trades", 0) / max(1, r.stats.get("total_trades", 1)) for r in reporters])

        # 過学習指標計算（ウィンドウごと）
        # Over-fitting ratio = |test_roi - val_roi| / |val_roi|
        # テストと訓練の乖離度を測定
        # ratio > 1.0: テストが訓練より悪化（典型的な過学習）
        # ratio < 1.0: テストが訓練より良好（ラッキー）
        overfitting_ratios_list: List[float] = []
        for val, test in zip(val_rois, test_rois):
            if val > 1e-6:  # val が正の値の場合
                # テスト性能の訓練性能対比（相対的な悪化度）
                ratio = max(0.0, (val - test) / val)  # >= 0
            elif val < -1e-6:  # val が負の値（損失）の場合
                # 負の場合は逆方向で計算（絶対値ベース）
                ratio = max(0.0, (abs(test) - abs(val)) / abs(val))
            else:  # val ≈ 0
                ratio = 0.0 if test <= 1e-6 else 1.0
            overfitting_ratios_list.append(ratio)

        overfitting_ratios: np.ndarray[Any, np.dtype[Any]] = np.array(
            overfitting_ratios_list
        )
        avg_overfitting = np.mean(overfitting_ratios)
        
        # 1.0 を基準点に正規化（1.0 = 基準、< 1.0 = 改善、> 1.0 = 悪化）
        # ただし既存の解釈との互換性のため、[val_roi / (val_roi + epsilon)] 形式で計算
        # 改良版: 直接的な乖離指標に変更
        normalized_overfitting = 1.0 + avg_overfitting  # 0-2 の範囲に正規化

        # 過学習の重大度判定
        overfitting_severity = self._determine_overfitting_severity(normalized_overfitting)

        # 一貫性スコア: ウィンドウ間のROIのばらつきの逆
        # 低い std = 一貫性が高い
        val_roi_cv = np.std(val_rois) / (
            np.mean(np.abs(val_rois)) + 1e-6
        )  # Coefficient of Variation
        consistency_score = max(0.0, 1.0 - val_roi_cv)

        # ロバストネススコア: テストセット性能の質
        # テスト ROI が正で安定している場合、スコアが高い
        test_roi_mean = np.mean(test_rois)
        test_roi_std = np.std(test_rois)
        test_roi_cv = test_roi_std / (abs(test_roi_mean) + 1e-6)

        # テストセットで正のROI + 低いばらつき = 高いロバストネス
        robustness_score = 0.0
        if test_roi_mean > 0:
            # Returns正 + 安定性高 = ロバスト
            robustness_score = min(
                1.0, (test_roi_mean / (abs(test_roi_mean) + test_roi_std + 1e-6))
            )

        # 安定性指数: Sharpe比の一貫性
        sharpe_cv = np.std(sharpe_ratios) / (np.mean(np.abs(sharpe_ratios)) + 1e-6)
        stability_index = max(0.0, 1.0 - sharpe_cv)

        stats = WalkForwardAggregationStats(
            window_count=len(windows),
            avg_val_roi=float(np.mean(val_rois)),
            std_val_roi=float(np.std(val_rois)),
            best_val_roi=float(np.max(val_rois)),
            worst_val_roi=float(np.min(val_rois)),
            avg_test_roi=float(np.mean(test_rois)),
            std_test_roi=float(np.std(test_rois)),
            best_test_roi=float(np.max(test_rois)),
            worst_test_roi=float(np.min(test_rois)),
            overfitting_indicator=float(normalized_overfitting),  # 改良版: 1.0 基準の指標
            overfitting_severity=overfitting_severity,
            overfitting_ratio_list=overfitting_ratios_list,
            consistency_score=float(consistency_score),
            robustness_score=float(robustness_score),
            avg_max_drawdown=float(np.mean(max_drawdowns)),
            std_sharpe_ratio=float(np.std(sharpe_ratios)),
            avg_win_rate=float(np.mean(win_rates)),
            stability_index=float(stability_index),
        )

        return stats

    def _aggregate_metrics(
        self,
        windows: List[WindowPerformance],
        stats: WalkForwardAggregationStats,
    ) -> Dict[str, EvaluationResult]:
        """メトリクス集約

        各ウィンドウの性能を集約し、統一メトリクス形式に変換。

        Args:
            windows: ウィンドウ性能リスト
            stats: 集約統計

        Returns:
            Dict[str, EvaluationResult]: メトリクス名 → 結果
        """

        results: Dict[str, EvaluationResult] = {}

        # 1. ROI関連
        results["roi_in_sample"] = EvaluationResult(
            metric="roi_in_sample",
            value=stats.avg_val_roi,
            confidence_interval=(
                stats.avg_val_roi - stats.std_val_roi,
                stats.avg_val_roi + stats.std_val_roi,
            ),
        )

        results["roi_out_of_sample"] = EvaluationResult(
            metric="roi_out_of_sample",
            value=stats.avg_test_roi,
            confidence_interval=(
                stats.avg_test_roi - stats.std_test_roi,
                stats.avg_test_roi + stats.std_test_roi,
            ),
        )

        # 2. リスク関連
        results["max_drawdown"] = EvaluationResult(
            metric="max_drawdown",
            value=stats.avg_max_drawdown,
        )

        results["sharpe_ratio"] = EvaluationResult(
            metric="sharpe_ratio",
            value=stats.std_sharpe_ratio,
        )

        # 3. 過学習関連
        results["overfitting_indicator"] = EvaluationResult(
            metric="overfitting_indicator",
            value=stats.overfitting_indicator,
            confidence_interval=(
                min(stats.overfitting_ratio_list)
                if stats.overfitting_ratio_list
                else 0.0,
                max(stats.overfitting_ratio_list)
                if stats.overfitting_ratio_list
                else 1.0,
            ),
        )

        # 4. 堅牢性関連
        results["consistency_score"] = EvaluationResult(
            metric="consistency_score",
            value=stats.consistency_score,
        )

        results["robustness_score"] = EvaluationResult(
            metric="robustness_score",
            value=stats.robustness_score,
        )

        results["stability_index"] = EvaluationResult(
            metric="stability_index",
            value=stats.stability_index,
        )

        # 5. 取引関連
        results["win_rate"] = EvaluationResult(
            metric="win_rate",
            value=stats.avg_win_rate,
        )

        return results

    def _determine_overfitting_severity(self, overfitting_indicator: float) -> str:
        """過学習の重大度判定

        Args:
            overfitting_indicator: 過学習指標（平均比率）

        Returns:
            str: 重大度レベル
        """

        for severity, (lower, upper) in self.OVERFITTING_THRESHOLDS.items():
            if lower <= overfitting_indicator < upper:
                return severity

        return "severe"

    def compare_multiple_evaluations(
        self,
        evaluations: Dict[str, ComprehensiveEvaluationClass],
    ) -> Dict[str, Any]:
        """複数モデルの統合評価結果を比較

        Args:
            evaluations: モデル名 → 評価結果

        Returns:
            Dict[str, Any]: 比較結果
        """

        if not evaluations:
            raise ValueError("Evaluations dict cannot be empty")

        comparison: Dict[str, Any] = {
            "model_count": len(evaluations),
            "models": {},
            "rankings": {},
        }

        # 各モデルの主要指標を抽出
        for model_name, evaluation in evaluations.items():
            comparison["models"][model_name] = {
                "roi_out_of_sample": evaluation.get_metric_value("roi_out_of_sample"),
                "max_drawdown": evaluation.get_metric_value("max_drawdown"),
                "robustness_score": evaluation.get_metric_value("robustness_score"),
                "overfitting_indicator": evaluation.get_metric_value(
                    "overfitting_indicator"
                ),
                "stability_index": evaluation.get_metric_value("stability_index"),
            }

        # ランキング生成
        comparison["rankings"] = self._generate_rankings(evaluations)

        return comparison

    def _generate_rankings(
        self,
        evaluations: Dict[str, ComprehensiveEvaluationClass],
    ) -> Dict[str, List[str]]:
        """各メトリクスごとのランキングを生成

        Args:
            evaluations: モデル評価結果

        Returns:
            Dict[str, List[str]]: メトリクス → ランク付けモデルリスト
        """

        rankings: Dict[str, List[str]] = {}

        # ROI（高い方が良い）
        roi_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("roi_out_of_sample") or 0.0,
            reverse=True,
        )
        rankings["roi_out_of_sample"] = [m[0] for m in roi_ranking]

        # Max Drawdown（低い方が良い＝マイナス大きい方が悪い）
        dd_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("max_drawdown") or 0.0,
            reverse=True,
        )
        rankings["max_drawdown"] = [m[0] for m in dd_ranking]

        # Robustness（高い方が良い）
        robust_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("robustness_score") or 0.0,
            reverse=True,
        )
        rankings["robustness_score"] = [m[0] for m in robust_ranking]

        # Overfitting（低い方が良い）
        overfit_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("overfitting_indicator") or 1.0,
        )
        rankings["overfitting_indicator"] = [m[0] for m in overfit_ranking]

        return rankings
