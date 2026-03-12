"""
Walk-Forward評価パイプライン統合

ztb.evaluation.walk_forward_evaluator (WindowPerformance生成)
↓
ztb.analysis.evaluation.walk_forward_adapter (WalkForwardUnifiedEvaluator)
↓
ComprehensiveEvaluation (統一評価フレームワーク出力)
↓
統計分析・レポート生成

このモジュールは、walk_forwardの評価結果をunified_evaluationフレームワークに
統合するための実行エントリーポイントを提供します。
"""

import logging
from pathlib import Path
from typing import Any
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from ztb.io.json_io import write_json
from ztb.evaluation.walk_forward.types import WindowPerformance
from ztb.analysis.evaluation.walk_forward_adapter import WalkForwardUnifiedEvaluator
from ztb.analysis.common.types import ComprehensiveEvaluationClass
from ztb.analysis.baseline_comparison import BaselineComparisonEngine, BaselineResult

from typing import TYPE_CHECKING

# Avoid importing heavyweight scripts at module import time. Import the
# BacktestReporter type only for static type checking.
if TYPE_CHECKING:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    scripts_path = project_root / "scripts" / "v457"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    from backtest_v457 import BacktestReporter  # type: ignore
else:
    BacktestReporter = None  # type: ignore

logger = logging.getLogger(__name__)

class WalkForwardEvaluationPipeline:
    """Walk-Forward結果を統一評価フレームワークに統合
    
    パイプラインの流れ：
    1. WindowPerformanceリストを入力
    2. WalkForwardUnifiedEvaluatorで集約
    3. ComprehensiveEvaluationを出力
    4. JSON保存 + 統計レポート生成
    """
    
    def __init__(
        self,
        model_name: str = "walk_forward_model",
        output_dir: str = "results/phase4/evaluation",
    ) -> None:
        """初期化
        
        Args:
            model_name: モデル名
            output_dir: 出力ディレクトリ
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = WalkForwardUnifiedEvaluator()
        self.evaluation_result: ComprehensiveEvaluationClass | None = None
    def run_evaluation(self, walk_forward_result: 'WalkForwardResult', reporter: 'WalkForwardReporter') -> 'ComprehensiveEvaluationClass':
        """WalkForwardResultから評価を実行
        
        Args:
            walk_forward_result: Walk-Forward評価結果
            reporter: WalkForwardReporter
            
        Returns:
            ComprehensiveEvaluationClass: 評価結果
        """
        from ztb.analysis.common.types import ComprehensiveEvaluationClass
        
        # Create evaluation result
        self.evaluation_result = ComprehensiveEvaluationClass(
            model_name=self.model_name,
            roi_out_of_sample=walk_forward_result.average_test_roi,
            sharpe_ratio=walk_forward_result.average_sharpe,
            max_drawdown=reporter.stats.get("max_drawdown", 0.0),
            win_rate=reporter.stats.get("win_rate", 0.0),
            total_trades=reporter.stats.get("total_trades", 0),
            summary_stats={
                "average_test_roi": walk_forward_result.average_test_roi,
                "average_sharpe": walk_forward_result.average_sharpe,
                "overfitting_ratio": walk_forward_result.overfitting_ratio,
                "num_windows": walk_forward_result.num_windows,
            }
        )
        
        return self.evaluation_result        
        logger.info(f"Pipeline initialized: {model_name}")
    
    def integrate_walk_forward_results(
        self,
        windows: list[WindowPerformance],
        reporters: list["BacktestReporter"] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ComprehensiveEvaluationClass:
        """Walk-Forward結果を統合評価に変換
        
        Args:
            windows: ウィンドウ別性能リスト
            reporters: ウィンドウ別BacktestReporterリスト
            model_metadata: モデルメタデータ（オプション）
            
        Returns:
            ComprehensiveEvaluationClass: 統合評価結果
        """
        
        logger.info(f"Integrating {len(windows)} windows")
        
        # 集約評価を生成
        self.evaluation_result = self.evaluator.aggregate_windows(
            windows=windows,
            reporters=reporters,
            model_name=self.model_name,
        )
        
        # メタデータを追加（存在する場合）
        if model_metadata:
            self.evaluation_result.summary_stats.update(model_metadata)
        
        return self.evaluation_result
    
    def save_evaluation(
        self,
        format: str = "json",
    ) -> Path:
        """評価結果を保存
        
        Args:
            format: 保存形式 ("json" | "yaml" | "pickle")
            
        Returns:
            Path: 保存ファイルパス
        """
        
        if self.evaluation_result is None:
            raise ValueError("Evaluation result not available. Call integrate_walk_forward_results() first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            output_file = self.output_dir / f"walk_forward_evaluation_{timestamp}.json"
            data = self.evaluation_result.to_dict()
            write_json(output_file, data, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation saved: {output_file}")
            return output_file
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_summary_report(self) -> str:
        """評価結果の要約レポートを生成
        
        Returns:
            str: レポートテキスト
        """
        
        if self.evaluation_result is None:
            raise ValueError("Evaluation result not available.")
        
        # メトリクス抽出
        roi_in_sample = self.evaluation_result.get_metric_value("roi_in_sample")
        roi_out_of_sample = self.evaluation_result.get_metric_value("roi_out_of_sample")
        max_drawdown = self.evaluation_result.get_metric_value("max_drawdown")
        sharpe_ratio = self.evaluation_result.get_metric_value("sharpe_ratio")
        consistency = self.evaluation_result.get_metric_value("consistency_score")
        robustness = self.evaluation_result.get_metric_value("robustness_score")
        overfitting = self.evaluation_result.get_metric_value("overfitting_indicator")
        
        # ロバストネス判定
        if robustness is not None:
            if robustness > 0.8:
                robustness_status = "✅ EXCELLENT"
            elif robustness > 0.6:
                robustness_status = "✅ GOOD"
            elif robustness > 0.4:
                robustness_status = "⚠️  MODERATE"
            else:
                robustness_status = "❌ POOR"
        else:
            robustness_status = "UNKNOWN"
        
        # 過学習判定
        overfitting_severity = self.evaluation_result.robustness_tests.get("overfitting_severity", "UNKNOWN")
        
        report = f"""
═══════════════════════════════════════════════════════════════
Walk-Forward統合評価レポート
═══════════════════════════════════════════════════════════════

【モデル情報】
  モデル名: {self.evaluation_result.model_name}
  評価タイプ: {self.evaluation_result.evaluation_type}
  タイムスタンプ: {self.evaluation_result.timestamp}

【パフォーマンス】
  In-Sample ROI:  {roi_in_sample:>10.4%} (訓練期間)
  Out-of-Sample ROI: {roi_out_of_sample:>6.4%} (テスト期間)
  最大ドローダウン: {max_drawdown:>6.4%}
  Sharpe比: {sharpe_ratio:>10.4f}

【堅牢性評価】
  一貫性スコア: {consistency:>6.4f} (ウィンドウ間一貫性)
  ロバストネス: {robustness:>6.4f} {robustness_status}
  安定性指数: {self.evaluation_result.get_metric_value("stability_index"):>6.4f}

【過学習検出】
  過学習指標: {overfitting:>8.4f}
  重大度: {overfitting_severity}
  
  解釈:
    - 指標 < 0.8: 過学習なし（堅牢）✅
    - 指標 0.8-1.0: 軽度の過学習 ⚠️
    - 指標 1.0-1.2: 中程度の過学習 ⚠️
    - 指標 > 1.2: 深刻な過学習 ❌

【取引統計】
  平均勝率: {self.evaluation_result.get_metric_value("win_rate"):>6.4%}

【推奨事項】
"""
        
        # 推奨事項の生成
        recommendations = self._generate_recommendations(
            roi_in_sample,
            roi_out_of_sample,
            robustness,
            overfitting,
        )
        
        for rec in recommendations:
            report += f"  • {rec}\n"
        
        report += "═══════════════════════════════════════════════════════════════\n"
        
        return report
    
    def _extract_model_result_from_evaluation(self, evaluation_result) -> BaselineResult:
        """Extract BaselineResult from ComprehensiveEvaluation for baseline comparison."""
        return BaselineResult(
            strategy_name=evaluation_result.model_name,
            total_return=evaluation_result.roi_out_of_sample,
            sharpe_ratio=evaluation_result.sharpe_ratio,
            max_drawdown=evaluation_result.max_drawdown,
            win_rate=evaluation_result.win_rate,
            total_trades=evaluation_result.total_trades,
            metrics=evaluation_result.summary_stats,
        )
    
    def _generate_recommendations(
        self,
        roi_in: float | None,
        roi_out: float | None,
        robustness: float | None,
        overfitting: float | None,
    ) -> list[str]:
        """推奨事項の自動生成
        
        Args:
            roi_in: In-Sample ROI
            roi_out: Out-of-Sample ROI
            robustness: ロバストネススコア
            overfitting: 過学習指標
            
        Returns:
            list[str]: 推奨事項のリスト
        """
        
        recommendations: list[str] = []
        
        # パフォーマンス評価
        if roi_out is not None:
            if roi_out > 0.10:
                recommendations.append("優秀な性能: 本格運用を検討")
            elif roi_out > 0.05:
                recommendations.append("良好な性能: さらなる最適化を推奨")
            elif roi_out > 0.0:
                recommendations.append("可能性あり: パラメータ調整で改善可能")
            else:
                recommendations.append("性能改善が必要: モデルアーキテクチャの見直し検討")
        
        # 過学習評価
        if overfitting is not None:
            if overfitting > 1.2:
                recommendations.append("⚠️  深刻な過学習: 正則化強化 / データ拡張を実施")
            elif overfitting > 1.0:
                recommendations.append("中程度の過学習: ハイパーパラメータ再調整推奨")
            elif overfitting > 0.8:
                recommendations.append("軽度の過学習: 監視継続")
        
        # ロバストネス評価
        if robustness is not None:
            if robustness < 0.5:
                recommendations.append("ロバストネス低下: ウィンドウ設定の見直し検討")
        
        # ROI差分評価
        if roi_in is not None and roi_out is not None:
            roi_diff = roi_in - roi_out
            if roi_diff > 0.05:
                recommendations.append(f"In/Out-of-Sample差が大きい ({roi_diff:.2%}): 過学習の可能性")
        
        if not recommendations:
            recommendations.append("フレームワークは基準を満たしています")
        
        return recommendations
    
    def generate_full_report(self) -> str:
        """完全なレポートを生成（サマリー + 詳細統計）
        
        Returns:
            str: 完全レポート
        """
        
        if self.evaluation_result is None:
            raise ValueError("Evaluation result not available.")
        
        # サマリーレポート
        summary = self.generate_summary_report()
        
        # 詳細統計
        details = "\n【詳細統計】\n"
        
        if self.evaluation_result.summary_stats:
            details += "  In-Sample統計:\n"
            if "avg_val_roi" in self.evaluation_result.summary_stats:
                avg_val_roi = self.evaluation_result.summary_stats["avg_val_roi"]
                std_val_roi = self.evaluation_result.summary_stats.get("std_val_roi", 0.0)
                details += f"    平均: {avg_val_roi:.4%}, 標準偏差: {std_val_roi:.4%}\n"
        
        if self.evaluation_result.summary_stats:
            details += "  Out-of-Sample統計:\n"
            if "avg_test_roi" in self.evaluation_result.summary_stats:
                avg_test_roi = self.evaluation_result.summary_stats["avg_test_roi"]
                std_test_roi = self.evaluation_result.summary_stats.get("std_test_roi", 0.0)
                details += f"    平均: {avg_test_roi:.4%}, 標準偏差: {std_test_roi:.4%}\n"
        
        return summary + details
    
    def compare_with_baselines(
        self,
        walk_forward_result: 'WalkForwardResult',
        price_data: pd.DataFrame,
        baseline_strategies: list[str] | None = None,
    ) -> dict[str, Any]:
        """ベースライン戦略との比較
        
        Args:
            price_data: 価格データ
            baseline_strategies: 比較するベースライン戦略リスト
            
        Returns:
            dict[str, Any]: 比較結果
        """
        if self.evaluation_result is None:
            raise ValueError("Evaluation result not available.")
        
        if baseline_strategies is None:
            baseline_strategies = ["buy_hold", "sma_crossover"]
        
        # BaselineComparisonEngineで比較
        engine = BaselineComparisonEngine()
        
        # モデルの結果をBaselineResult形式に変換
        from ztb.analysis.baseline_comparison import BaselineResult
        model_result = BaselineResult(
            strategy_name=self.model_name,
            total_return=self.evaluation_result.get_metric_value("roi_out_of_sample", 0.0),
            sharpe_ratio=self.evaluation_result.get_metric_value("sharpe_ratio", 0.0),
            max_drawdown=self.evaluation_result.get_metric_value("max_drawdown", 0.0),
            win_rate=self.evaluation_result.get_metric_value("win_rate", 0.0),
            total_trades=self.evaluation_result.get_metric_value("total_trades", 0),
            metrics=self.evaluation_result.summary_stats,
        )
        
        # ベースライン評価
        comparison_report = engine.compare(
            model_result=model_result,
            price_data=price_data,
            strategies=baseline_strategies,
        )
        
        # ComparisonReportをdictに変換
        comparison = {
            "model_result": {
                "strategy_name": comparison_report.model_result.strategy_name,
                "total_return": comparison_report.model_result.total_return,
                "sharpe_ratio": comparison_report.model_result.sharpe_ratio,
                "max_drawdown": comparison_report.model_result.max_drawdown,
                "win_rate": comparison_report.model_result.win_rate,
                "total_trades": comparison_report.model_result.total_trades,
            },
            "baseline_results": [
                {
                    "strategy_name": b.strategy_name,
                    "total_return": b.total_return,
                    "sharpe_ratio": b.sharpe_ratio,
                    "max_drawdown": b.max_drawdown,
                    "win_rate": b.win_rate,
                    "total_trades": b.total_trades,
                }
                for b in comparison_report.baseline_results
            ],
            "superiority_metrics": comparison_report.superiority_metrics,
            "statistical_tests": comparison_report.statistical_tests,
        }
        
        return comparison

    def compare_with_baseline(
        self,
        baseline_evaluation: ComprehensiveEvaluationClass,
    ) -> dict[str, Any]:
        """現在の評価とベースライン評価（ComprehensiveEvaluationClass）を比較します。

        Args:
            baseline_evaluation: 比較対象の ComprehensiveEvaluationClass

        Returns:
            dict[str, Any]: 'model', 'baseline', 'metrics' を含む比較結果
        """
        if self.evaluation_result is None:
            raise ValueError("Evaluation result not available.")
        if baseline_evaluation is None:
            raise ValueError("baseline_evaluation is required.")

        model_eval = self.evaluation_result
        base = baseline_evaluation

        def mk_metric(name: str):
            m = model_eval.get_metric_value(name, None)
            b = base.get_metric_value(name, None)
            diff = None
            if (m is not None) and (b is not None):
                try:
                    diff = m - b
                except Exception:
                    diff = None
            return {"metric": name, "model": m, "baseline": b, "diff": diff}

        metric_names = ["roi_out_of_sample", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"]
        metrics = [mk_metric(n) for n in metric_names if (mk_metric(n)["model"] is not None) or (mk_metric(n)["baseline"] is not None)]

        comparison = {
            "model": model_eval.model_name,
            "baseline": base.model_name,
            "metrics": metrics,
        }
        return comparison

    @staticmethod
    def compare_multiple_evaluations(
        walk_forward_results: list['WalkForwardResult'],
    ) -> list[str]:
        """複数WalkForwardResultの比較（A/Bテスト）
        
        Args:
            walk_forward_results: Walk-Forward評価結果のリスト
            
        Returns:
            list[str]: 比較結果のサマリー文字列リスト
        """
        if len(walk_forward_results) < 2:
            return ["Need at least 2 results for comparison"]
        
        results = []
        for i, result1 in enumerate(walk_forward_results):
            for j, result2 in enumerate(walk_forward_results):
                if i < j:
                    roi_diff = result1.average_test_roi - result2.average_test_roi
                    sharpe_diff = result1.average_sharpe - result2.average_sharpe
                    results.append(f"Result{i} vs Result{j}: ROI diff={roi_diff:.4f}, Sharpe diff={sharpe_diff:.4f}")
        
        return results
    
    def _rank_evaluations(
        self,
        evaluations: dict[str, ComprehensiveEvaluationClass],
    ) -> dict[str, list[str]]:
        """評価結果をメトリクスごとにランキング
        
        Args:
            evaluations: モデル名 → 評価結果
            
        Returns:
            dict[str, list[str]]: メトリクス → ランク付けモデルリスト
        """
        rankings = {}
        
        # ROI（高い方が良い）
        roi_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("roi_out_of_sample") or 0.0,
            reverse=True,
        )
        rankings["roi_out_of_sample"] = [m[0] for m in roi_ranking]
        
        # Sharpe（高い方が良い）
        sharpe_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("sharpe_ratio") or 0.0,
            reverse=True,
        )
        rankings["sharpe_ratio"] = [m[0] for m in sharpe_ranking]
        
        # Max Drawdown（低い方が良い）
        dd_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("max_drawdown") or 0.0,
        )
        rankings["max_drawdown"] = [m[0] for m in dd_ranking]
        
        # Win Rate（高い方が良い）
        win_rate_ranking = sorted(
            evaluations.items(),
            key=lambda x: x[1].get_metric_value("win_rate") or 0.0,
            reverse=True,
        )
        rankings["win_rate"] = [m[0] for m in win_rate_ranking]
        
        return rankings
    
    def _run_ab_tests(
        self,
        evaluations: dict[str, ComprehensiveEvaluationClass],
    ) -> dict[str, Any]:
        """A/Bテストの統計的有意性検定
        
        Args:
            evaluations: 評価結果
            
        Returns:
            dict[str, Any]: 検定結果
        """
        # 簡易版：各メトリクスの平均比較
        tests = {}
        
        if len(evaluations) < 2:
            return tests
        
        # 基準モデル（最初のもの）
        baseline_name = list(evaluations.keys())[0]
        baseline = evaluations[baseline_name]
        
        for name, eval_result in evaluations.items():
            if name == baseline_name:
                continue
            
            comparison = {
                "vs_baseline": baseline_name,
                "roi_diff": eval_result.get_metric_value("roi_out_of_sample", 0.0) - baseline.get_metric_value("roi_out_of_sample", 0.0),
                "sharpe_diff": eval_result.get_metric_value("sharpe_ratio", 0.0) - baseline.get_metric_value("sharpe_ratio", 0.0),
                "dd_diff": eval_result.get_metric_value("max_drawdown", 0.0) - baseline.get_metric_value("max_drawdown", 0.0),
                "win_rate_diff": eval_result.get_metric_value("win_rate", 0.0) - baseline.get_metric_value("win_rate", 0.0),
            }
            tests[name] = comparison
        
        return tests

if __name__ == "__main__":
    main()
    """メイン実行用のサンプル
    
    実際の使用例：
    1. walk_forwardから得たWindowPerformanceリストを用意
    2. WalkForwardEvaluationPipelineをインスタンス化
    3. integrate_walk_forward_results()で統合
    4. save_evaluation()で保存
    5. generate_full_report()でレポート生成
    """
    
    logging.basicConfig(level=logging.INFO)
    
    # サンプルWindowPerformanceデータ
    sample_windows = [
        WindowPerformance(
            window_id=0,
            val_roi=0.0523,
            test_roi=0.0419,
            val_final_balance=1052300.0,
            test_final_balance=1041900.0,
            sharpe_ratio=1.25,
            max_drawdown=-0.082,
            win_rate=0.65,
            trades=42,
        ),
        WindowPerformance(
            window_id=1,
            val_roi=0.0451,
            test_roi=0.0387,
            val_final_balance=1045100.0,
            test_final_balance=1038700.0,
            sharpe_ratio=1.15,
            max_drawdown=-0.095,
            win_rate=0.62,
            trades=38,
        ),
    ]
    
    # パイプライン実行
    pipeline = WalkForwardEvaluationPipeline(
        model_name="sac_v456_walk_forward",
        output_dir="results/phase4/evaluation",
    )
    
    # 統合評価を実行
    evaluation = pipeline.integrate_walk_forward_results(
        windows=sample_windows,
        model_metadata={"version": "v456", "algorithm": "SAC"},
    )
    
    # 結果を表示
    print(pipeline.generate_full_report())
    
    # 結果を保存
    output_path = pipeline.save_evaluation(format="json")
    print(f"\nEvaluation saved: {output_path}")

if __name__ == "__main__":
    main()
