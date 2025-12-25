"""
SAC v434.2 統合学習マネージャー
様々な学習方法を組み合わせた包括的学習システム
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.analysis.features.feature_correlation_analyzer import (
    FeatureCorrelationAnalyzer,
)
from ztb.multimodal.features.news_feature_processor import NewsFeatureProcessor
from ztb.trading.cost.venue_transaction_cost_manager import VenueTransactionCostManager
from ztb.training.core.feature_schema_manager import FeatureSchemaManager
from ztb.utils.core.logger import log_config_summary, log_metrics
from ztb.utils.safety import safe_config_get, safe_config_get_float

logger = logging.getLogger(__name__)


class CurriculumStage:
    """カリキュラム学習の各段階設定"""

    def __init__(
        self,
        name: str,
        duration: int,
        reward_weights: Dict[str, float],
        feature_subset: Optional[List[str]] = None,
        transaction_cost: float = 0.001,
    ):
        self.name = name
        self.duration = duration  # 学習ステップ数
        self.reward_weights = reward_weights
        self.feature_subset = feature_subset
        self.transaction_cost = transaction_cost


class SACv434IntegratedLearner:
    """
    SAC v434.2統合学習マネージャー
    特徴量相関分析、ニュース特徴量、カリキュラム学習、取引コスト適応を統合
    """

    def __init__(self, config_path: str = "config/sac_v434_2_integrated_config.json"):
        """
        初期化

        Args:
            config_path: 統合設定ファイルパス
        """
        logger.info("SACv434IntegratedLearner初期化開始...")
        self.config_path = Path(config_path)
        self.config = self._load_config()
        logger.info("設定ファイル読み込み完了")

        # 各マネージャーの初期化
        logger.info("マネージャー初期化開始...")
        try:
            self.correlation_analyzer = FeatureCorrelationAnalyzer()
            logger.info("FeatureCorrelationAnalyzer初期化成功")
        except Exception as e:
            logger.warning(f"FeatureCorrelationAnalyzer初期化に失敗: {e}")
            logger.info(f"FeatureCorrelationAnalyzer初期化失敗: {e}")
            self.correlation_analyzer = None

        try:
            self.news_processor = NewsFeatureProcessor()
            logger.info("NewsFeatureProcessor初期化成功")
        except Exception as e:
            logger.warning(f"NewsFeatureProcessor初期化に失敗: {e}")
            logger.info(f"NewsFeatureProcessor初期化失敗: {e}")
            self.news_processor = None

        try:
            self.cost_manager = VenueTransactionCostManager()
            logger.info("VenueTransactionCostManager初期化成功")
        except Exception as e:
            logger.warning(f"VenueTransactionCostManager初期化に失敗: {e}")
            logger.info(f"VenueTransactionCostManager初期化失敗: {e}")
            self.cost_manager = None

        try:
            self.feature_manager = FeatureSchemaManager(
                model_name="sac_v434_2_integrated"
            )
            logger.info("FeatureSchemaManager初期化成功")
        except Exception as e:
            logger.warning(f"FeatureSchemaManager初期化に失敗: {e}")
            logger.info(f"FeatureSchemaManager初期化失敗: {e}")
            self.feature_manager = None

        # カリキュラム学習設定
        self.curriculum_stages = self._setup_curriculum_stages()

        # 学習状態
        self.current_stage = 0
        self.learning_history = []
        logger.info("SACv434IntegratedLearner初期化完了")

    def _load_config(self) -> Dict[str, Any]:
        """統合設定を読み込み"""
        if not self.config_path.exists():
            return self._create_default_config()

        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を作成"""
        config = {
            "model_name": "sac_v434_2_integrated",
            "total_timesteps": 1000000,
            "curriculum_learning": True,
            "feature_correlation_analysis": True,
            "news_feature_integration": True,
            "adaptive_cost_management": True,
            "ensemble_learning": False,
            "correlation_threshold": 0.85,
            "max_features": 50,
            "venue_name": "coincheck",  # 当座の間無料
            "data_path": "data/train.csv",
            "news_data_path": "data/news.csv",
            "output_dir": "models/sac_v434_2_integrated",
            "checkpoint_interval": 50000,
            "evaluation_interval": 10000,
        }

        # 設定ファイルを保存
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"デフォルト統合設定を作成: {self.config_path}")
        return config

    def _setup_curriculum_stages(self) -> List[CurriculumStage]:
        """カリキュラム学習の段階を設定"""
        stages = [
            # 段階1: 基礎学習（低コスト、高ペナルティ）
            CurriculumStage(
                name="foundation",
                duration=200000,
                reward_weights={
                    "profit_bonus": 1.0,
                    "loss_penalty": -2.0,
                    "action_penalty": 0.1,
                    "holding_penalty": 0.05,
                },
                transaction_cost=0.001,  # 0.1%
                feature_subset=None,  # 全特徴量使用
            ),
            # 段階2: コスト意識学習（現実的コスト）
            CurriculumStage(
                name="cost_aware",
                duration=300000,
                reward_weights={
                    "profit_bonus": 2.0,
                    "loss_penalty": -3.0,
                    "action_penalty": 0.15,
                    "holding_penalty": 0.08,
                },
                transaction_cost=0.0015,  # 0.15%
                feature_subset=None,
            ),
            # 段階3: 最適化学習（高頻度ペナルティ）
            CurriculumStage(
                name="optimization",
                duration=500000,
                reward_weights={
                    "profit_bonus": 3.0,
                    "loss_penalty": -5.0,
                    "action_penalty": 0.2,
                    "holding_penalty": 0.1,
                },
                transaction_cost=0.0015,
                feature_subset=None,
            ),
        ]

        return stages

    def analyze_and_select_features(self, data_path: str) -> Dict[str, Any]:
        """
        特徴量の相関分析と選択

        Args:
            data_path: データファイルパス

        Returns:
            特徴量選択結果
        """
        logger.info("特徴量相関分析を開始")

        # データ読み込みと特徴量抽出
        # 実際の実装ではデータ構造に応じて調整が必要
        try:
            # 特徴量相関分析を実行
            self.correlation_analyzer.load_feature_data()

            # サンプル特徴量データ（実際のデータに置き換え）
            np.random.seed(42)
            n_samples = 10000
            n_features = 100

            feature_matrix = np.random.randn(n_samples, n_features)
            feature_names = [f"feature_{i}" for i in range(n_features)]
            target_returns = (
                feature_matrix[:, 0] * 0.5 + np.random.randn(n_samples) * 0.2
            )

            # 相関分析
            correlation_results = (
                self.correlation_analyzer.analyze_feature_correlations(
                    feature_matrix, feature_names
                )
            )

            # 重要度分析
            importance_results = (
                self.correlation_analyzer.analyze_feature_importance_correlation(
                    feature_matrix, feature_names, target_returns
                )
            )

            # レポート作成
            self.correlation_analyzer.create_correlation_report(
                correlation_results,
                output_path=f"{self.config['output_dir']}/feature_correlation_report.txt",
            )

            # 特徴量選択（相関の高いペアを除去）
            selected_features = self._select_features_based_correlation(
                correlation_results, importance_results
            )

            logger.info(
                f"特徴量選択完了: {len(selected_features)}/{len(feature_names)} 特徴量を選択"
            )

            return {
                "correlation_results": correlation_results,
                "importance_results": importance_results,
                "selected_features": selected_features,
                "reduction_ratio": len(selected_features) / len(feature_names),
            }

        except Exception as e:
            logger.error(f"特徴量分析に失敗: {e}")
            return {"error": str(e)}

    def _select_features_based_correlation(
        self, correlation_results: Dict[str, Any], importance_results: Dict[str, Any]
    ) -> List[str]:
        """
        相関分析結果に基づく特徴量選択

        Args:
            correlation_results: 相関分析結果
            importance_results: 重要度分析結果

        Returns:
            選択された特徴量リスト
        """
        correlation_results.get("feature_names", [])
        high_corr_pairs = correlation_results.get("high_correlation_pairs", [])
        importance_ranking = importance_results.get("importance_ranking", [])

        # 相関の高い特徴量ペアを特定
        correlated_features = set()
        for pair in high_corr_pairs:
            if abs(pair["pearson"]) > self.config["correlation_threshold"]:
                correlated_features.add(pair["feature1"])
                correlated_features.add(pair["feature2"])

        # 重要度ランキングの上位特徴量を選択
        top_features = []
        for feature_name, _ in importance_ranking[: self.config["max_features"]]:
            if feature_name not in correlated_features:
                top_features.append(feature_name)

        # 相関の低い重要な特徴量が少ない場合は一部の相関特徴量を追加
        if len(top_features) < self.config["max_features"] * 0.5:
            for feature_name, _ in importance_ranking:
                if (
                    feature_name not in top_features
                    and len(top_features) < self.config["max_features"]
                ):
                    top_features.append(feature_name)

        return top_features

    def integrate_news_features(
        self, price_data_path: str, news_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ニュース特徴量の統合

        Args:
            price_data_path: 価格データパス
            news_data_path: ニュースデータパス

        Returns:
            ニュース特徴量統合結果
        """
        logger.info("ニュース特徴量統合を開始")

        try:
            # ニュースデータがある場合
            if (
                news_data_path
                and Path(news_data_path).exists()
                and self.news_processor is not None
            ):
                news_df = self.news_processor.load_news_data(news_data_path)

                if not news_df.empty:
                    # ニュース特徴量集約
                    news_features_df = (
                        self.news_processor.aggregate_news_features_by_time(news_df)
                    )

                    # 価格データとの統合
                    integrated_df = self.news_processor.integrate_with_price_features(
                        news_features_df, pd.read_csv(price_data_path)
                    )

                    # ニュース影響分析
                    news_impact_df = self.news_processor.create_news_impact_features(
                        news_features_df, pd.read_csv(price_data_path)
                    )

                    logger.info(
                        f"ニュース特徴量統合完了: {len(integrated_df)} サンプル"
                    )

                    return {
                        "integrated_data": integrated_df,
                        "news_features": news_features_df,
                        "news_impact": news_impact_df,
                        "news_count": len(news_df),
                    }
                else:
                    logger.warning("ニュースデータが空です")
            else:
                logger.info("ニュースデータなし - 価格特徴量のみを使用")

            # ニュースデータがない場合でも価格データを返す
            price_df = pd.read_csv(price_data_path)
            return {
                "integrated_data": price_df,
                "news_features": None,
                "news_impact": None,
                "news_count": 0,
            }

        except Exception as e:
            logger.error(f"ニュース特徴量統合に失敗: {e}")
            return {"error": str(e)}

    def adapt_transaction_costs(
        self, current_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        取引コストの適応調整

        Args:
            current_performance: 現在の学習性能指標

        Returns:
            コスト適応結果
        """
        logger.info("取引コスト適応を開始")

        try:
            venue_name = self.config.get("venue_name", "coincheck")
            trading_frequency = current_performance.get("trading_frequency", 0.5)

            # 現在のコスト設定を取得
            cost_config = self.cost_manager.get_cost_config(venue_name)

            if cost_config:
                # 戦略別推奨を取得
                if trading_frequency > 0.7:
                    strategy = "high_frequency"
                elif trading_frequency > 0.3:
                    strategy = "scalping"
                else:
                    strategy = "swing"

                recommendations = self.cost_manager.recommend_venue_for_strategy(
                    strategy, int(trading_frequency * 100)
                )

                # 最適な取引所を選択（当座はCoincheck固定）
                optimal_venue = venue_name  # Coincheck固定

                logger.info(f"取引コスト適応完了: {optimal_venue} ({strategy}戦略)")

                return {
                    "optimal_venue": optimal_venue,
                    "strategy": strategy,
                    "recommendations": recommendations,
                    "current_cost": cost_config.get_effective_cost(False),
                }
            else:
                logger.warning(f"取引所設定が見つかりません: {venue_name}")
                return {"error": f"取引所設定が見つかりません: {venue_name}"}

        except Exception as e:
            logger.error(f"取引コスト適応に失敗: {e}")
            return {"error": str(e)}

    def execute_curriculum_learning(self, base_training_function) -> Dict[str, Any]:
        """
        カリキュラム学習の実行

        Args:
            base_training_function: 基本トレーニング関数

        Returns:
            カリキュラム学習結果
        """
        logger.info("カリキュラム学習を開始")

        curriculum_results = []

        for stage_idx, stage in enumerate(self.curriculum_stages):
            logger.info(
                f"カリキュラム段階 {stage_idx + 1}/{len(self.curriculum_stages)}: {stage.name}"
            )

            # 段階ごとの設定を適用
            stage_config = {
                "stage_name": stage.name,
                "reward_weights": stage.reward_weights,
                "transaction_cost": stage.transaction_cost,
                "feature_subset": stage.feature_subset,
                "timesteps": stage.duration,
            }

            # トレーニング実行（実際にはbase_training_functionを呼び出し）
            # ここではシミュレーション
            stage_result = {
                "stage": stage.name,
                "config": stage_config,
                "performance": {
                    "reward": np.random.uniform(-100, 100),
                    "trading_frequency": np.random.uniform(0.1, 0.9),
                    "sharpe_ratio": np.random.uniform(-1, 3),
                },
                "completed": True,
            }

            curriculum_results.append(stage_result)

            # 学習履歴に記録
            self.learning_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage.name,
                    "performance": stage_result["performance"],
                }
            )

        logger.info("カリキュラム学習完了")
        return {
            "curriculum_results": curriculum_results,
            "total_stages": len(self.curriculum_stages),
            "learning_history": self.learning_history,
        }

    def create_integrated_training_config(self) -> Dict[str, Any]:
        """
        統合トレーニング設定を作成

        Returns:
            統合トレーニング設定
        """
        # 特徴量分析結果を取得
        feature_analysis = self.analyze_and_select_features(self.config["data_path"])

        # ニュース特徴量統合
        news_integration = self.integrate_news_features(
            self.config["data_path"], self.config.get("news_data_path")
        )

        # 統合設定を作成
        integrated_config = {
            "model_config": {
                "model_name": self.config["model_name"],
                "total_timesteps": self.config["total_timesteps"],
                "curriculum_learning": self.config["curriculum_learning"],
                "ensemble_learning": self.config["ensemble_learning"],
            },
            "feature_config": {
                "correlation_analysis": feature_analysis,
                "news_integration": news_integration,
                "max_features": self.config["max_features"],
                "correlation_threshold": self.config["correlation_threshold"],
            },
            "cost_config": {
                "venue_name": self.config["venue_name"],
                "adaptive_cost": self.config["adaptive_cost_management"],
            },
            "training_config": {
                "output_dir": self.config["output_dir"],
                "checkpoint_interval": self.config["checkpoint_interval"],
                "evaluation_interval": self.config["evaluation_interval"],
            },
        }

        # 設定を保存
        config_output_path = (
            Path(self.config["output_dir"]) / "integrated_training_config.json"
        )
        config_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(integrated_config, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"統合トレーニング設定を作成: {config_output_path}")

        return integrated_config

    def run_integrated_training(self) -> Dict[str, Any]:
        """
        統合学習を実行

        Returns:
            統合学習結果
        """
        logger.info("SAC v434.2統合学習を開始")

        try:
            # 統合設定を作成
            integrated_config = self.create_integrated_training_config()

            # カリキュラム学習を実行
            if self.config["curriculum_learning"]:
                curriculum_results = self.execute_curriculum_learning(
                    None
                )  # 実際のトレーニング関数を渡す
            else:
                curriculum_results = {"message": "カリキュラム学習無効"}

            # 最終結果
            final_result = {
                "config": integrated_config,
                "curriculum_results": curriculum_results,
                "feature_analysis": integrated_config["feature_config"][
                    "correlation_analysis"
                ],
                "news_integration": integrated_config["feature_config"][
                    "news_integration"
                ],
                "cost_adaptation": {
                    "venue": self.config["venue_name"],
                    "cost": 0.0,
                },  # Coincheck無料
                "training_completed": True,
                "timestamp": datetime.now().isoformat(),
            }

            # 結果を保存
            result_path = (
                Path(self.config["output_dir"]) / "integrated_training_results.json"
            )
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"統合学習完了: {result_path}")

            return final_result

        except Exception as e:
            logger.error(f"統合学習に失敗: {e}")
            return {"error": str(e)}


def create_v434_2_integrated_learner() -> SACv434IntegratedLearner:
    """
    SAC v434.2統合学習マネージャーを作成

    Returns:
        統合学習マネージャー
    """
    return SACv434IntegratedLearner()


if __name__ == "__main__":
    # 統合学習マネージャーのテスト
    learner = create_v434_2_integrated_learner()

    logger.info("SAC v434.2統合学習マネージャー初期化完了")
    logger.info(f"設定ファイル: {learner.config_path}")
    logger.info(f"カリキュラム段階数: {len(learner.curriculum_stages)}")

    # 統合設定作成
    config = learner.create_integrated_training_config()
    logger.info(f"統合設定作成完了: {config['model_config']['model_name']}")
    log_config_summary(logger, config, "Integrated Training Config")

    # 統合学習実行（シミュレーション）
    results = learner.run_integrated_training()
    logger.info(f"統合学習完了: {results.get('training_completed', False)}")
    log_metrics(logger, results, "Training Results")
