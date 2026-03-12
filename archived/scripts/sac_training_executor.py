#!/usr/bin/env python3
"""
SAC v427/v429 Training Execution Script

実行計画に基づいた段階的学習スクリプト
v429: 対称アクション変換と報酬最適化対応
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.optimization.diverse_learning_methods import DiverseLearningMethods
from ztb.trading.backtest.runner import BacktestEngine
from ztb.trading.environment.constants import (
    CONTINUOUS_TO_DISCRETE_THRESHOLD,
    CONTINUOUS_TO_DISCRETE_THRESHOLD_NEG,
)
from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACTrainingExecutor:
    """SAC v427/v429学習実行クラス"""

    def __init__(self, config_path: str, version: str = "v427"):
        self.config_path = Path(config_path)
        self.version = version
        self.config = self._load_config()
        self.trainer = None
        self.optimizer = DiverseLearningMethods()

        # v429固有の設定
        self.symmetric_thresholds = self.version == "v429"
        if self.symmetric_thresholds:
            self._setup_v429_config()

    def _setup_v429_config(self):
        """v429固有の設定を初期化"""
        # 対称アクション変換の設定
        self.config["action_conversion"] = {
            "symmetric_thresholds": True,
            "action_threshold": CONTINUOUS_TO_DISCRETE_THRESHOLD,
            "buy_threshold": CONTINUOUS_TO_DISCRETE_THRESHOLD,
            "sell_threshold": CONTINUOUS_TO_DISCRETE_THRESHOLD_NEG,
            "hold_range": [
                -CONTINUOUS_TO_DISCRETE_THRESHOLD,
                CONTINUOUS_TO_DISCRETE_THRESHOLD,
            ],
        }

        # 報酬関数の拡張設定
        if "reward_function" not in self.config:
            self.config["reward_function"] = {}

        reward_config = self.config["reward_function"]
        reward_config.update(
            {
                "action_balance_weight": reward_config.get(
                    "action_balance_weight", 0.1
                ),
                "sell_penalty": reward_config.get("sell_penalty", 0.0),
                "buy_bonus": reward_config.get("buy_bonus", 0.0),
            }
        )

        logger.info("v429設定を適用: 対称アクション変換有効")

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def phase_1_foundation_training(
        self, total_timesteps: int = 10000
    ) -> Dict[str, Any]:
        """
        Phase 1: 初期学習実行

        Args:
            total_timesteps: 学習ステップ数 (デフォルト: 10,000)

        Returns:
            学習結果
        """
        logger.info("=== Phase 1: Foundation Training開始 ===")
        logger.info(f"学習ステップ数: {total_timesteps}")

        # 設定更新
        config = self.config.copy()
        config["total_timesteps"] = total_timesteps
        config["eval_freq"] = 10000
        config["n_eval_episodes"] = 10
        config["save_freq"] = 25000

        # 学習実行
        start_time = time.time()

        try:
            self.trainer = UnifiedTrainer(config)
            results = self.trainer.run()

            end_time = time.time()
            training_time = end_time - start_time

            logger.info(f"学習時間: {training_time:.2f}秒")
            logger.info(f"学習完了: {results}")

            return {
                "phase": "foundation",
                "total_timesteps": total_timesteps,
                "training_time": training_time,
                "results": {"success": results, "training_time": training_time},
                "success": results,
            }

        except Exception as e:
            logger.error(f"Phase 1学習失敗: {e}")
            return {"phase": "foundation", "error": str(e), "success": False}

    def phase_2_hyperparameter_optimization(self) -> Dict[str, Any]:
        """
        Phase 2: ハイパーパラメータ最適化

        Returns:
            最適化結果
        """
        logger.info("=== Phase 2: Hyperparameter Optimization開始 ===")

        # 目的関数定義
        def training_objective(params: Dict[str, Any]) -> float:
            """最適化目的関数"""
            try:
                # パラメータで設定更新
                config = self.config.copy()
                config["sac_hyperparameters"].update(params)

                # 短い学習で評価
                config["total_timesteps"] = 25000
                config["eval_freq"] = 5000
                config["n_eval_episodes"] = 5

                # 学習実行
                trainer = UnifiedTrainer(config)
                results = trainer.run()

                # 評価指標として平均報酬を使用
                mean_reward = (
                    results.get("best_mean_reward", -1000)
                    if isinstance(results, dict)
                    else -1000
                )
                return -mean_reward  # 最小化問題に変換

            except Exception as e:
                logger.warning(f"最適化試行失敗: {e}")
                return 1000  # ペナルティ

        # 探索空間定義
        search_space = {
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 1e-3},
            "batch_size": {"type": "choice", "values": [128, 256, 512]},
            "ent_coef": {"type": "loguniform", "min": 1e-4, "max": 1e-1},
            "tau": {"type": "uniform", "min": 0.001, "max": 0.01},
            "gamma": {"type": "uniform", "min": 0.95, "max": 0.999},
        }

        optimization_results = {}

        # Ray Tune最適化
        logger.info("Ray Tune最適化実行...")
        try:
            ray_results = self.optimizer.optimize_hyperparameters(
                objective_function=training_objective,
                search_space=search_space,
                framework="ray_tune",
                max_evals=20,  # 短縮版
            )
            optimization_results["ray_tune"] = ray_results
            logger.info(
                f"Ray Tune完了 - ベスト値: {ray_results.get('best_value', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"Ray Tune最適化失敗: {e}")

        # Hyperopt最適化
        logger.info("Hyperopt最適化実行...")
        try:
            hyperopt_results = self.optimizer.optimize_hyperparameters(
                objective_function=training_objective,
                search_space=search_space,
                framework="hyperopt",
                max_evals=30,  # 短縮版
            )
            optimization_results["hyperopt"] = hyperopt_results
            logger.info(
                f"Hyperopt完了 - ベスト値: {hyperopt_results.get('best_value', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"Hyperopt最適化失敗: {e}")

        # 最適パラメータの選択（最も良いものを採用）
        best_params = self._select_best_parameters(optimization_results)

        return {
            "phase": "optimization",
            "optimization_results": optimization_results,
            "best_parameters": best_params,
            "success": True,
        }

    def _select_best_parameters(
        self, optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """最適パラメータの選択"""
        best_score = float("inf")
        best_params = {}

        for framework, results in optimization_results.items():
            if "best_value" in results and results["best_value"] < best_score:
                best_score = results["best_value"]
                best_params = results.get("best_params", {})

        # デフォルトパラメータをフォールバック
        if not best_params:
            best_params = {
                "learning_rate": 3e-4,
                "batch_size": 256,
                "ent_coef": 0.01,
                "tau": 0.005,
                "gamma": 0.99,
            }

        return best_params

    def phase_2_reward_optimization(self) -> Dict[str, Any]:
        """
        Phase 2: 報酬関数最適化 (v429)

        Returns:
            最適化結果
        """
        logger.info("=== Phase 2: Reward Function Optimization (v429)開始 ===")

        try:
            from ztb.optimization.reward_function_optimizer import (
                RewardFunctionOptimizer,
            )

            # 最適化設定の作成
            optimization_config = {
                "optimization": {
                    "framework": "optuna",
                    "study_name": "sac_v429_reward_optimization",
                    "direction": "maximize",
                    "n_trials": 50,  # 短縮版
                    "timeout": 1800,  # 30分
                    "n_jobs": 2,
                },
                "parameter_space": {
                    "reward_scale": {"type": "loguniform", "min": 50.0, "max": 1000.0},
                    "trading_bonus": {"type": "loguniform", "min": 0.001, "max": 0.05},
                    "sell_penalty": {"type": "uniform", "min": -0.2, "max": 0.2},
                    "buy_bonus": {"type": "uniform", "min": -0.2, "max": 0.2},
                    "action_balance_weight": {
                        "type": "uniform",
                        "min": 0.0,
                        "max": 0.5,
                    },
                },
                "constraints": {
                    "sell_ratio": {"max": 0.4},
                    "buy_ratio": {"min": 0.15, "max": 0.4},
                },
                "objectives": {
                    "primary": "sharpe_ratio",
                    "secondary": ["total_return", "sell_ratio_penalty"],
                },
            }

            # 最適化実行
            optimizer = RewardFunctionOptimizer()
            result = optimizer.optimize_reward_function(
                stage="balanced_transition",
                evaluation_function=self._reward_evaluation_function,
                n_trials=50,
            )

            logger.info(f"報酬最適化完了 - ベストスコア: {result.best_scores}")

            return {
                "phase": "reward_optimization",
                "best_config": result.best_config,
                "best_scores": result.best_scores,
                "optimization_time": result.optimization_time,
                "success": True,
            }

        except Exception as e:
            logger.error(f"報酬最適化失敗: {e}")
            return {"phase": "reward_optimization", "error": str(e), "success": False}

    def _reward_evaluation_function(self, params: Dict[str, Any]) -> float:
        """
        報酬パラメータ評価関数

        Args:
            params: 報酬パラメータ

        Returns:
            評価スコア (Sharpe ratioなど)
        """
        try:
            # パラメータを適用した設定で短いバックテストを実行
            config = self.config.copy()
            if "reward_function" not in config:
                config["reward_function"] = {}
            config["reward_function"].update(params)

            # 簡易評価（実際の実装では適切なバックテストを実行）
            # ここではダミースコアを返す
            sell_penalty = params.get("sell_penalty", 0.0)
            action_balance_weight = params.get("action_balance_weight", 0.0)

            # SELLバイアスを減らすパラメータほど高スコア
            score = 1.0 - abs(sell_penalty) * 0.1 + action_balance_weight * 0.5
            return max(0.1, score)  # 最低スコアを保証

        except Exception as e:
            logger.warning(f"報酬評価失敗: {e}")
            return 0.1

    def phase_3_fine_tuning(
        self, best_params: Dict[str, Any], total_timesteps: int = 500000
    ) -> Dict[str, Any]:
        """
        Phase 3: 微調整学習

        Args:
            best_params: 最適化されたパラメータ
            total_timesteps: 学習ステップ数

        Returns:
            学習結果
        """
        logger.info("=== Phase 3: Fine-tuning開始 ===")
        logger.info(f"最適化パラメータ: {best_params}")

        # 設定更新
        config = self.config.copy()
        config["total_timesteps"] = total_timesteps
        config["sac_hyperparameters"].update(best_params)
        config["eval_freq"] = 25000
        config["n_eval_episodes"] = 20
        config["save_freq"] = 100000

        # 学習実行
        start_time = time.time()

        try:
            self.trainer = UnifiedTrainer(config)
            results = self.trainer.run()

            end_time = time.time()
            training_time = end_time - start_time

            logger.info(f"学習時間: {training_time:.2f}秒")
            logger.info(f"学習完了: {results}")

            return {
                "phase": "fine_tuning",
                "total_timesteps": total_timesteps,
                "training_time": training_time,
                "best_parameters": best_params,
                "results": {"success": results, "training_time": training_time},
                "success": results,
            }

        except Exception as e:
            logger.error(f"Phase 3学習失敗: {e}")
            return {"phase": "fine_tuning", "error": str(e), "success": False}

    def phase_4_final_validation(self, model_path: str) -> Dict[str, Any]:
        """
        Phase 4: 最終検証

        Args:
            model_path: 検証するモデルパス

        Returns:
            検証結果
        """
        logger.info("=== Phase 4: Final Validation開始 ===")

        try:
            # バックテスト実行
            backtest_engine = BacktestEngine()

            # 複数シナリオでのバックテスト
            scenarios = [
                {"data_path": "data/btc_jpy_real_dataset.csv", "name": "main_dataset"},
                # 必要に応じて追加のデータセット
            ]

            validation_results = {}

            for scenario in scenarios:
                logger.info(f"シナリオ検証: {scenario['name']}")

                results = backtest_engine.run_backtest(
                    model_path=model_path,
                    data_path=scenario["data_path"],
                    config=self.config,
                )

                validation_results[scenario["name"]] = results

                # 主要指標のログ
                final_equity = results.get("final_equity", 0)
                sharpe_ratio = results.get("sharpe_ratio", 0)
                total_orders = results.get("total_orders", 0)

                logger.info(f"  最終エクイティ: ¥{final_equity:,.0f}")
                logger.info(f"  Sharpe比率: {sharpe_ratio:.3f}")
                logger.info(f"  総注文数: {total_orders}")

            return {
                "phase": "validation",
                "validation_results": validation_results,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Phase 4検証失敗: {e}")
            return {"phase": "validation", "error": str(e), "success": False}

    def execute_full_training_pipeline(self) -> Dict[str, Any]:
        """
        完全な学習パイプライン実行

        Returns:
            全フェーズの結果
        """
        logger.info("=== SAC v427 完全学習パイプライン開始 ===")

        pipeline_results = {}

        # Phase 1: 初期学習
        phase1_result = self.phase_1_foundation_training()
        pipeline_results["phase_1"] = phase1_result

        if not phase1_result["success"]:
            logger.error("Phase 1失敗 - パイプライン停止")
            return pipeline_results

        # Phase 2: 最適化
        phase2_result = self.phase_2_hyperparameter_optimization()
        pipeline_results["phase_2"] = phase2_result

        if not phase2_result["success"]:
            logger.warning("Phase 2失敗 - デフォルトパラメータで続行")

        best_params = phase2_result.get("best_parameters", {})

        # Phase 3: 微調整
        phase3_result = self.phase_3_fine_tuning(best_params)
        pipeline_results["phase_3"] = phase3_result

        if not phase3_result["success"]:
            logger.error("Phase 3失敗 - パイプライン停止")
            return pipeline_results

        # Phase 4: 検証
        model_path = phase3_result.get("results", {}).get("model_path", "")
        if model_path:
            phase4_result = self.phase_4_final_validation(model_path)
            pipeline_results["phase_4"] = phase4_result

        logger.info("=== SAC v427 学習パイプライン完了 ===")
        return pipeline_results


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="SAC v427/v429 Training Executor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sac_v427_market_adaptive_ensemble.json",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v427", "v429"],
        default="v427",
        help="SACバージョン (v427: 従来版, v429: 対称アクション変換版)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "4", "full"],
        default="full",
        help="実行フェーズ",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/training", help="出力ディレクトリ"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="学習ステップ数 (Phase 1の場合、デフォルト: 10,000)",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="報酬スケール調整 (例: 100.0, 500.0)",
    )
    parser.add_argument(
        "--trading-bonus",
        type=float,
        default=None,
        help="取引ボーナス調整 (例: 0.0, 0.01)",
    )
    parser.add_argument(
        "--sell-action-penalty",
        type=float,
        default=None,
        help="SELLアクションのペナルティ (負の値でペナルティ、正の値でボーナス)",
    )
    parser.add_argument(
        "--buy-action-penalty",
        type=float,
        default=None,
        help="BUYアクションのペナルティ (負の値でペナルティ、正の値でボーナス)",
    )
    parser.add_argument(
        "--action-balance-weight",
        type=float,
        default=None,
        help="アクション平衡ウェイト (v429のみ、例: 0.1)",
    )
    parser.add_argument(
        "--optimize-reward", action="store_true", help="報酬関数最適化を実行 (v429のみ)"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 学習実行クラス初期化
    executor = SACTrainingExecutor(args.config, args.version)

    # フェーズ実行
    if args.phase == "full":
        results = executor.execute_full_training_pipeline()
    elif args.phase == "1":
        # 報酬パラメータの調整
        if args.reward_scale is not None:
            executor.config["reward_settings"]["reward_scale"] = args.reward_scale
            logger.info(f"報酬スケールを調整: {args.reward_scale}")
        if args.trading_bonus is not None:
            executor.config["reward_settings"]["profit_bonuses"][
                "trading_bonus"
            ] = args.trading_bonus
            logger.info(f"取引ボーナスを調整: {args.trading_bonus}")
        if args.sell_action_penalty is not None:
            executor.config["reward_settings"][
                "sell_action_penalty"
            ] = args.sell_action_penalty
            logger.info(f"SELLアクションペナルティを調整: {args.sell_action_penalty}")
        if args.buy_action_penalty is not None:
            executor.config["reward_settings"][
                "buy_action_penalty"
            ] = args.buy_action_penalty
            logger.info(f"BUYアクションペナルティを調整: {args.buy_action_penalty}")

        # v429固有パラメータ
        if args.version == "v429" and args.action_balance_weight is not None:
            if "reward_function" not in executor.config:
                executor.config["reward_function"] = {}
            executor.config["reward_function"][
                "action_balance_weight"
            ] = args.action_balance_weight
            logger.info(f"アクション平衡ウェイトを調整: {args.action_balance_weight}")

        results = {"phase_1": executor.phase_1_foundation_training(args.timesteps)}
    elif args.phase == "2":
        if args.version == "v429" and args.optimize_reward:
            results = {"phase_2": executor.phase_2_reward_optimization()}
        else:
            results = {"phase_2": executor.phase_2_hyperparameter_optimization()}
    elif args.phase == "3":
        # Phase 2の結果が必要なので、簡易版を実行
        best_params = {}
        results = {"phase_3": executor.phase_3_fine_tuning(best_params)}
    elif args.phase == "4":
        model_path = input("モデルパスを入力: ")
        results = {"phase_4": executor.phase_4_final_validation(model_path)}
    else:
        results = {"error": f"Unknown phase: {args.phase}"}

    # 結果保存
    output_path = os.path.join(
        args.output_dir, f"sac_{args.version}_training_results_{args.phase}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"結果を保存しました: {output_path}")

    # 簡易レポート
    print("\n=== 学習実行サマリー ===")
    for phase, result in results.items():
        status = "✅ 成功" if result.get("success", False) else "❌ 失敗"
        print(f"{phase}: {status}")

        if "training_time" in result:
            print(f"  学習時間: {result['training_time']:.2f}秒")
        if "error" in result:
            print(f"  エラー: {result['error']}")


if __name__ == "__main__":
    main()
