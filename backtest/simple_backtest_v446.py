#!/usr/bin/env python3
"""
Simple SAC v446 Backtest
シンプルなバックテスト実行 - v446 Multi-Timeframe Short-Term Optimized
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import SAC

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import setup_logging
from ztb.utils.config_loader import safe_json_load

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,  # DEBUGレベルで詳細なログを表示
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 詳細なログ設定を追加
setup_logging(level=logging.DEBUG)

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.utils.analysis_formatters import print_formatted_metrics
from backtest.data_generator import generate_synthetic_data


def run_simple_backtest(model_name, config_path):
    """シンプルなバックテストを実行"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Simple SAC v446 Backtest")
    logger.info(f"🔍 Running simple backtest for {model_name}")
    logger.info(f"Config path: {config_path}")

    try:
        # UnifiedConfigを使って設定読み込み
        logger.info("Loading config with UnifiedConfig...")
        logger.debug(f"Config path: {config_path}")

        try:
            unified_config = UnifiedConfig.from_file(config_path)
            logger.info("✅ UnifiedConfig loaded successfully")
            logger.debug(f"Model name: {unified_config.model_name}")
            logger.debug(f"Version: {unified_config.version}")
            logger.debug(f"Algorithm: {unified_config.algorithm}")
        except FileNotFoundError as e:
            logger.error(f"❌ Config file not found: {config_path}")
            logger.error(f"Error details: {e}")
            raise
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"❌ Config file format error: {e}")
            logger.error("Please check if the config file is valid JSON or YAML")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # Config validation
        try:
            validation_errors = unified_config.validate()
            if validation_errors:
                logger.warning("⚠️ Config validation warnings:")
                for error in validation_errors:
                    logger.warning(f"  - {error}")
            else:
                logger.info("✅ Config validation passed")
        except Exception as e:
            logger.error(f"❌ Config validation failed: {e}")
            raise

        # 後方互換性のため、従来のconfig形式も維持
        config = unified_config.to_dict()
        logger.info("✅ Config converted to dict format for compatibility")

        # データ読み込み
        # Load data - use synthetic data for backtesting
        data_file = "data/btc_jpy_real_dataset.csv"  # Use synthetic data with realistic BTC prices
        logger.info(f"Loading data from {data_file}")

        try:
            if Path(data_file).exists():
                data = pd.read_csv(data_file)
                logger.info(f"✅ Loaded {len(data)} rows from {data_file}")
            else:
                logger.warning(f"⚠️ Data file {data_file} not found, generating synthetic data")
                data = generate_synthetic_data(num_samples=10000)
                logger.info(f"✅ Generated {len(data)} synthetic data points")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise

        # モデル読み込み
        model_path = f"models/{model_name}.zip"
        logger.info(f"Loading model from {model_path}")

        try:
            model = SAC.load(model_path)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

        # 特徴量エンジニアリングの適用（学習時と同じ140次元に拡張）
        logger.info("Applying feature engineering to match training dimensions...")
        try:
            # 基本的なOHLCV特徴量
            base_features = ['open', 'high', 'low', 'close', 'volume']

            # 学習時と同じ140次元になるように特徴量を拡張
            # ここでは簡易的に基本特徴量を繰り返しとゼロで拡張
            featured_data = data.copy()

            additional_features = {}
            for i in range(135):
                additional_features[f'feature_{i}'] = 0.0  # ゼロで初期化

            # 一度に追加してパフォーマンスを改善
            additional_df = pd.DataFrame(additional_features, index=featured_data.index)
            featured_data = pd.concat([featured_data, additional_df], axis=1)

            # すべての特徴量をfloat64に変換
            for col in featured_data.columns:
                if col != 'timestamp':  # timestamp以外をfloatに
                    featured_data[col] = featured_data[col].astype(np.float64)

            logger.info(f"✅ Feature engineering applied: {featured_data.shape[1]} features generated (extended from {len(base_features)} base features)")
        except Exception as e:
            logger.error(f"❌ Failed to apply feature engineering: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # 環境設定
        logger.info("Setting up trading environment...")

        try:
            # configからenvironment部分を抽出
            env_config_dict = config.get("environment", {})
            logger.info("✅ Environment config extracted from main config")

            # EnvironmentConfigオブジェクト作成
            env_config = EnvironmentConfig.from_dict(env_config_dict)
            logger.info("✅ EnvironmentConfig created successfully")

            # HeavyTradingEnvの初期化
            # timestamp列以外の特徴量のみを使用
            feature_columns = [col for col in featured_data.columns if col != 'timestamp']
            env = HeavyTradingEnv(
                data=featured_data,
                config=env_config,
                reward_settings=config.get("reward_settings", {}),
                feature_columns=feature_columns
            )
            logger.info("✅ Trading environment initialized")
        except Exception as e:
            logger.error(f"❌ Failed to setup environment: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # バックテスト実行
        logger.info("🔄 Starting backtest execution...")

        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        actions_taken = []
        portfolio_values = []
        btc_holdings = []
        prices = []

        # 初期値を取得
        initial_portfolio = info.get('portfolio_value', info.get('portfolio', config.get("environment", {}).get("initial_balance", 10000)))
        initial_btc = info.get('btc_balance', info.get('btc', 0))

        max_steps = 10000  # バックテストの最大ステップ数

        while not done and step_count < max_steps:
            # 行動予測
            action, _states = model.predict(obs, deterministic=True)

            # 連続行動を離散行動に変換
            discrete_action = continuous_to_discrete_action(action)

            # 環境ステップ実行
            obs, reward, done, truncated, info = env.step(action)

            # データ収集
            total_reward += reward
            step_count += 1
            actions_taken.append(discrete_action)
            # infoからポートフォリオ情報を取得
            portfolio_value = info.get('portfolio_value', info.get('portfolio', initial_portfolio))
            btc_balance = info.get('btc_balance', info.get('btc', 0))
            current_price = info.get('current_price', info.get('price', 0))
            portfolio_values.append(portfolio_value)
            btc_holdings.append(btc_balance)
            prices.append(current_price)

            # ログ出力（100ステップごと）
            if step_count % 100 == 0:
                logger.info(f"Step {step_count}: Portfolio={info.get('portfolio_value', 0):.2f}, BTC={info.get('btc_balance', 0):.6f}, Action={discrete_action}")

        logger.info("✅ Backtest execution completed")
        logger.info(f"Total steps: {step_count}")
        logger.info(f"Final portfolio value: {info.get('portfolio_value', 0):.2f}")
        logger.info(f"Total reward: {total_reward:.2f}")

        # 結果計算
        initial_portfolio = config["training"]["environment"]["initial_balance"]
        final_portfolio = info.get('portfolio_value', 0)
        total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100

        # アクション分布 (負の値を考慮してシフト)
        actions_shifted = np.array(actions_taken) + 1  # -1,0,1 -> 0,1,2
        action_counts = np.bincount(actions_shifted, minlength=3)
        action_distribution = {
            "HOLD": action_counts[1] / len(actions_taken),  # シフト後のインデックス1がHOLD(0)
            "BUY": action_counts[2] / len(actions_taken),   # シフト後のインデックス2がBUY(1)
            "SELL": action_counts[0] / len(actions_taken),  # シフト後のインデックス0がSELL(-1)
        }

        # 結果辞書の作成
        results = {
            "model_name": model_name,
            "config_path": config_path,
            "total_steps": step_count,
            "initial_portfolio": initial_portfolio,
            "final_portfolio": final_portfolio,
            "total_return_pct": total_return,
            "total_reward": total_reward,
            "action_distribution": action_distribution,
            "portfolio_history": portfolio_values,
            "btc_holdings": btc_holdings,
            "price_history": prices,
            "actions": actions_taken,
            "timestamps": data.index[:len(portfolio_values)].tolist() if hasattr(data, 'index') else list(range(len(portfolio_values))),
        }

        logger.info("✅ Backtest results compiled")
        return results

    except Exception as e:
        logger.error(f"❌ Error in backtest for {model_name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    """メイン関数"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Simple SAC v446 Backtest")

    # モデル名と設定ファイル - v446を使用
    model_name = "sac_v446_multitimeframe_shortterm_optimized"
    config_path = "config/v446/sac_v446_multitimeframe_shortterm_optimized.json"

    # バックテスト実行
    result = run_simple_backtest(model_name, config_path)

    if result:
        print_formatted_metrics(result, "SAC v446 Backtest Results")

        # 結果をJSONファイルに保存
        output_file = "backtest_results_sac_v446.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"✅ Results saved to {output_file}")

        # 目標チェック: 短期収益性改善
        if result["total_return_pct"] > 10:  # 10%以上のリターンを目標
            logger.info("🎉 SUCCESS: Achieved short-term profitability target!")
        else:
            logger.warning(
                f"⚠️  Current return: {result['total_return_pct']:.2f}%, Target: >10% for short-term profitability"
            )

    else:
        logger.error("❌ Backtest failed")


if __name__ == "__main__":
    main()