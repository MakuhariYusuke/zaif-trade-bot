#!/usr/bin/env python3
"""
Simple SAC v444 Backtest
シンプルなバックテスト実行
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

    logger.info("🚀 Simple SAC v444 Backtest")
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
        
        # Generate synthetic data if file doesn't exist or is corrupted
        if not Path(data_file).exists():
            logger.info("Generating synthetic BTC price data...")
            synthetic_df = generate_synthetic_data(n_periods=5000, start_price=50000.0, volatility=500)
            synthetic_df.to_csv(data_file)
            logger.info(f"✅ Synthetic data generated and saved to {data_file}")
        
        df = pd.read_csv(data_file)
        logger.info(f"✅ Data loaded: {len(df)} rows from {data_file}")

        # 環境作成 (reward_settingsを適用するため)
        reward_settings = config.get("reward_settings", {})
        logger.info(f"✅ Reward settings loaded: {reward_settings}")

        # 特徴量エンジニアリングを使用せずに基本的な特徴量のみを使用
        # 学習時と同じ観測空間にするため
        basic_features = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in basic_features):
            featured_df = df.copy()
            available_features = basic_features
            logger.info(f"✅ Using basic features: {available_features}")
        else:
            logger.error("❌ Basic OHLCV features not found in data")
            return None

        # モデルロード
        model_path = f"models/{model_name}.zip"
        logger.info(f"Loading model from {model_path}")
        model = SAC.load(model_path)
        logger.info(f"✅ Model loaded: {model_name}")
        logger.debug(f"   Observation space: {model.observation_space}")
        logger.debug(f"   Action space: {model.action_space}")

        # 環境設定を取得
        env_config = config.get("environment", {})
        logger.info("Environment config extracted from main config")
        logger.debug(f"env_config keys: {list(env_config.keys())}")
        logger.debug(f"env_config action_bonuses: {env_config.get('action_bonuses', 'NOT_FOUND')}")
        logger.debug(f"env_config base_action_penalty: {env_config.get('base_action_penalty', 'NOT_FOUND')}")
        logger.debug(f"env_config curriculum_stage: {env_config.get('curriculum_stage', 'NOT_FOUND')}")

        # EnvironmentConfigオブジェクト作成
        logger.info("Creating EnvironmentConfig object...")
        try:
            env_config_obj = EnvironmentConfig.from_dict(env_config)
            logger.info("✅ EnvironmentConfig created successfully")
            logger.debug(f"env_config_obj.base_action_penalty: {env_config_obj.base_action_penalty}")
            logger.debug(f"env_config_obj.action_bonuses: {env_config_obj.action_bonuses}")
        except Exception as e:
            logger.error(f"❌ Failed to create EnvironmentConfig: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # 環境作成 (reward_settingsとfeature_columnsを渡す)
        logger.info("Creating HeavyTradingEnv...")
        try:
            env = HeavyTradingEnv(
                data=featured_df,  # 特徴量生成済みのデータを使用
                config=env_config_obj,
                reward_settings=reward_settings,
                feature_columns=available_features,
            )
            logger.info("✅ HeavyTradingEnv created successfully")
            logger.debug(f"Environment observation space: {env.observation_space}")
            logger.debug(f"Environment action space: {env.action_space}")
        except Exception as e:
            logger.error(f"❌ Failed to create HeavyTradingEnv: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # 簡単なバックテスト実行
        initial_balance = 10000
        balance = initial_balance
        position = 0  # 0: no position, 1: long
        trades = []
        total_reward = 0

        # 行動の分布をデバッグ (1000ステップ分) - 環境を使って行動生成
        action_values_raw = []
        action_values_mapped = []
        logger.info(f"Starting action analysis for {model_name} with {len(df)} rows...")
        max_steps = min(1100, len(df))

        # 環境をリセットして行動分析
        obs, _ = env.reset()
        for i in range(100, max_steps):  # 1000ステップ分確認
            # 環境を使って行動を生成 (reward_settingsが適用される)
            action, _ = model.predict(obs, deterministic=False)
            action_raw = float(action[0])  # SAC outputs in [-1, 1] range
            action_mapped = action_raw  # No transformation needed for SAC
            action_values_raw.append(action_raw)
            action_values_mapped.append(action_mapped)

            # 次のステップへ
            obs, reward, terminated, truncated, info = env.step(action)

        action_values_raw = np.array(action_values_raw)
        action_values_mapped = np.array(action_values_mapped)

        action_values_raw = np.array(action_values_raw)
        action_values_mapped = np.array(action_values_mapped)

        logger.debug(f"   Collected {len(action_values_raw)} action samples")
        logger.debug(
            f"   Raw action stats - Min: {action_values_raw.min():.4f}, Max: {action_values_raw.max():.4f}, Mean: {action_values_raw.mean():.4f}, Std: {action_values_raw.std():.4f}"
        )
        logger.debug(
            f"   Mapped action stats - Min: {action_values_mapped.min():.4f}, Max: {action_values_mapped.max():.4f}, Mean: {action_values_mapped.mean():.4f}, Std: {action_values_mapped.std():.4f}"
        )

        # 定数を使用した閾値チェック
        from ztb.trading.constants import (
            SAC_CONTINUOUS_THRESHOLD,
            SAC_CONTINUOUS_THRESHOLD_NEG,
        )

        # 連続アクションを離散アクションに変換
        discrete_actions = [
            continuous_to_discrete_action(action) for action in action_values_mapped
        ]

        # 行動の分布を分析
        buy_signals = discrete_actions.count(ACTION_BUY)
        sell_signals = discrete_actions.count(ACTION_SELL)
        hold_signals = discrete_actions.count(ACTION_HOLD)

        # 統計をファイルに書き出し
        with open(f"action_analysis_{model_name}.txt", "w") as f:
            f.write(f"=== Action Analysis for {model_name} ===\n")
            f.write(f"Sample size: {len(action_values_mapped)}\n")
            f.write(
                f"Raw actions - Mean: {np.mean(action_values_raw):.4f}, Std: {np.std(action_values_raw):.4f}, Min: {np.min(action_values_raw):.4f}, Max: {np.max(action_values_raw):.4f}\n"
            )
            f.write(
                f"Mapped actions - Mean: {np.mean(action_values_mapped):.4f}, Std: {np.std(action_values_mapped):.4f}, Min: {np.min(action_values_mapped):.4f}, Max: {np.max(action_values_mapped):.4f}\n"
            )
            f.write(
                f"Action distribution - BUY: {buy_signals}, HOLD: {hold_signals}, SELL: {sell_signals}\n"
            )
            f.write(
                f"Thresholds - Buy: {SAC_CONTINUOUS_THRESHOLD}, Sell: {SAC_CONTINUOUS_THRESHOLD_NEG}\n"
            )
            f.write(f"Unique raw values: {len(np.unique(action_values_raw))}\n")
            f.write(f"Unique mapped values: {len(np.unique(action_values_mapped))}\n")

        print(
            f"   Action distribution - BUY: {buy_signals}, HOLD: {hold_signals}, SELL: {sell_signals} (out of {len(action_values_mapped)} steps)"
        )

        # 実際のバックテスト実行 (環境ベース)
        logger.info(f"Starting full backtest for {model_name}...")
        obs, _ = env.reset()
        # ウォームアップ期間をスキップするためcurrent_stepを99に設定
        env.current_step = 99
        balance = initial_balance
        position = 0  # 0: no position, 1: long
        trades = []
        total_reward = 0
        entry_price = 0

        for i in range(100, len(df)):  # ウォームアップ期間をスキップ
            # 環境を使って行動を生成 (reward_settingsが適用される)
            action, _ = model.predict(obs, deterministic=False)
            action_raw = action[0].item()  # 生の行動値 [-1,1]

            # 環境のステップ実行
            obs, reward, terminated, truncated, info = env.step(action)

            current_price = df["close"].iloc[i]

            # 環境のトレード情報を取得
            env_balance = info.get("balance", balance)
            env_position = info.get("position", position)
            env_trades_count = info.get("trades_count", 0)

            # トレードが発生した場合のログ
            if env_trades_count > len(trades):
                if env_position > 0 and position <= 0:  # BUY
                    position = 1
                    entry_price = current_price
                    trades.append({"type": "BUY", "price": current_price, "index": i})
                    logger.info(
                        f"BUY at {current_price:.2f} (raw: {action_raw:.4f}, env_position: {env_position:.4f})"
                    )
                elif env_position < 0 and position >= 0:  # SELL
                    position = -1
                    entry_price = current_price
                    trades.append({"type": "SELL", "price": current_price, "index": i})
                    logger.info(
                        f"SELL at {current_price:.2f} (raw: {action_raw:.4f}, env_position: {env_position:.4f})"
                    )
                elif env_position == 0 and position != 0:  # CLOSE
                    exit_price = current_price
                    pnl = (
                        (exit_price - entry_price) / entry_price * 100
                        if position > 0
                        else (entry_price - exit_price) / entry_price * 100
                    )
                    total_reward += pnl
                    trades.append(
                        {
                            "type": "CLOSE",
                            "price": current_price,
                            "pnl": pnl,
                            "index": i,
                        }
                    )
                    logger.info(
                        f"CLOSE at {current_price:.2f}, PnL: {pnl:.2f}% (raw: {action_raw:.4f}, env_position: {env_position:.4f})"
                    )
                    position = 0

            balance = env_balance

        # 最終ポジションの決済
        if position != 0:
            final_price = df["close"].iloc[-1]
            pnl = (
                (final_price - entry_price) / entry_price * 100
                if position > 0
                else (entry_price - final_price) / entry_price * 100
            )
            total_reward += pnl
            trades.append(
                {
                    "type": "FINAL_CLOSE",
                    "price": final_price,
                    "pnl": pnl,
                    "index": len(df) - 1,
                }
            )
            logger.info(f"FINAL CLOSE at {final_price:.2f}, PnL: {pnl:.2f}%")

        # 結果の計算
        final_balance = balance
        total_return_pct = (final_balance - initial_balance) / initial_balance * 100

        results = {
            "model_name": model_name,
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return_pct,
            "total_trades": len(
                [t for t in trades if t["type"] in ["SELL", "FINAL_SELL"]]
            ),
            "winning_trades": len(
                [
                    t
                    for t in trades
                    if t.get("pnl", 0) > 0 and t["type"] in ["SELL", "FINAL_SELL"]
                ]
            ),
            "losing_trades": len(
                [
                    t
                    for t in trades
                    if t.get("pnl", 0) < 0 and t["type"] in ["SELL", "FINAL_SELL"]
                ]
            ),
            "avg_win_pct": np.mean(
                [
                    t["pnl"]
                    for t in trades
                    if t.get("pnl", 0) > 0 and t["type"] in ["SELL", "FINAL_SELL"]
                ]
            )
            if any(
                t.get("pnl", 0) > 0
                for t in trades
                if t["type"] in ["SELL", "FINAL_SELL"]
            )
            else 0,
            "avg_loss_pct": np.mean(
                [
                    t["pnl"]
                    for t in trades
                    if t.get("pnl", 0) < 0 and t["type"] in ["SELL", "FINAL_SELL"]
                ]
            )
            if any(
                t.get("pnl", 0) < 0
                for t in trades
                if t["type"] in ["SELL", "FINAL_SELL"]
            )
            else 0,
            "max_drawdown_pct": 0,  # 簡易版なので計算しない
            "sharpe_ratio": 0,  # 簡易版なので計算しない
            "action_distribution": {
                "BUY": buy_signals / len(discrete_actions) if discrete_actions else 0,
                "HOLD": hold_signals / len(discrete_actions) if discrete_actions else 0,
                "SELL": sell_signals / len(discrete_actions) if discrete_actions else 0,
            },
            "trades": trades,
        }

        return results

    except Exception as e:
        logger.error(f"❌ Error in backtest for {model_name}: {e}")
        return None


def main():
    """メイン関数"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Simple SAC v444.2 Backtest")

    # モデル名と設定ファイル - v444.2を使用
    model_name = "sac_v444_2_final_model"
    config_path = "config/sac_v444_2_integrated_regime_adaptation_config.json"

    # バックテスト実行
    result = run_simple_backtest(model_name, config_path)

    if result:
        print_formatted_metrics(result, "SAC v444.2 Backtest Results")

        # 結果をJSONファイルに保存
        with open("backtest_results_sac_v444_2.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info("✅ Results saved to backtest_results_sac_v444_2.json")

        # 目標チェック: 25% リターン改善
        if result["total_return_pct"] > 25:
            logger.info("🎉 SUCCESS: Achieved 25% return improvement target!")
        else:
            logger.warning(
                f"⚠️  Current return: {result['total_return_pct']:.2f}%, Target: 25% improvement"
            )

    else:
        logger.error("❌ Backtest failed")


if __name__ == "__main__":
    main()
