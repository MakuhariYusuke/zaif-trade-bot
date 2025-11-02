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
from stable_baselines3 import SAC

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,  # DEBUGレベルで詳細なログを表示
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from ztb.training.environments.environment_config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.utils.analysis_formatters import print_formatted_metrics
from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer


def run_simple_backtest(model_name, config_path):
    """シンプルなバックテストを実行"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Simple SAC v444 Backtest")
    logger.info(f"🔍 Running simple backtest for {model_name}")
    logger.info(f"Config path: {config_path}")

    try:
        # 設定読み込み
        logger.info("Loading config...")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info("✅ Config loaded successfully")

        # データ読み込み
        # Load data - use the same data the model was trained on
        data_file = "data/btc_jpy_yahoo_real_20251021_featured.csv"  # より多くの特徴量を持つデータファイルを使用
        df = pd.read_csv(data_file)
        logger.info(f"✅ Data loaded: {len(df)} rows")

        # 環境作成 (reward_settingsを適用するため)
        reward_settings = config.get("reward_settings", {})
        logger.info(f"✅ Reward settings loaded: {reward_settings}")

        # 特徴量エンジニアリングを使用して本物の特徴量を生成
        # v444は高度なレジーム適応を使用するため、より多くの特徴量が必要
        # モデルが212次元を期待しているので、トレーニング時と同じ特徴量生成を使用
        feature_engineer = SACv427FeatureEngineer()
        logger.info("Generating features using SAC v427 Feature Engineer...")

        # 特徴量生成（トレーニング時と同じ方法）
        try:
            featured_df = feature_engineer.generate_v427_features(
                df.copy(),
                window_sizes=[3, 5, 7, 10, 14, 20, 30, 50],
                feature_set="full"  # full feature setを使用
            )
            logger.info(f"✅ Features generated: {len(featured_df.columns)} columns")

            # 数値特徴量のみを使用
            numeric_cols = featured_df.select_dtypes(include=[np.number]).columns
            available_features = list(numeric_cols)
            logger.info(f"✅ Available numeric features: {len(available_features)}")

            # モデルが212次元を期待している場合、特徴量数を調整
            if len(available_features) > 212:
                available_features = available_features[:212]
            elif len(available_features) < 212:
                # 不足分を既存の特徴量で埋める（重複を許容）
                while len(available_features) < 212:
                    available_features.extend(available_features[:212 - len(available_features)])

            available_features = available_features[:212]  # 確実に212個に制限
            logger.info(f"✅ Final feature count: {len(available_features)}")

        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            logger.warning("Falling back to basic features...")
            # フォールバック: 基本的な特徴量のみ
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            available_features = list(numeric_cols)
            featured_df = df.copy()
            logger.info(f"✅ Using fallback features: {len(available_features)}")

        if len(available_features) == 0:
            logger.error("❌ No features available for backtest")
            return None

        # モデルロード
        model_path = f"models/{model_name}.zip"
        model = SAC.load(model_path)
        logger.info(f"✅ Model loaded: {model_name}")
        logger.debug(f"   Observation space: {model.observation_space}")
        logger.debug(f"   Action space: {model.action_space}")

        # 環境設定を取得
        env_config = config.get("environment", {})

        # EnvironmentConfigオブジェクト作成
        env_config_obj = EnvironmentConfig(**env_config)

        # 環境作成 (reward_settingsとfeature_columnsを渡す)
        env = HeavyTradingEnv(
            data=featured_df,  # 特徴量生成済みのデータを使用
            config=env_config_obj,
            reward_settings=reward_settings,
            feature_columns=available_features,
        )
        logger.info("✅ Environment created with reward_settings and feature_columns")

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
    logger.info("🚀 Simple SAC v444 Backtest")

    # モデル名と設定ファイル
    model_name = "sac_v444_advanced_regime_adaptation"
    config_path = "config/sac_v444_advanced_regime_adaptation_config.json"

    # バックテスト実行
    result = run_simple_backtest(model_name, config_path)

    if result:
        print_formatted_metrics(result, "SAC v444 Backtest Results")

        # 結果をJSONファイルに保存
        with open("backtest_results_sac_v444.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info("✅ Results saved to backtest_results_sac_v444.json")

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
