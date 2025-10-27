#!/usr/bin/env python3
"""
Simple SAC v435.7 Backtest
シンプルなバックテスト実行
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# 環境をインポート
sys.path.append(str(Path(__file__).parent.parent))


def run_simple_backtest(model_name, config_path):
    """シンプルなバックテストを実行"""
    print(f"\n🔍 Running simple backtest for {model_name}")

    try:
        # 設定読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # モデルロード
        model_path = f"models/{model_name}.zip"
        model = SAC.load(model_path)
        print(f"✅ Model loaded: {model_name}")
        print(f"   Observation space: {model.observation_space}")
        print(f"   Action space: {model.action_space}")

        # データ読み込み
        data_path = "data/btc_jpy_yahoo_real_20251021_featured.csv"
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df)} rows")

        # 環境設定で特徴量を取得
        env_config_path = Path("v435/v435.7/sac_v435_environment_config.json")
        with open(env_config_path, "r", encoding="utf-8") as f:
            env_config_data = json.load(f)

        # 特徴量を手動で設定 (モデルは3次元で訓練されている)
        # 基本的な3つの特徴量を使用
        feature_cols = ["close", "volume", "rsi_14"]
        print(f"✅ Using 3 features as trained: {feature_cols}")

        # 簡単なバックテスト実行
        initial_balance = 10000
        balance = initial_balance
        position = 0  # 0: no position, 1: long
        trades = []
        total_reward = 0

        # 行動の分布をデバッグ (1000ステップ分)
        action_values_raw = []
        action_values_mapped = []
        print(f"Starting action analysis for {model_name} with {len(df)} rows...")
        for i in range(100, min(1100, len(df))):  # 1000ステップ分確認
            obs = df[feature_cols].iloc[i].values.astype(np.float32)
            obs = obs.reshape(1, -1)
            action, _ = model.predict(obs, deterministic=True)
            action_raw = float(action[0])  # 変換前の値 [0,1]
            action_mapped = 2 * action_raw - 1  # 変換後の値 [-1,1]
            action_values_raw.append(action_raw)
            action_values_mapped.append(action_mapped)

        action_values_raw = np.array(action_values_raw)
        action_values_mapped = np.array(action_values_mapped)

        print(f"   Collected {len(action_values_raw)} action samples")
        print(
            f"   Raw action stats (1000 steps) - Min: {action_values_raw.min():.4f}, Max: {action_values_raw.max():.4f}, Mean: {action_values_raw.mean():.4f}, Std: {action_values_raw.std():.4f}"
        )
        print(
            f"   Mapped action stats (1000 steps) - Min: {action_values_mapped.min():.4f}, Max: {action_values_mapped.max():.4f}, Mean: {action_values_mapped.mean():.4f}, Std: {action_values_mapped.std():.4f}"
        )

        # 定数を使用した閾値チェック
        from ztb.trading.constants import (
            SAC_CONTINUOUS_THRESHOLD,
            SAC_CONTINUOUS_THRESHOLD_NEG,
        )

        buy_threshold = SAC_CONTINUOUS_THRESHOLD  # 0.3333
        sell_threshold = SAC_CONTINUOUS_THRESHOLD_NEG  # -0.3333
        print(f"   Thresholds - Buy: {buy_threshold}, Sell: {sell_threshold}")

        # 行動の分布を分析
        buy_signals = np.sum(action_values_mapped > buy_threshold)
        sell_signals = np.sum(action_values_mapped < sell_threshold)
        hold_signals = len(action_values_mapped) - buy_signals - sell_signals

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
                f"Buy threshold: {buy_threshold}, Sell threshold: {sell_threshold}\n"
            )
            f.write(f"Unique raw values: {len(np.unique(action_values_raw))}\n")
            f.write(f"Unique mapped values: {len(np.unique(action_values_mapped))}\n")

        print(
            f"   Action distribution - BUY: {buy_signals}, HOLD: {hold_signals}, SELL: {sell_signals} (out of {len(action_values_mapped)} steps)"
        )
        print(f"   Action analysis saved to action_analysis_{model_name}.txt")

        for i in range(100, len(df)):  # ウォームアップ期間をスキップ
            # 観測データの作成
            obs = df[feature_cols].iloc[i].values.astype(np.float32)
            obs = obs.reshape(1, -1)

            # 行動予測
            action, _ = model.predict(obs, deterministic=True)
            action_raw = action[0].item()  # 生の行動値 [0,1]

            # 行動空間を[-1, 1]に変換
            action_mapped = 2 * action_raw - 1

            current_price = df["close"].iloc[i]

            # 定数を使用した閾値で取引判断
            if action_mapped > buy_threshold and position == 0:  # BUY
                position = 1
                entry_price = current_price
                trades.append({"type": "BUY", "price": current_price, "index": i})
                print(
                    f"BUY at {current_price:.2f} (raw: {action_raw:.4f}, mapped: {action_mapped:.4f})"
                )
            elif action_mapped < sell_threshold and position == 1:  # SELL
                position = 0
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price * 100
                total_reward += pnl
                trades.append(
                    {"type": "SELL", "price": current_price, "pnl": pnl, "index": i}
                )
                print(
                    f"SELL at {current_price:.2f}, PnL: {pnl:.2f}% (raw: {action_raw:.4f}, mapped: {action_mapped:.4f})"
                )

        # 結果計算
        final_balance = initial_balance * (1 + total_reward / 100)
        profit = final_balance - initial_balance

        results = {
            "model": model_name,
            "trades": len(trades) // 2,  # BUY/SELLペア
            "total_return_pct": total_reward,
            "final_balance": final_balance,
            "profit": profit,
            "frequency_penalty": config["training"]["reward_function"][
                "action_frequency_penalty"
            ],
            "profit_bonus_atr": config["training"]["reward_function"][
                "base_profit_bonus_atr_coeff"
            ],
            "profit_bonus_portfolio": config["training"]["reward_function"][
                "base_profit_bonus_portfolio_coeff"
            ],
        }

        print(f"  Total Return: {total_reward:.2f}%")
        print(f"  Trades: {len(trades) // 2}")
        print(f"  Final Balance: ${final_balance:.2f}")
        return results

    except Exception as e:
        print(f"❌ Error in backtest for {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("🚀 Simple SAC v435.7 Backtest")
    print("=" * 40)

    config_dir = Path("v435/v435.7")
    models = [
        ("sac_v435.7a", config_dir / "sac_v435_7a_config.json"),
        ("sac_v435.7b", config_dir / "sac_v435_7b_config.json"),
        ("sac_v435.7c", config_dir / "sac_v435_7c_config.json"),
    ]

    results = []
    for model_name, config_path in models:
        if config_path.exists():
            result = run_simple_backtest(model_name, config_path)
            if result:
                results.append(result)
        else:
            print(f"❌ Config not found: {config_path}")

    if not results:
        print("❌ No results to compare")
        return

    # 結果をDataFrameに変換して表示
    df = pd.DataFrame(results)
    print("\n📊 Backtest Results Summary:")
    print(df.to_string(index=False, float_format="%.2f"))

    # 最も良いモデルを選択
    if not df.empty:
        best_model = df.loc[df["profit"].idxmax()]
        print("\n🏆 Best Performing Model:")
        print(f"  Model: {best_model['model']}")
        print(f"  Profit: ${best_model['profit']:.2f}")
        print(f"  Trades: {best_model['trades']}")
        print(f"  Frequency Penalty: {best_model['frequency_penalty']}")
        print(f"  Profit Bonus ATR: {best_model['profit_bonus_atr']}")
        print(f"  Profit Bonus Portfolio: {best_model['profit_bonus_portfolio']}")


if __name__ == "__main__":
    main()
