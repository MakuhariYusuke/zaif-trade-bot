#!/usr/bin/env python3
"""
SAC v436 Signal Guided Backtest
シグナルガイド付きSAC v436モデルのバックテスト
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def get_v436_features():
    """v436の特徴量セットを取得（モデルの観測空間に合わせた次元）"""
    try:
        # データファイルに存在する特徴量を取得
        df = pd.read_csv("data/btc_jpy_featured_dataset.csv")
        available_features = [
            col for col in df.columns if col not in ["timestamp", "episode_id"]
        ]

        # モデルは156次元の観測空間を期待しているので、最初の156個を使用
        selected_features = available_features[:156]
        print(
            f"✅ Model expects 156D observation space, using first 156 features: {selected_features[:10]}... (showing first 10)"
        )
        print(
            f"   Available features: {len(available_features)} -> Selected: {len(selected_features)}"
        )

        return selected_features

    except Exception as e:
        print(f"⚠️ Failed to load features from data: {e}, using fallback")
        # フォールバック: 基本的な10個の特徴量
        return [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
        ]


def run_v436_backtest(
    model_path="models/sac_model.zip",
    config_path="config/sac_v436_signal_guided_config.json",
    data_path="data/btc_jpy_featured_dataset.csv",
):
    """v436モデルのバックテストを実行"""
    print("\n🔍 Running SAC v436 Signal Guided Backtest")
    print(f"  Model: {model_path}")
    print(f"  Config: {config_path}")

    try:
        # 設定読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # モデルロード
        model = SAC.load(model_path)
        print("✅ Model loaded successfully")
        print(f"   Observation space: {model.observation_space}")
        print(f"   Action space: {model.action_space}")

        # 観測空間の次元を確認
        obs_dim = model.observation_space.shape[0]
        print(f"   Model expects {obs_dim}D observation space")

        # データ読み込み
        if not Path(data_path).exists():
            print(f"❌ Data file not found: {data_path}")
            return None

        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df)} rows")

        # 観測空間の次元に基づいて特徴量を選択
        available_features = [
            col for col in df.columns if col not in ["timestamp", "episode_id"]
        ]
        feature_cols = available_features[:obs_dim]
        print(
            f"✅ Using {len(feature_cols)} features matching model's {obs_dim}D observation space"
        )
        print(f"   Features: {feature_cols[:10]}... (showing first 10)")

        # 特徴量の存在チェック
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"❌ Missing features in data: {missing_features}")
            return None

        # バックテスト実行
        initial_balance = config["training"]["environment"]["initial_balance"]
        balance = initial_balance
        position = 0  # 0: no position, 1: long
        trades = []
        total_reward = 0
        transaction_cost = config["training"]["environment"]["transaction_cost"]

        # 行動の分布を分析 (5000ステップ分)
        action_values_raw = []
        action_values_final = []
        print("Starting detailed action analysis with 5000 steps...")

        # 同じ観測値で複数回予測してstochastic性を確認
        test_obs = None
        test_predictions = []
        print("Testing stochastic predictions on same observation...")

        for i in range(100, min(5100, len(df))):  # 5000ステップ分確認
            obs = df[feature_cols].iloc[i].values.astype(np.float32)
            obs = obs.reshape(1, -1)  # モデルは(10,)の形状を期待

            # 最初の観測値を保存して複数回予測
            if test_obs is None and i == 100:
                test_obs = obs.copy()
                print(f"Testing same observation {test_obs.shape} times...")
                with open("debug_output.txt", "w") as f:
                    f.write(f"Testing same observation {test_obs.shape} times...\n")
                for j in range(10):
                    action_test, _ = model.predict(test_obs, deterministic=False)
                    test_predictions.append(float(action_test[0]))
                print(f"  Same obs predictions: {test_predictions}")
                print(
                    f"  Min: {min(test_predictions):.6f}, Max: {max(test_predictions):.6f}, Std: {np.std(test_predictions):.6f}"
                )
                with open("debug_output.txt", "a") as f:
                    f.write(f"  Same obs predictions: {test_predictions}\n")
                    f.write(
                        f"  Min: {min(test_predictions):.6f}, Max: {max(test_predictions):.6f}, Std: {np.std(test_predictions):.6f}\n"
                    )
                # 強制的に出力
                sys.stdout.flush()

            action, _ = model.predict(obs, deterministic=False)
            action_raw = float(action[0])  # モデルの生の出力値

            # アクションスペースの確認（最初の10ステップのみ）
            if i < 110:
                print(f"Model action space: {model.action_space}")
                print(f"Raw action value: {action_raw}")

            # モデルの生の[-1,1]出力を極力そのまま使用
            # 極端な値対策として軽いクリッピングのみ適用
            action_final_val = np.clip(action_raw, -0.99, 0.99)

            action_values_raw.append(action_raw)
            action_values_final.append(action_final_val)

        action_values_raw = np.array(action_values_raw)
        action_values_final = np.array(action_values_final)

        print(f"   Collected {len(action_values_raw)} action samples")
        print(
            f"   Raw action stats - Min: {action_values_raw.min():.6f}, Max: {action_values_raw.max():.6f}, Mean: {action_values_raw.mean():.6f}, Std: {action_values_raw.std():.6f}"
        )
        print(
            f"   Final action stats - Min: {action_values_final.min():.6f}, Max: {action_values_final.max():.6f}, Mean: {action_values_final.mean():.6f}, Std: {action_values_final.std():.6f}"
        )

        # 値の焦げ付き分析
        print("\n   🔥 Value Clamping Analysis:")
        extreme_high = np.sum(action_values_raw >= 0.99)
        extreme_low = np.sum(action_values_raw <= -0.99)
        near_zero = np.sum(np.abs(action_values_raw) <= 0.1)

        print(
            f"   - Extreme high (≥0.99): {extreme_high} samples ({extreme_high/len(action_values_raw)*100:.1f}%)"
        )
        print(
            f"   - Extreme low (≤-0.99): {extreme_low} samples ({extreme_low/len(action_values_raw)*100:.1f}%)"
        )
        print(
            f"   - Near zero (±0.1): {near_zero} samples ({near_zero/len(action_values_raw)*100:.1f}%)"
        )

        print("\n   📊 Action Distribution Histogram:")
        bins = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(action_values_final, bins=bins)
        for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
            print(
                f"   - [{start:.1f}, {end:.1f}): {hist[i]} samples ({hist[i]/len(action_values_final)*100:.1f}%)"
            )

        # 閾値設定
        buy_threshold = config["training"]["environment"][
            "continuous_to_discrete_threshold"
        ]
        sell_threshold = -buy_threshold
        print(f"   Thresholds - Buy: {buy_threshold}, Sell: {sell_threshold}")

        # 行動の分布を分析
        buy_signals = np.sum(action_values_final > buy_threshold)
        sell_signals = np.sum(action_values_final < sell_threshold)
        hold_signals = len(action_values_final) - buy_signals - sell_signals

        # 詳細な行動分析をファイルに書き出し
        analysis_file = "action_analysis_sac_v436_detailed.txt"
        with open(analysis_file, "w") as f:
            f.write("=== Detailed Action Analysis for SAC v436 Signal Guided ===\n")
            f.write(f"Sample size: {len(action_values_raw)}\n\n")

            f.write("RAW ACTION VALUES (-1 to 1 range):\n")
            f.write(f"  Min: {action_values_raw.min():.6f}\n")
            f.write(f"  Max: {action_values_raw.max():.6f}\n")
            f.write(f"  Mean: {action_values_raw.mean():.6f}\n")
            f.write(f"  Std: {action_values_raw.std():.6f}\n")
            f.write(f"  Median: {np.median(action_values_raw):.6f}\n\n")

            f.write("FINAL ACTION VALUES (clipped -0.99 to 0.99):\n")
            f.write(f"  Min: {action_values_final.min():.6f}\n")
            f.write(f"  Max: {action_values_final.max():.6f}\n")
            f.write(f"  Mean: {action_values_final.mean():.6f}\n")
            f.write(f"  Std: {action_values_final.std():.6f}\n")
            f.write(f"  Median: {np.median(action_values_final):.6f}\n\n")

            f.write("VALUE CLAMPING ANALYSIS:\n")
            f.write(
                f"  Extreme high (≥0.99): {np.sum(action_values_raw >= 0.99)} ({np.sum(action_values_raw >= 0.99)/len(action_values_raw)*100:.2f}%)\n"
            )
            f.write(
                f"  Extreme low (≤-0.99): {np.sum(action_values_raw <= -0.99)} ({np.sum(action_values_raw <= -0.99)/len(action_values_raw)*100:.2f}%)\n"
            )
            f.write(
                f"  Near zero (±0.1): {np.sum(np.abs(action_values_raw) <= 0.1)} ({np.sum(np.abs(action_values_raw) <= 0.1)/len(action_values_raw)*100:.2f}%)\n\n"
            )

            f.write("HISTOGRAM DISTRIBUTION:\n")
            for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
                f.write(
                    f"  [{start:.1f}, {end:.1f}): {hist[i]} ({hist[i]/len(action_values_final)*100:.1f}%)\n"
                )

            f.write("\nACTION DISTRIBUTION:\n")
            f.write(
                f"  BUY: {buy_signals} ({buy_signals/len(action_values_final)*100:.1f}%)\n"
            )
            f.write(
                f"  HOLD: {hold_signals} ({hold_signals/len(action_values_final)*100:.1f}%)\n"
            )
            f.write(
                f"  SELL: {sell_signals} ({sell_signals/len(action_values_final)*100:.1f}%)\n"
            )
            f.write(f"  Buy threshold: {buy_threshold}\n")
            f.write(f"  Sell threshold: {sell_threshold}\n\n")

            f.write("FEATURE INFORMATION:\n")
            f.write(f"  Total features used: {len(feature_cols)}\n")
            f.write("  Features list:\n")
            for i, feat in enumerate(feature_cols, 1):
                f.write(f"    {i:2d}. {feat}\n")

        print(
            f"   Action distribution - BUY: {buy_signals}, HOLD: {hold_signals}, SELL: {sell_signals} (out of {len(action_values_final)} steps)"
        )
        print(f"   Detailed analysis saved to {analysis_file}")

        # 完全なバックテスト実行
        print("Starting full backtest...")
        for i in range(100, len(df)):  # ウォームアップ期間をスキップ
            # 観測データの作成
            obs = df[feature_cols].iloc[i].values.astype(np.float32)
            obs = obs.reshape(1, -1)  # モデルは(10,)の形状を期待

            # 行動予測
            action, _ = model.predict(obs, deterministic=False)
            action_raw = action[0].item()  # 生の行動値 [-1,1]

            # 最小限のクリッピングのみ適用（値の焦げ付き対策）
            action_final = np.clip(action_raw, -0.99, 0.99)

            current_price = df["close"].iloc[i]

            # 閾値で取引判断（強化版クリッピング適用）
            if action_final > buy_threshold and position == 0:  # BUY
                position = 1
                entry_price = current_price
                trades.append({"type": "BUY", "price": current_price, "index": i})
                print(
                    f"BUY at {current_price:.2f} (raw: {action_raw:.4f}, final: {action_final:.4f})"
                )
            elif action_final < sell_threshold and position == 1:  # SELL
                position = 0
                exit_price = current_price
                pnl_pct = (exit_price - entry_price) / entry_price
                # 取引コストを考慮
                pnl_pct -= transaction_cost
                total_reward += pnl_pct * 100
                trades.append(
                    {
                        "type": "SELL",
                        "price": current_price,
                        "pnl_pct": pnl_pct,
                        "index": i,
                    }
                )
                print(
                    f"SELL at {current_price:.2f}, PnL: {pnl_pct:.4f} (raw: {action_raw:.4f}, final: {action_final:.4f})"
                )

        # 結果計算
        final_balance = initial_balance * (1 + total_reward / 100)
        profit = final_balance - initial_balance

        results = {
            "model": "sac_v436_signal_guided",
            "trades": len(trades) // 2,  # BUY/SELLペア
            "total_return_pct": total_reward,
            "final_balance": final_balance,
            "profit": profit,
            "signal_guidance_enabled": config["training"]["environment"][
                "reward_settings"
            ]["signal_guidance"]["enabled"],
            "signal_bonus_weight": config["training"]["environment"]["reward_settings"][
                "signal_guidance"
            ]["signal_bonus_weight"],
            "signal_penalty_weight": config["training"]["environment"][
                "reward_settings"
            ]["signal_guidance"]["signal_penalty_weight"],
        }

        print("\n📊 Backtest Results:")
        print(f"  Total Return: {total_reward:.2f}%")
        print(f"  Trades: {len(trades) // 2}")
        print(f"  Final Balance: ${final_balance:.2f}")
        print(f"  Profit: ${profit:.2f}")
        return results

    except Exception as e:
        print(f"❌ Error in v436 backtest: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("🚀 SAC v436 Signal Guided Backtest")
    print("=" * 50)

    # コマンドライン引数パーサー
    import argparse

    parser = argparse.ArgumentParser(description="SAC v436 Backtest")
    parser.add_argument(
        "--model", type=str, default="models/sac_model.zip", help="Path to model file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v436_signal_guided_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/btc_jpy_featured_dataset.csv",
        help="Path to data file",
    )

    args = parser.parse_args()

    # v436モデルのバックテスト実行
    result = run_v436_backtest(args.model, args.config, args.data)
    if result:
        print("\n✅ Backtest completed successfully!")
        print(f"Model: {result['model']}")
        print(f"Profit: ${result['profit']:.2f}")
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Trades: {result['trades']}")
        print(f"Signal Guidance: {result['signal_guidance_enabled']}")
        print(f"Signal Bonus Weight: {result['signal_bonus_weight']}")
        print(f"Signal Penalty Weight: {result['signal_penalty_weight']}")
    else:
        print("❌ Backtest failed")


if __name__ == "__main__":
    main()
