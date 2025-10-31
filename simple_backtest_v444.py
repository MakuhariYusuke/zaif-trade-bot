#!/usr/bin/env python3
"""
Simple SAC v444 Backtest
シンプルなバックテスト実行
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from ztb.utils.analysis_formatters import print_formatted_metrics


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
        # Load data - use the same data the model was trained on
        data_file = "data/btc_jpy_featured_dataset.csv"  # Same as training data
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded: {len(df)} rows")

        # 特徴量を手動で設定 (v444の環境設定に基づく)
        # v444は高度なレジーム適応を使用するため、より多くの特徴量が必要
        # モデルが10次元を期待しているので、最初の10個の数値特徴量を使用
        exclude_cols = ['Date', 'open', 'high', 'low', 'dividends', 'stock splits', 'regime', 'regime_confidence']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_features = [col for col in numeric_cols if col not in exclude_cols][:10]  # 最初の10個を使用

        print(f"✅ Using first 10 numeric features: {available_features}")
        print(f"   Total available numeric columns: {len(numeric_cols)}")

        if len(available_features) == 0:
            print("❌ No features available for backtest")
            return None

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
        max_steps = min(1100, len(df))
        for i in range(100, max_steps):  # 1000ステップ分確認
            obs = df[available_features].iloc[i].values.astype(np.float32)
            obs = obs.reshape(1, -1)
            action, _ = model.predict(obs, deterministic=True)
            action_raw = float(action[0])  # SAC outputs in [-1, 1] range
            action_mapped = action_raw  # No transformation needed for SAC
            action_values_raw.append(action_raw)
            action_values_mapped.append(action_mapped)

        action_values_raw = np.array(action_values_raw)
        action_values_mapped = np.array(action_values_mapped)

        print(f"   Collected {len(action_values_raw)} action samples")
        print(
            f"   Raw action stats - Min: {action_values_raw.min():.4f}, Max: {action_values_raw.max():.4f}, Mean: {action_values_raw.mean():.4f}, Std: {action_values_raw.std():.4f}"
        )
        print(
            f"   Mapped action stats - Min: {action_values_mapped.min():.4f}, Max: {action_values_mapped.max():.4f}, Mean: {action_values_mapped.mean():.4f}, Std: {action_values_mapped.std():.4f}"
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
            obs = df[available_features].iloc[i].values.astype(np.float32)
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

        # 最終ポジションの決済
        if position == 1:
            final_price = df["close"].iloc[-1]
            pnl = (final_price - entry_price) / entry_price * 100
            total_reward += pnl
            trades.append(
                {"type": "FINAL_SELL", "price": final_price, "pnl": pnl, "index": len(df)-1}
            )
            print(f"FINAL SELL at {final_price:.2f}, PnL: {pnl:.2f}%")

        # 結果の計算
        final_balance = initial_balance * (1 + total_reward / 100)
        total_return_pct = total_reward

        results = {
            "model_name": model_name,
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return_pct,
            "total_trades": len([t for t in trades if t["type"] in ["SELL", "FINAL_SELL"]]),
            "winning_trades": len([t for t in trades if t.get("pnl", 0) > 0 and t["type"] in ["SELL", "FINAL_SELL"]]),
            "losing_trades": len([t for t in trades if t.get("pnl", 0) < 0 and t["type"] in ["SELL", "FINAL_SELL"]]),
            "avg_win_pct": np.mean([t["pnl"] for t in trades if t.get("pnl", 0) > 0 and t["type"] in ["SELL", "FINAL_SELL"]]) if any(t.get("pnl", 0) > 0 for t in trades if t["type"] in ["SELL", "FINAL_SELL"]) else 0,
            "avg_loss_pct": np.mean([t["pnl"] for t in trades if t.get("pnl", 0) < 0 and t["type"] in ["SELL", "FINAL_SELL"]]) if any(t.get("pnl", 0) < 0 for t in trades if t["type"] in ["SELL", "FINAL_SELL"]) else 0,
            "max_drawdown_pct": 0,  # 簡易版なので計算しない
            "sharpe_ratio": 0,  # 簡易版なので計算しない
            "trades": trades
        }

        return results

    except Exception as e:
        print(f"❌ Error in backtest for {model_name}: {e}")
        return None


def main():
    """メイン関数"""
    print("🚀 Simple SAC v444 Backtest")

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

        print("✅ Results saved to backtest_results_sac_v444.json")

        # 目標チェック: 25% リターン改善
        if result["total_return_pct"] > 25:
            print("🎉 SUCCESS: Achieved 25% return improvement target!")
        else:
            print(f"⚠️  Current return: {result['total_return_pct']:.2f}%, Target: 25% improvement")

    else:
        print("❌ Backtest failed")


if __name__ == "__main__":
    main()