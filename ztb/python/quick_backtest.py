#!/usr/bin/env python3
"""
簡潔なバックテストスクリプト（収益性評価用）
Python 3.11環境での動作確認済み
SACモデル対応版
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from ztb.utils.file_utils import get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import talib
from sb3_contrib import MaskablePPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.training.core.feature_schema_manager import FeatureSchemaManager
from ztb.io.data_loader import DataLoader


def generate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate basic features for SAC model (5-dimensional observation space)
    """
    df = df.copy()

    # Basic price features
    if "price_change" not in df.columns:
        df["price_change"] = df["close"].pct_change()

    if "volume_change" not in df.columns:
        df["volume_change"] = df["volume"].pct_change()

    # Fill NaN values
    df = df.fillna(0)

    # Return only the 5 basic features that SAC model expects
    basic_features = ["close", "volume", "price_change", "volume_change"]
    # Add one more feature to make it 5 dimensions (maybe a simple moving average)
    df["close_ma5"] = df["close"].rolling(5).mean()
    df = df.fillna(0)

    return df[["close", "volume", "price_change", "volume_change", "close_ma5"]]


def generate_required_features(
    df: pd.DataFrame, required_features: list
) -> pd.DataFrame:
    # Basic price features
    if "close" not in df.columns:
        df["close"] = df["close"]  # Already exists

    if "price_change" in required_features and "price_change" not in df.columns:
        df["price_change"] = df["close"].pct_change()

    if "volume_change" in required_features and "volume_change" not in df.columns:
        df["volume_change"] = df["volume"].pct_change()

    # Technical indicators (only generate what's commonly available)
    try:
        if "rsi_14" in required_features and "rsi_14" not in df.columns:
            df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)

        if (
            all(x in required_features for x in ["macd", "macd_signal", "macd_hist"])
            and "macd" not in df.columns
        ):
            macd, macdsignal, macdhist = talib.MACD(df["close"])
            df["macd"] = macd
            df["macd_signal"] = macdsignal
            df["macd_hist"] = macdhist

        # Bollinger Bands
        if (
            all(x in required_features for x in ["bb_upper", "bb_middle", "bb_lower"])
            and "bb_upper" not in df.columns
        ):
            upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
            df["bb_upper"] = upper
            df["bb_middle"] = middle
            df["bb_lower"] = lower
            if "bb_width" in required_features:
                df["bb_width"] = (upper - lower) / middle

        # Stochastic
        if (
            all(x in required_features for x in ["stoch_k", "stoch_d"])
            and "stoch_k" not in df.columns
        ):
            slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"])
            df["stoch_k"] = slowk
            df["stoch_d"] = slowd

        if "williams_r" in required_features and "williams_r" not in df.columns:
            df["williams_r"] = talib.WILLR(df["high"], df["low"], df["close"])

        # Other indicators
        if "atr_14" in required_features and "atr_14" not in df.columns:
            df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

        if "cci_14" in required_features and "cci_14" not in df.columns:
            df["cci_14"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)

        if "mfi_14" in required_features and "mfi_14" not in df.columns:
            df["mfi_14"] = talib.MFI(
                df["high"], df["low"], df["close"], df["volume"], timeperiod=14
            )

        if "roc_12" in required_features and "roc_12" not in df.columns:
            df["roc_12"] = talib.ROC(df["close"], timeperiod=12)

        if "mom_10" in required_features and "mom_10" not in df.columns:
            df["mom_10"] = talib.MOM(df["close"], timeperiod=10)

    except Exception as e:
        print(f"Warning: Some technical indicators failed to generate: {e}")

    # Generate dummy features for feature_XX patterns (simplified approach)
    existing_features = set(df.columns)
    missing_features = [f for f in required_features if f not in existing_features]

    if missing_features:
        print(f"Generating {len(missing_features)} dummy features...")
        for feature in missing_features:
            if feature.startswith("feature_"):
                # Create simple dummy features
                df[feature] = np.random.randn(len(df)) * 0.1
            elif feature.startswith("ichimoku_"):
                # Skip Ichimoku for now
                df[feature] = df["close"] * 0.98  # Dummy value

    # Fill NaN values
    df = df.fillna(0)

    return df


def run_quick_backtest(
    model_path: str, data_path: str, episodes: int = 10, model_type: str = "auto"
):
    """
    モデルの収益性を素早く評価
    model_type: "ppo", "sac", or "auto" (auto-detect from model file)
    """
    model_path_obj = Path(model_path)
    model_name = model_path_obj.stem  # モデル名を取得
    print(f"\n{'='*80}")
    print(f"Quick Backtest: {model_name}")
    print(f"{'='*80}\n")

    # データ読み込み
    df = DataLoader.load_csv_optimized(data_path)
    print(f"Data: {len(df):,} rows, {len(df.columns)} columns")

    # 特徴量エンジニアリング適用（スキーマに必要な特徴量を生成）
    if len(df.columns) < 10:  # 基本OHLCVのみの場合
        print("Applying feature engineering for required features...")

        # モデルが5次元観測空間を持つため、基本特徴量のみを使用
        if model_type == "sac":
            print(
                "SAC model uses 5-dimensional observation space, using basic OHLCV features"
            )
            required_features = ["close", "volume", "price_change", "volume_change"]
            # 基本的な特徴量のみ生成
            df = generate_basic_features(df)
        else:
            # スキーマから必要な特徴量を取得
            try:
                manager = FeatureSchemaManager("v434_1_combined_learning_model")
                metadata = manager.load_schema()
                required_features = metadata.feature_names
                print(f"Required features from schema: {len(required_features)}")
            except:
                # フォールバック: 基本的な特徴量
                required_features = [
                    "close",
                    "volume",
                    "price_change",
                    "volume_change",
                    "rsi_14",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                ]
                print("Using fallback feature list")

            df = generate_required_features(df, required_features)
        print(f"Features engineered: {len(df.columns)} features")
    else:
        print("Using pre-engineered features")

    print(f"Final data: {len(df):,} rows, {len(df.columns)} features")

    # 環境作成（モデルタイプに応じて異なる方法を使用）
    if model_type == "sac":
        # SACの場合はスキーマを使用せず、直接HeavyTradingEnvを作成（連続行動）
        from ztb.trading.environment.environment import HeavyTradingEnv

        base_env = HeavyTradingEnv(
            df=df, config={"random_start": False, "use_continuous_actions": True}
        )
        print(
            f"Environment: {base_env.observation_space.shape[0]} features (direct HeavyTradingEnv, continuous actions)"
        )
    else:
        # PPOの場合はスキーマベースの環境作成を使用
        base_env = create_env_from_model_path(str(model_path_obj), df)
        print(f"Environment: {base_env.observation_space.shape[0]} features")

    # v434.2報酬設定の適用（SACの場合のみ）
    if model_type == "sac":
        try:
            import json
            from pathlib import Path

            # v434.2報酬設定読み込み
            config_path = Path("../../../config/sac_v434_2_reward_config.json")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    reward_config = json.load(f)

                # 報酬設定を適用
                if hasattr(base_env, "reward_calculator") and hasattr(
                    base_env.reward_calculator, "reward_settings"
                ):
                    for key, value in reward_config.items():
                        if not key.startswith("_"):
                            base_env.reward_calculator.reward_settings[key] = value

                    print("✅ Applied v434.2 enhanced reward function:")
                    for improvement in reward_config.get("_improvements", []):
                        print(f"   • {improvement}")
                else:
                    print("⚠️  Could not apply v434.2 reward settings")
            else:
                print("⚠️  v434.2 config not found, using default reward function")

        except Exception as e:
            print(f"⚠️  Failed to apply v434.2 reward settings: {e}")

    # モデルタイプの自動判定

    # モデルタイプの自動判定
    if model_type == "auto":
        try:
            # SACモデルとして読み込みを試行
            model = SAC.load(str(model_path))
            model_type = "sac"
            print("Detected SAC model")
        except:
            try:
                # PPOモデルとして読み込みを試行
                model = MaskablePPO.load(str(model_path))
                model_type = "ppo"
                print("Detected PPO model")
            except:
                raise ValueError("Could not load model as SAC or PPO")

    # モデル読み込み（SAC/PPO統一）
    env = DummyVecEnv([lambda: base_env])  # 両方のモデルタイプでVecEnvを使用
    if model_type == "sac":
        model = SAC.load(str(model_path), env=env)
        print("SAC Model loaded (VecEnv)")
    elif model_type == "ppo":
        model = MaskablePPO.load(str(model_path), env=env)
        print("PPO Model loaded (VecEnv)")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print("Model loaded\n")

    # バックテスト実行
    episode_rewards = []
    episode_returns = []
    total_trades = 0

    # 環境から初期ポートフォリオ値を取得
    initial_portfolio_value = base_env.initial_portfolio_value

    for ep in range(episodes):
        obs = env.reset()  # VecEnv APIを使用（両方のモデルタイプで統一）

        done = False
        ep_reward = 0.0
        ep_trades = 0

        while not done:
            if model_type == "ppo":
                # アクションマスク取得（VecEnvから）
                action_masks = np.array([base_env.action_mask()])

                # 予測（決定的）
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:  # SAC
                # SAC予測（決定的、連続アクション）
                action, _ = model.predict(obs, deterministic=True)

            # VecEnvステップ（両方のモデルタイプで統一）
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward

            # トレード回数カウント
            if model_type == "ppo":
                if action[0] != 0:  # HOLD以外
                    ep_trades += 1
            else:  # SAC
                if abs(action[0]) > 0.1:  # 閾値以上のアクション
                    ep_trades += 1

            if done:
                break

        # エピソード統計
        # 🔧 CRITICAL FIX: エピソード終了時にポジションを強制クローズ
        # resetではポジションを単に0にするだけでPnLを実現しないため、
        # ここで明示的にクローズしてrealized PnLを確定させる
        if base_env.position != 0:
            final_close_pnl = base_env.position_manager.close_position(
                base_env.current_step
            )
            base_env._sync_from_position_manager()
            print(f"  ⚠️  Forced position close: PnL = {final_close_pnl:+.2f} 円")

        # 最終ポートフォリオ値を取得（realized PnL のみ）
        final_value = base_env.initial_portfolio_value + base_env.realized_pnl
        return_pct = (
            (final_value - initial_portfolio_value) / initial_portfolio_value
        ) * 100

        episode_rewards.append(ep_reward)
        episode_returns.append(return_pct)
        total_trades += ep_trades

        print(
            f"Episode {ep+1:2d}: Reward={ep_reward:7.2f}, Return={return_pct:6.2f}%, Trades={ep_trades:3d}, Final={final_value:,.2f}円"
        )

    # サマリー
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(
        f"Average Reward:  {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):6.2f}"
    )
    print(
        f"Average Return:  {np.mean(episode_returns):6.2f}% ± {np.std(episode_returns):5.2f}%"
    )
    print(f"Best Return:     {np.max(episode_returns):6.2f}%")
    print(f"Worst Return:    {np.min(episode_returns):6.2f}%")
    print(f"Total Trades:    {total_trades}")
    print(f"Trades/Episode:  {total_trades/episodes:.1f}")
    print(f"{'='*80}\n")

    # 結果を保存
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "best_return": float(np.max(episode_returns)),
        "worst_return": float(np.min(episode_returns)),
        "total_trades": int(total_trades),
        "trades_per_episode": float(total_trades / episodes),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_returns": [float(r) for r in episode_returns],
        "model_type": model_type,
        "data_path": str(data_path),
    }

    # 結果をJSONファイルに保存
    results_file = (
        f"backtest_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, required=True, help="Data path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "ppo", "sac"],
        help="Model type (auto-detect if not specified)",
    )
    args = parser.parse_args()

    run_quick_backtest(args.model, args.data, args.episodes, args.model_type)
