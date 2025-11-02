#!/usr/bin/env python3
"""
Fixed SAC v444 Backtest - Uses same environment as training
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ztb.utils.normalization import load_scaler

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# suppress overly verbose internal DEBUG logs from ztb modules during diagnostics
logging.getLogger("ztb").setLevel(logging.INFO)

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.v4xx_config_converter import V4XXConfigConverter


def run_backtest():
    """学習時と同じ環境設定でバックテストを実行"""
    print("🚀 Fixed SAC v444 Backtest")

    try:
        # 設定読み込み (学習時と同じ)
        config_path = "config/sac_v444_advanced_regime_adaptation_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print("✅ Config loaded successfully")

        # データ読み込み (学習時と同じ)
        data_file = "data/btc_jpy_featured_dataset.csv"
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded: {len(df)} rows")

        # configを学習時と同じ形式に変換
        converter = V4XXConfigConverter()
        unified_config = converter.convert_to_unified(config)

        # 環境設定を取得 (変換後のconfigを使用)
        env_config = unified_config.get("environment", {})

        # SACモデル用にcontinuous action spaceを強制設定
        env_config["use_continuous_actions"] = True

        # 学習時と同じcurriculum_stageを設定
        env_config["curriculum_stage"] = "forced_balance"

        # EnvironmentConfigオブジェクト作成
        env_config_obj = EnvironmentConfig(**env_config)

        # 報酬設定を取得
        reward_settings = unified_config.get("reward_settings", {})

        # 学習時と同じcurriculum_stageを設定
        reward_settings["curriculum_stage"] = "forced_balance"

        # reward clipping範囲を広げてbalance_penaltyが効くようにする
        reward_settings["reward_clip_min"] = -10000.0
        reward_settings["reward_clip_max"] = 10000.0

        # 環境作成 (学習時と同じ方法で)
        base_env = HeavyTradingEnv(
            df=df,
            config=env_config_obj,
        )
        print("✅ Base environment created with same settings as training")

        # VecNormalize 用に DummyVecEnv でラップし、訓練時の正規化統計を適用する
        vec = DummyVecEnv([lambda: base_env])
        vec_norm = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

        # ウォームアップを実行して正規化統計を収集
        print("🔄 Warming up environment to collect normalization statistics...")
        warmup_steps = 5000  # 十分なステップ数で統計を収集
        obs = vec_norm.reset()
        for _ in range(warmup_steps):
            # ランダムアクションで環境をステップ
            action = vec_norm.action_space.sample()
            obs, _, done, _ = vec_norm.step([action])
            if done:
                obs = vec_norm.reset()
        print("✅ Environment warmed up, normalization statistics collected")

        # 正規化統計を保存
        from ztb.utils.normalization import NormalizationStats

        try:
            # 特徴量名を取得（環境から）
            feature_names = getattr(base_env, "feature_names", None)
            if feature_names is None:
                # デフォルトの特徴量名を生成
                obs_shape = vec_norm.observation_space.shape[0]
                feature_names = [f"feature_{i}" for i in range(obs_shape)]

            # NormalizationStats を作成
            stats = NormalizationStats.from_vec_normalize(
                vec_norm, feature_names=feature_names
            )

            # 保存
            stats.save(Path("models") / "scaler_v444_regenerated.npz")
            print(
                "✅ Saved regenerated normalization statistics to models/scaler_v444_regenerated.npz"
            )

            # 統計を適用
            stats.apply_to_vec_normalize(vec_norm, strict=False)
            print("✅ Applied regenerated normalization stats to VecNormalize wrapper")
        except Exception as e:
            print(f"⚠️ Failed to regenerate normalization stats: {e}")

        # モデルディレクトリに保存された scaler.npz があれば適用する（フォールバック）
        try:
            stats = load_scaler(Path("models"), strict=False)
            if stats is not None:
                stats.apply_to_vec_normalize(vec_norm, strict=False)
                print("✅ Applied saved normalization stats to VecNormalize wrapper")
            else:
                print(
                    "⚠️ No normalization stats found (models/scaler.npz). Proceeding without applying stats."
                )
        except Exception as e:
            print(f"⚠️ Failed to apply normalization stats: {e}")

        # 最終的にモデルには正規化済み vec 環境を使う
        env = vec_norm
        print("✅ Environment wrapped with VecNormalize")

        # モデル読み込み
        model_path = "models/sac_v444_advanced_regime_adaptation.zip"
        model = SAC.load(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"   Observation space: {model.observation_space}")
        print(f"   Action space: {model.action_space}")

        # アクション分析
        print("Starting action analysis...")
        # VecNormalize の reset は配列を返す (batched)
        obs = env.reset()
        # obs は形 (1, n) なので model.predict に渡すために 1 次元配列に変換
        if isinstance(obs, (list, tuple)):
            # 古い env.reset() の戻り値に対応（念のため）
            obs = obs[0]

        actions_raw = []
        actions_discrete = []
        # use a longer run for better statistics
        max_steps = min(1000, len(df))
        print(f"Running for {max_steps} steps to analyze action distribution...")
        for i in range(max_steps):
            # VecEnv が返す obs はバッチ次元付き (1, n)。predict に渡す obs を作る
            obs_for_predict = (
                obs[0] if hasattr(obs, "shape") and obs.shape[0] == 1 else obs
            )

            # デバッグ: pre-normalized (実際は VecNormalize 後の obs) を最初の数ステップで出力
            if i < 5:
                try:
                    print(
                        f"DEBUG obs_for_predict (step {i}): {np.array(obs_for_predict).ravel()}"
                    )
                except Exception:
                    pass

            # 予測: deterministic=False で確率的サンプリングを使用
            action_det, _ = model.predict(obs_for_predict, deterministic=False)
            action_sto, _ = model.predict(
                obs_for_predict, deterministic=False
            )  # 同じく確率的

            action_raw = float(action_det[0])
            action_raw_sto = float(action_sto[0])

            # 離散アクションに変換 (学習時と同じロジック)
            def to_discrete(a):
                if a > 0.3333:
                    return 1
                elif a < -0.3333:
                    return -1
                else:
                    return 0

            discrete_action = to_discrete(action_raw)
            discrete_action_sto = to_discrete(action_raw_sto)

            actions_raw.append(action_raw)
            actions_discrete.append(discrete_action)

            # 最初の数ステップは詳細ログ
            if i < 5:
                print(f"Step {i}: action={action_raw:.4f}({discrete_action})")

            # 次のステップへ (continuous actionを使用)
            step_action = (
                np.array([[action_raw]])
                if np.ndim(action_raw) == 0
                else np.array([action_raw])
            )
            step_obs = env.step(step_action)
            # VecEnv.step -> (obs, rewards, dones, infos)
            if len(step_obs) == 4:
                obs, rewards, dones, infos = step_obs
                reward = float(rewards[0])
                terminated = bool(dones[0])
                truncated = False
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
            else:
                # 互換性フォールバック
                obs, reward, terminated, truncated, info = step_obs

            # 100ステップごとにreward_calculatorをリセットしてaction historyをクリア
            if i > 0 and i % 100 == 0:
                # VecNormalize ラッパの下の env にアクセスする
                underlying_env = (
                    getattr(env, "venv", None) or getattr(env, "envs", None) or env
                )
                try:
                    # attempt to find base env with reward_calculator
                    base = None
                    if hasattr(underlying_env, "envs") and len(underlying_env.envs) > 0:
                        base = underlying_env.envs[0]
                    elif hasattr(underlying_env, "env"):
                        base = underlying_env.env
                    if base is not None and hasattr(base, "reward_calculator"):
                        base.reward_calculator.reset()
                        print(f"🔄 Reset reward_calculator at step {i}")
                except Exception:
                    pass

            if terminated or truncated:
                break

        # 結果分析
        actions_raw = np.array(actions_raw)
        actions_discrete = np.array(actions_discrete)

        # 離散アクションの分布を計算
        hold_count = np.sum(actions_discrete == 0)
        buy_count = np.sum(actions_discrete == 1)
        sell_count = np.sum(actions_discrete == -1)

        total_actions = len(actions_discrete)

        print("\n📊 Action Distribution Results:")
        print(f"Total actions analyzed: {total_actions}")
        print(f"HOLD: {hold_count} ({hold_count/total_actions*100:.1f}%)")
        print(f"BUY: {buy_count} ({buy_count/total_actions*100:.1f}%)")
        print(f"SELL: {sell_count} ({sell_count/total_actions*100:.1f}%)")

        print("\n📈 Raw Action Stats:")
        print(f"Mean: {actions_raw.mean():.4f}")
        print(f"Std: {actions_raw.std():.4f}")
        print(f"Min: {actions_raw.min():.4f}")
        print(f"Max: {actions_raw.max():.4f}")

        return True

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_backtest()
    sys.exit(0 if success else 1)
