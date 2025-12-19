#!/usr/bin/env python3
"""
Fixed SAC v444 Backtest - Uses same environment as training
"""
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

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
from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def run_backtest():
    """学習時と同じ環境設定でバックテストを実行"""
    print("🚀 Fixed SAC v444 Backtest")

    try:
        # 設定ファイルのパス
        config_path = "config/sac_v444_3_balanced_penalty_scale_200.json"
        
        # V4XXUnifiedTrainerを使って環境を作成（学習時と同じ）
        print(f"Loading config: {config_path}")
        trainer = V4XXUnifiedTrainer(config_path=config_path)
        
        # 設定を検証
        if not trainer.validate_config():
            raise ValueError("Configuration validation failed")
        
        # トレーニングデータを取得するためにtrainerの内部データを使用
        # trainer.configからデータパスを取得
        data_config = trainer.config.get("training", {}).get("data_config", {})
        data_path = data_config.get("data_path", "data/btc_jpy_featured_dataset.csv")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"✅ Data loaded: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df)} rows")
        
        # trainerを使って環境を作成（学習時と同じ方法）
        # trainerの内部メソッドを呼び出して環境を取得
        from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
        
        # SAC trainerを作成
        sac_trainer = SACTrainer(trainer.config, logger=trainer.logger)
        
        # 環境設定を取得
        env_config = trainer.config.get("training", {}).get("environment", {})
        actual_env_config = env_config.get("config", env_config)
        
        # EnvironmentConfigに変換
        from ztb.trading.environment.utils.config import EnvironmentConfig
        if isinstance(actual_env_config, EnvironmentConfig):
            env_config_obj = actual_env_config
        elif isinstance(actual_env_config, dict):
            env_config_obj = EnvironmentConfig.from_dict(actual_env_config)
        else:
            env_config_obj = EnvironmentConfig.from_dict(trainer.config)
        
        # HeavyTradingEnvを作成
        env = HeavyTradingEnv(
            df=df,
            config=env_config_obj,
            optimizer_tracker=trainer.optimizer_tracker,
        )
        
        # 市場レジーム適応を有効化（設定されている場合）
        market_regime_config = trainer.config.get("training", {}).get("market_regime_adaptation", {})
        if market_regime_config.get("enabled", False):
            try:
                from ztb.analysis.market_regime_classifier import MarketRegimeClassifier
                regime_classifier = MarketRegimeClassifier(config=market_regime_config)
                env.enable_market_regime_adaptation(
                    regime_classifier=regime_classifier,
                    adaptation_config=market_regime_config,
                )
                print("✅ Market regime adaptation enabled in environment")
            except Exception as e:
                print(f"⚠️ Failed to enable market regime adaptation: {e}")
        
        print(f"✅ Base environment created with observation space: {env.observation_space}")
        print(f"✅ Base environment created with action space: {env.action_space}")

        # VecNormalize でラップ
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.vec_env import VecNormalize
        
        vec = DummyVecEnv([lambda: env])
        vec_norm = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # モデルディレクトリに保存された scaler.npz があれば適用する
        try:
            from ztb.utils.normalization import NormalizationStats
            
            # まずトレーニング時に保存されたscaler_v444_regenerated.npzを試す
            scaler_path = Path("models") / "scaler_v444_regenerated.npz"
            if scaler_path.exists():
                stats = NormalizationStats.load(scaler_path)
                stats.apply_to_vec_normalize(vec_norm, strict=False)
                print(f"✅ Applied regenerated normalization stats from {scaler_path}")
            else:
                # フォールバックとして通常のscaler.npzを試す
                stats = load_scaler(Path("models"), strict=False)
                if stats is not None:
                    stats.apply_to_vec_normalize(vec_norm, strict=False)
                    print("✅ Applied saved normalization stats to VecNormalize wrapper")
                else:
                    print(
                        "⚠️ No normalization stats found. Proceeding without applying stats."
                    )
        except Exception as e:
            print(f"⚠️ Failed to apply normalization stats: {e}")

        # モデル読み込み
        model_path = "models/sac_v444_3_balanced_penalty_scale_200.zip"
        model = SAC.load(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"   Observation space: {model.observation_space}")
        print(f"   Action space: {model.action_space}")

        # アクション分析
        print("Starting action analysis...")
        # VecNormalize の reset は配列を返す (batched)
        obs = vec_norm.reset()
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
            step_obs = vec_norm.step(step_action)
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
                    getattr(vec_norm, "venv", None) or getattr(vec_norm, "envs", None) or vec_norm
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
