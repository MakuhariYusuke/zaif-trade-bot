"""
Stochastic推論モードでのバックテスト
訓練時vs推論時の乖離を調査
"""

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_model_path


def main():
    model_path = "models/ppo_profitable_v392_bugfix.zip"
    data_path = "btc_jpy_real_dataset.csv"
    episodes = 10

    print(f"🔍 Stochastic Inference Test: {model_path}")
    print(f"{'='*80}\n")

    # データ読込
    df = pd.read_csv(data_path)
    print(f"Data: {len(df)} rows\n")

    # 環境作成
    base_env = create_env_from_model_path(model_path, df)
    env = DummyVecEnv([lambda: base_env])

    # モデル読み込み
    model = MaskablePPO.load(str(model_path), env=env)
    print("Model loaded\n")

    # Deterministicモードとstochasticモードの両方でテスト
    for mode_name, deterministic in [("Deterministic", True), ("Stochastic", False)]:
        print(f"\n{'='*80}")
        print(f"🎲 Mode: {mode_name} (deterministic={deterministic})")
        print(f"{'='*80}\n")

        action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
        total_steps = 0

        for ep in range(episodes):
            obs = env.reset()
            done = False
            ep_actions = {0: 0, 1: 0, 2: 0}

            while not done:
                # Action masking
                action_masks = np.array([base_env.get_legal_actions()])

                # 予測（deterministicモード切り替え）
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )

                action_val = action[0] if isinstance(action, np.ndarray) else action
                action_counts[action_val] += 1
                ep_actions[action_val] += 1
                total_steps += 1

                obs, reward, done, info = env.step(action)
                done = done[0] if isinstance(done, np.ndarray) else done

            print(
                f"Episode {ep+1:2d}: HOLD={ep_actions[0]:3d}, BUY={ep_actions[1]:3d}, SELL={ep_actions[2]:3d}"
            )

        # 結果サマリー
        print(f"\n{'='*80}")
        print(f"Summary ({mode_name} mode)")
        print(f"{'='*80}")
        print(f"Total steps: {total_steps}")
        print(
            f"HOLD:  {action_counts[0]:4d} ({action_counts[0]/total_steps*100:5.1f}%)"
        )
        print(
            f"BUY:   {action_counts[1]:4d} ({action_counts[1]/total_steps*100:5.1f}%)"
        )
        print(
            f"SELL:  {action_counts[2]:4d} ({action_counts[2]/total_steps*100:5.1f}%)"
        )
        print("")


if __name__ == "__main__":
    main()
