"""
v393モデルでStochastic vs Deterministic推論を比較
"""

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.utils.data_utils import load_csv_data_optimized
from pathlib import Path

# モデルとデータ読み込み
model_path = Path("models/ppo_session.zip")
df = load_csv_data_optimized("btc_jpy_real_dataset.csv")

# 環境作成（スキーマベース）
base_env = create_env_from_model_path(str(model_path), df)
env = DummyVecEnv([lambda: base_env])

# モデル読み込み
model = MaskablePPO.load(str(model_path), env=env)

print("=" * 70)
print("v393モデル: Stochastic vs Deterministic推論比較")
print("=" * 70)

# Deterministicテスト
print("\n📊 Deterministic推論（quick_backtestと同じ）:")
obs = env.reset()
action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    action_counts[int(action[0])] += 1
    obs, reward, done, info = env.step(action)
    if done[0]:
        break

total = sum(action_counts.values())
print(f"  HOLD: {action_counts[0]}/{total} ({action_counts[0]/total*100:.1f}%)")
print(f"  BUY:  {action_counts[1]}/{total} ({action_counts[1]/total*100:.1f}%)")
print(f"  SELL: {action_counts[2]}/{total} ({action_counts[2]/total*100:.1f}%)")

# Stochasticテスト
print("\n📊 Stochastic推論（訓練時と同じ）:")
obs = env.reset()
action_counts = {0: 0, 1: 0, 2: 0}
for _ in range(100):
    action, _ = model.predict(obs, deterministic=False)
    action_counts[int(action[0])] += 1
    obs, reward, done, info = env.step(action)
    if done[0]:
        break

total = sum(action_counts.values())
print(f"  HOLD: {action_counts[0]}/{total} ({action_counts[0]/total*100:.1f}%)")
print(f"  BUY:  {action_counts[1]}/{total} ({action_counts[1]/total*100:.1f}%)")
print(f"  SELL: {action_counts[2]}/{total} ({action_counts[2]/total*100:.1f}%)")

print("\n" + "=" * 70)
print("📌 訓練時のAction分布（ログより）:")
print("   HOLD: 230/256 = 89.8%")
print("   BUY:  16/256 = 6.3%")
print("   SELL: 10/256 = 3.9%")
print("=" * 70)
