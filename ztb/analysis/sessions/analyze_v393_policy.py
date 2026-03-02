"""Analyze action probability distribution for v393 model"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.io.data_loader import DataLoader

MODEL_PATH = Path("models/ppo_session.zip")
DATA_PATH = Path("btc_jpy_real_dataset.csv")

def main() -> None:
    # Import torch lazily to avoid import-time ABI issues
    try:
        import torch as th
    except Exception:
        th = None  # type: ignore

    df = DataLoader.load_csv_optimized(str(DATA_PATH))
    base_env = create_env_from_model_path(str(MODEL_PATH), df)
    vec_env = DummyVecEnv([lambda: base_env])

    model = MaskablePPO.load(str(MODEL_PATH), env=vec_env)
    device = model.policy.device

    obs = vec_env.reset()
    mask = vec_env.env_method("get_action_masks")[0]

    probs_hold = []
    probs_buy = []
    probs_sell = []
    logits_log = []
    argmax_actions = []

    mask_counts = np.zeros(3, dtype=int)

    for step in range(len(df)):
        obs_tensor = th.as_tensor(obs, device=device, dtype=th.float32)
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.ndim == 1:
            mask_np = mask_np.reshape(1, -1)

        mask_counts += mask_np[0].astype(int)

        distribution = model.policy.get_distribution(obs_tensor, action_masks=mask_np)
        dist = distribution.distribution
        if dist is None:
            raise RuntimeError("Distribution object is None")
        probs = dist.probs.detach().cpu().numpy()[0]
        logits = dist.logits.detach().cpu().numpy()[0]

        probs_hold.append(probs[0])
        probs_buy.append(probs[1])
        probs_sell.append(probs[2])
        logits_log.append(logits)
        argmax_actions.append(int(np.argmax(probs)))

        action_array: NDArray[np.int64] = np.array([np.argmax(probs)])
        obs, reward, done, info = vec_env.step(action_array)
        if done[0]:
            break
        mask = vec_env.env_method("get_action_masks")[0]

    probs_hold_array: NDArray[np.float64] = np.array(probs_hold)
    probs_buy_array: NDArray[np.float64] = np.array(probs_buy)
    probs_sell_array: NDArray[np.float64] = np.array(probs_sell)
    argmax_actions_array: NDArray[np.int64] = np.array(argmax_actions)
    logits_log_array: NDArray[np.float64] = np.array(logits_log)

    print("=" * 70)
    print("v393 Action Probability Analysis (deterministic logits)")
    print("=" * 70)
    print(f"Samples analyzed: {len(probs_hold_array)}")
    print(
        f"P(HOLD)  mean={probs_hold_array.mean():.3f}, median={np.median(probs_hold_array):.3f}, min={probs_hold_array.min():.3f}, max={probs_hold_array.max():.3f}"
    )
    print(
        f"P(BUY)   mean={probs_buy_array.mean():.3f}, median={np.median(probs_buy_array):.3f}, min={probs_buy_array.min():.3f}, max={probs_buy_array.max():.3f}"
    )
    print(
        f"P(SELL)  mean={probs_sell_array.mean():.3f}, median={np.median(probs_sell_array):.3f}, min={probs_sell_array.min():.3f}, max={probs_sell_array.max():.3f}"
    )
    print(f"Logits mean: {logits_log_array.mean(axis=0)}")
    print(f"Logits std:  {logits_log_array.std(axis=0)}")
    print(f"Action mask availability counts: {mask_counts} (HOLD, BUY, SELL)")
    print()
    print("Argmax action distribution (deterministic policy):")
    unique, counts = np.unique(argmax_actions_array, return_counts=True)
    for action, count in zip(unique, counts):
        label = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, str(action))
        print(
            f"  {label}: {count}/{len(argmax_actions)} ({count/len(argmax_actions)*100:.1f}%)"
        )

    print("=" * 70)

if __name__ == "__main__":
    main()
