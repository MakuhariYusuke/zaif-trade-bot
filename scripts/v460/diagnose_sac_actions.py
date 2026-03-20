"""SAC モデルの行動出力を診断するスクリプト.

問題: 全50Kステップ × 4seeds で ROI=0.0000
仮説: SACの出力が閾値内に収まり全てHOLDに変換されている

使用法:
  python scripts/v460/diagnose_sac_actions.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import SAC

from scripts.v460.lib.tasks.sac_train import _create_training_env, section
from ztb.training.sac.runtime import extract_roi_from_env


def diagnose(model_path: str, config_path: str, n_steps: int = 2000) -> None:
    """モデルの行動分布を診断."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # データ読み込み
    from scripts.v460.lib.data_loader import load_parquet
    data_cfg = section(cfg, "data")
    ohlcv_path = data_cfg.get("ohlcv_path", "")
    train_end_index = data_cfg.get("train_end_index")
    df = load_parquet(ohlcv_path, max_rows=train_end_index)

    # 環境構築
    env, env_info = _create_training_env(df, cfg)
    print(f"obs_dim={env_info['obs_dim']}, action_dim={env_info['action_dim']}")

    # 閾値確認
    threshold = getattr(env, "action_threshold", "N/A")
    neg_threshold = getattr(env, "negative_action_threshold", "N/A")
    print(f"action_threshold={threshold}, negative_action_threshold={neg_threshold}")

    # モデルロード
    model = SAC.load(model_path, env=env)

    # モデルの内部状態を確認
    print(f"\n=== Model Info ===")
    print(f"  model type: {type(model)}")
    print(f"  model.__module__: {type(model).__module__}")
    print(f"  has action_space: {hasattr(model, 'action_space')}")
    print(f"  has policy: {hasattr(model, 'policy')}")

    # action_space を env から取得
    act_space = env.action_space
    print(f"  env.action_space: {act_space}")

    # policy 確認
    if hasattr(model, 'policy'):
        policy = model.policy
        print(f"  policy type: {type(policy)}")
        if hasattr(policy, 'actor'):
            actor = policy.actor
            print(f"  actor type: {type(actor)}")
            for name, param in list(actor.named_parameters())[:10]:
                print(f"  actor.{name}: shape={param.shape}, mean={param.data.mean():.6f}, std={param.data.std():.6f}")
    
    # 単一の predict の出力型を確認
    obs_test, _ = env.reset()
    action_raw, _state = model.predict(obs_test, deterministic=True)
    print(f"\n  predict det type: {type(action_raw)}, dtype: {getattr(action_raw, 'dtype', 'N/A')}, value: {action_raw}")
    action_sto, _state = model.predict(obs_test, deterministic=False)
    print(f"  predict sto type: {type(action_sto)}, dtype: {getattr(action_sto, 'dtype', 'N/A')}, value: {action_sto}")
    
    # policy.forward で直接確認
    if hasattr(model, 'policy') and hasattr(model.policy, 'actor') and hasattr(model.policy.actor, 'get_action_dist_params'):
        import torch
        obs_tensor = torch.as_tensor(obs_test, dtype=torch.float32).unsqueeze(0).to(model.device)
        with torch.no_grad():
            mean_actions, log_std, kwargs = model.policy.actor.get_action_dist_params(obs_tensor)
            print(f"  actor mean_actions: {mean_actions}")
            print(f"  actor log_std: {log_std}")

    # 行動分析
    obs, _ = env.reset()
    raw_actions = []
    discrete_actions = []
    rewards = []

    for step in range(n_steps):
        # deterministic
        action_det, _ = model.predict(obs, deterministic=True)
        # stochastic
        action_sto, _ = model.predict(obs, deterministic=False)

        raw_val = float(action_det[0]) if hasattr(action_det, '__len__') else float(action_det)
        raw_actions.append(raw_val)

        # step with deterministic action
        obs, reward, terminated, truncated, info = env.step(action_det)
        rewards.append(float(reward))

        # infer discrete action from info
        actual_action = info.get("actual_action", info.get("action", -99))
        discrete_actions.append(int(actual_action))

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    raw_actions = np.array(raw_actions)
    rewards_arr = np.array(rewards)

    print(f"\n=== Action Distribution (n={len(raw_actions)}) ===")
    print(f"  raw_action: mean={raw_actions.mean():.6f}, std={raw_actions.std():.6f}")
    print(f"  raw_action: min={raw_actions.min():.6f}, max={raw_actions.max():.6f}")
    print(f"  raw_action: median={np.median(raw_actions):.6f}")

    # Percentiles
    for p in [1, 5, 10, 25, 75, 90, 95, 99]:
        print(f"  P{p:02d}={np.percentile(raw_actions, p):.6f}", end="")
    print()

    # Threshold analysis
    above = np.sum(raw_actions > float(threshold))
    below = np.sum(raw_actions < float(neg_threshold))
    hold_zone = len(raw_actions) - above - below
    print(f"\n=== Threshold Analysis (±{threshold}) ===")
    print(f"  BUY  (>{threshold}): {above} ({100*above/len(raw_actions):.1f}%)")
    print(f"  HOLD (in zone):      {hold_zone} ({100*hold_zone/len(raw_actions):.1f}%)")
    print(f"  SELL (<{neg_threshold}): {below} ({100*below/len(raw_actions):.1f}%)")

    # Discrete action counts
    from collections import Counter
    action_counts = Counter(discrete_actions)
    print(f"\n=== Discrete Actions (from env info) ===")
    for a, cnt in sorted(action_counts.items()):
        name = {0: "HOLD", 1: "BUY", -1: "SELL"}.get(a, f"UNK({a})")
        print(f"  {name}: {cnt} ({100*cnt/len(discrete_actions):.1f}%)")

    # Reward analysis
    print(f"\n=== Reward Distribution ===")
    print(f"  mean={rewards_arr.mean():.8f}, std={rewards_arr.std():.8f}")
    print(f"  min={rewards_arr.min():.8f}, max={rewards_arr.max():.8f}")
    print(f"  sum={rewards_arr.sum():.8f}")
    print(f"  non-zero count: {np.count_nonzero(rewards_arr)}/{len(rewards_arr)}")

    # ROI
    roi = extract_roi_from_env(env)
    portfolio = getattr(env, "portfolio_value", "N/A")
    initial = getattr(env, "initial_portfolio_value", "N/A")
    print(f"\n=== ROI ===")
    print(f"  portfolio_value={portfolio}, initial={initial}")
    print(f"  ROI={roi:.6f}")

    # Stochastic comparison
    obs2, _ = env.reset()
    sto_actions = []
    for _ in range(min(500, n_steps)):
        action_sto, _ = model.predict(obs2, deterministic=False)
        sto_val = float(action_sto[0]) if hasattr(action_sto, '__len__') else float(action_sto)
        sto_actions.append(sto_val)
        obs2, _, term, trunc, _ = env.step(action_sto)
        if term or trunc:
            break
    sto_arr = np.array(sto_actions)
    print(f"\n=== Stochastic Actions (n={len(sto_arr)}) ===")
    print(f"  mean={sto_arr.mean():.6f}, std={sto_arr.std():.6f}")
    print(f"  min={sto_arr.min():.6f}, max={sto_arr.max():.6f}")
    sto_above = np.sum(sto_arr > float(threshold))
    sto_below = np.sum(sto_arr < float(neg_threshold))
    sto_hold = len(sto_arr) - sto_above - sto_below
    print(f"  BUY: {sto_above} ({100*sto_above/len(sto_arr):.1f}%)")
    print(f"  HOLD: {sto_hold} ({100*sto_hold/len(sto_arr):.1f}%)")
    print(f"  SELL: {sto_below} ({100*sto_below/len(sto_arr):.1f}%)")


if __name__ == "__main__":
    model_path = "models/v460/sac_v460_seed42.zip"
    config_path = "configs/v460/experiments/g2_sac_train.yaml"
    diagnose(model_path, config_path)
