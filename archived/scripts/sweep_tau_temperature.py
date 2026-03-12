#!/usr/bin/env python3
"""
τ × T Parameter Sweep for Action Decoding.

Grid search over tiebreaker_tau and temperature to find optimal combination.

Usage:
    python scripts/sweep_tau_temperature.py \
        --model models/ppo_session \
        --steps 300 \
        --data-up data/synth_up.csv \
        --data-down data/synth_down.csv \
        --data-real data/btc_jpy_eval.csv \
        --out artifacts/sweep/tau_temp_results.csv
"""

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.inference.decode import InferenceConfig, compute_legal_sell_rate, decode_action
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.data_utils import load_csv_data_optimized


def generate_synthetic_uptrend(n_steps: int = 300) -> pd.DataFrame:
    """Generate synthetic uptrend data."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2024-01-01", periods=n_steps, freq="1h")
    trend = np.linspace(0, 0.5, n_steps)
    noise = np.random.normal(0, 0.01, n_steps)
    prices = 100.0 * (1 + trend + noise)

    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.uniform(100, 200, n_steps),
        },
        index=timestamps,
    )


def generate_synthetic_downtrend(n_steps: int = 300) -> pd.DataFrame:
    """Generate synthetic downtrend data."""
    np.random.seed(43)
    timestamps = pd.date_range(start="2024-01-01", periods=n_steps, freq="1h")
    trend = np.linspace(0, -0.5, n_steps)
    noise = np.random.normal(0, 0.01, n_steps)
    prices = 100.0 * (1 + trend + noise)

    return pd.DataFrame(
        {
            "open": prices * 1.01,
            "high": prices * 1.02,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(100, 200, n_steps),
        },
        index=timestamps,
    )


def run_sweep_experiment(
    model: MaskablePPO,
    data: pd.DataFrame,
    data_name: str,
    tau: float,
    temperature: float,
    max_steps: int = 300,
) -> Dict[str, Any]:
    """
    Run single sweep experiment with given tau and temperature.

    Returns:
        Dictionary with metrics
    """
    config = InferenceConfig(
        temperature=temperature,
        tiebreaker_tau=tau,
        enable_tiebreaker=True,
        enable_advantage_tiebreaker=True,  # ON for production-like eval
        enable_cost_gate=True,  # ON for production-like eval
        deterministic=True,
    )

    # Create environment
    env = HeavyTradingEnv(
        df=data,
        config={
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.0,
            "initial_portfolio_value": 10000.0,
        },
    )

    # Run episode
    obs = env.reset()
    actions_list = []
    probabilities_list = []
    margins_list = []
    tiebreaker_activated = []
    tiebreaker_reasons = []
    cost_gate_triggered = []
    legal_masks_list = []
    rewards_list = []

    for _ in range(min(len(data) - 1, max_steps)):
        # Get legal actions
        legal_mask = env.get_legal_actions()
        legal_masks_list.append(legal_mask)

        # Get logits from policy
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            features = model.policy.extract_features(
                obs_tensor, model.policy.features_extractor
            )
            if model.policy.share_features_extractor:
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                latent_pi = model.policy.mlp_extractor.forward_actor(features[0])
            logits = model.policy.action_net(latent_pi).cpu().numpy()[0]

        # Decode action
        action, info = decode_action(logits, legal_mask, config)

        # Store diagnostics
        actions_list.append(action)
        probabilities_list.append(info["probabilities"])
        margins_list.append(info["margin"])
        tiebreaker_activated.append(info["tiebreaker_activated"])
        tiebreaker_reasons.append(info.get("tiebreaker_reason"))
        cost_gate_triggered.append(info.get("cost_gate_triggered", False))

        # Step environment
        obs, reward, done, _ = env.step(action)
        rewards_list.append(reward)

        if done:
            break

    # Convert to arrays
    actions_array = np.array(actions_list)
    probabilities_array = np.array(probabilities_list)
    margins_array = np.array(margins_list)
    legal_masks_array = np.array(legal_masks_list)
    rewards_array = np.array(rewards_list)

    # Compute metrics
    legal_sell_stats = compute_legal_sell_rate(actions_array, legal_masks_array)

    prob_stds = np.std(probabilities_array, axis=0)

    # Count trades (position changes)
    position_changes = np.sum(actions_array != 0)

    # Sharpe proxy (short-distance approximation)
    if len(rewards_array) > 1:
        reward_mean = np.mean(rewards_array)
        reward_std = np.std(rewards_array)
        sharpe_proxy = reward_mean / (reward_std + 1e-6)
    else:
        sharpe_proxy = 0.0

    # Tiebreaker breakdown
    tiebreaker_counts = {
        "advantage_sign": sum(1 for r in tiebreaker_reasons if r == "advantage_sign"),
        "prob_margin": sum(1 for r in tiebreaker_reasons if r == "prob_margin"),
    }

    # Cost gate stats
    cost_gate_count = sum(cost_gate_triggered)

    return {
        "data_name": data_name,
        "tau": tau,
        "temperature": temperature,
        "total_steps": len(actions_array),
        "legal_sell_rate": legal_sell_stats["legal_sell_rate"],
        "trades": int(position_changes),
        "sharpe_proxy": float(sharpe_proxy),
        "margin_mean": float(np.mean(margins_array)),
        "margin_std": float(np.std(margins_array)),
        "prob_std_mean": float(np.mean(prob_stds)),
        "tiebreaker_advantage_count": tiebreaker_counts["advantage_sign"],
        "tiebreaker_probmargin_count": tiebreaker_counts["prob_margin"],
        "cost_gate_count": cost_gate_count,
    }


def main():
    parser = argparse.ArgumentParser(description="τ × T parameter sweep")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Max steps per experiment (default: 300)",
    )
    parser.add_argument(
        "--data-up",
        type=Path,
        help="Path to uptrend data CSV (or use synthetic)",
    )
    parser.add_argument(
        "--data-down",
        type=Path,
        help="Path to downtrend data CSV (or use synthetic)",
    )
    parser.add_argument(
        "--data-real",
        type=Path,
        help="Path to real data CSV (optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for results CSV",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("τ × T Parameter Sweep")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps}")
    print(f"Output: {args.out}")
    print()

    # Load model
    print("Loading model...")
    model = MaskablePPO.load(args.model)
    print("✅ Model loaded")
    print()

    # Prepare datasets
    datasets = []

    if args.data_up:
        df_up = load_csv_data_optimized(args.data_up, index_col=0).tail(args.steps)
        datasets.append(("uptrend", df_up))
    else:
        datasets.append(("synth_up", generate_synthetic_uptrend(args.steps)))

    if args.data_down:
        df_down = load_csv_data_optimized(args.data_down, index_col=0).tail(args.steps)
        datasets.append(("downtrend", df_down))
    else:
        datasets.append(("synth_down", generate_synthetic_downtrend(args.steps)))

    if args.data_real:
        df_real = load_csv_data_optimized(args.data_real, index_col=0).tail(args.steps)
        datasets.append(("real", df_real))

    print(f"Datasets: {len(datasets)}")
    for name, df in datasets:
        print(f"  - {name}: {len(df)} steps")
    print()

    # Grid search
    tau_values = [0.03, 0.05, 0.07]
    temp_values = [0.6, 0.7, 0.9]

    total_experiments = len(tau_values) * len(temp_values) * len(datasets)
    print(
        f"Total experiments: {total_experiments} ({len(tau_values)} tau × {len(temp_values)} temp × {len(datasets)} datasets)"
    )
    print()

    # Run sweep
    results = []

    for i, (tau, temp) in enumerate(product(tau_values, temp_values), 1):
        print(f"[{i}/{len(tau_values) * len(temp_values)}] tau={tau}, T={temp}")

        for data_name, data in datasets:
            print(f"  Running {data_name}...", end=" ", flush=True)

            result = run_sweep_experiment(
                model=model,
                data=data,
                data_name=data_name,
                tau=tau,
                temperature=temp,
                max_steps=args.steps,
            )

            results.append(result)

            print(
                f"✓ (legal_sell={result['legal_sell_rate']:.1%}, sharpe={result['sharpe_proxy']:.3f})"
            )

        print()

    # Save results
    print("Saving results...")
    df_results = pd.DataFrame(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.out, index=False)

    print(f"✅ Results saved to: {args.out}")
    print()

    # Print summary
    print("Summary Statistics:")
    print(
        f"  Legal SELL rate: {df_results['legal_sell_rate'].mean():.1%} ± {df_results['legal_sell_rate'].std():.1%}"
    )
    print(
        f"  Sharpe proxy: {df_results['sharpe_proxy'].mean():.3f} ± {df_results['sharpe_proxy'].std():.3f}"
    )
    print(
        f"  Trades: {df_results['trades'].mean():.1f} ± {df_results['trades'].std():.1f}"
    )
    print()

    # Find best configurations
    print("Top 3 Configurations (by Sharpe proxy):")
    top3 = df_results.nlargest(3, "sharpe_proxy")

    for idx, row in top3.iterrows():
        print(
            f"  {idx+1}. tau={row['tau']}, T={row['temperature']}, {row['data_name']}"
        )
        print(
            f"     Sharpe={row['sharpe_proxy']:.3f}, legal_sell={row['legal_sell_rate']:.1%}, trades={row['trades']}"
        )

    print()
    print("✅ Sweep complete. Review results CSV for Pareto analysis.")


if __name__ == "__main__":
    main()
