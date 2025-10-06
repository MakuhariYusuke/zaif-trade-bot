#!/usr/bin/env python3
"""
Short-Distance A/B Diagnostics for Action Selection.

Tests model behavior on synthetic and real data to verify:
1. Probability time-variance (std > 0)
2. Legal SELL rate ≥ 15%
3. Tiebreaker activation
4. Decode order correctness

Usage:
    python scripts/short_distance_ab_diagnostics.py --model models/ppo_session --steps 300
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.inference.decode import decode_action, InferenceConfig, compute_legal_sell_rate
from ztb.utils.normalization import NormalizationStats
from ztb.utils.calibration import compute_full_calibration_report


def generate_synthetic_uptrend(n_steps: int = 300, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic uptrend data."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2024-01-01", periods=n_steps, freq="1h")
    
    # Monotonic uptrend with noise
    trend = np.linspace(0, 0.5, n_steps)  # 50% gain over period
    noise = np.random.normal(0, 0.01, n_steps)  # 1% noise
    
    prices = base_price * (1 + trend + noise)
    volumes = np.random.uniform(100, 200, n_steps)
    
    return pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "close": prices,
        "volume": volumes,
    }, index=timestamps)


def generate_synthetic_downtrend(n_steps: int = 300, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic downtrend data."""
    np.random.seed(43)
    timestamps = pd.date_range(start="2024-01-01", periods=n_steps, freq="1h")
    
    # Monotonic downtrend with noise
    trend = np.linspace(0, -0.5, n_steps)  # 50% loss over period
    noise = np.random.normal(0, 0.01, n_steps)  # 1% noise
    
    prices = base_price * (1 + trend + noise)
    volumes = np.random.uniform(100, 200, n_steps)
    
    return pd.DataFrame({
        "open": prices * 1.01,
        "high": prices * 1.02,
        "low": prices * 0.99,
        "close": prices,
        "volume": volumes,
    }, index=timestamps)


def load_real_data(data_path: Path, n_steps: int = 300) -> pd.DataFrame:
    """Load real BTC/JPY data (last n_steps)."""
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Take last n_steps
    return df.tail(n_steps)


def run_diagnostic(
    model_path: Path,
    data: pd.DataFrame,
    data_name: str,
    config: InferenceConfig,
    max_steps: int = 300,
) -> Dict[str, Any]:
    """
    Run diagnostic on single dataset.
    
    Returns:
        Dictionary with diagnostic results
    """
    print(f"\n{'='*60}")
    print(f"Running diagnostic: {data_name}")
    print(f"Data shape: {data.shape}, Steps: {min(len(data), max_steps)}")
    print(f"Config: T={config.temperature}, tau={config.tiebreaker_tau}, "
          f"tiebreaker={config.enable_tiebreaker}, deterministic={config.deterministic}")
    print(f"{'='*60}\n")
    
    # Load model
    model = MaskablePPO.load(model_path)
    
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
    top2_actions_list = []
    top2_probs_list = []
    margins_list = []
    tiebreaker_activations = []
    legal_masks_list = []
    
    for step in range(min(len(data) - 1, max_steps)):
        # Get legal actions
        legal_mask = env.get_legal_actions()
        legal_masks_list.append(legal_mask)
        
        # Get logits from policy
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            features = model.policy.extract_features(obs_tensor, model.policy.features_extractor)
            if model.policy.share_features_extractor:
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                latent_pi = model.policy.mlp_extractor.forward_actor(features[0])
            logits = model.policy.action_net(latent_pi).cpu().numpy()[0]
        
        # Decode action with strict order
        action, info = decode_action(logits, legal_mask, config)
        
        # Store diagnostics
        actions_list.append(action)
        probabilities_list.append(info["probabilities"])
        top2_actions_list.append(info["top2_actions"])
        top2_probs_list.append(info["top2_probs"])
        margins_list.append(info["margin"])
        tiebreaker_activations.append(info["tiebreaker_activated"])
        
        # Step environment
        obs, _, done, _ = env.step(action)
        if done:
            break
    
    # Convert to arrays
    actions_array = np.array(actions_list)
    probabilities_array = np.array(probabilities_list)
    margins_array = np.array(margins_list)
    tiebreaker_array = np.array(tiebreaker_activations)
    legal_masks_array = np.array(legal_masks_list)
    
    # Compute statistics
    legal_sell_stats = compute_legal_sell_rate(actions_array, legal_masks_array)
    
    # Probability variance (per action)
    prob_stds = np.std(probabilities_array, axis=0)
    
    # Action distribution
    action_counts = np.bincount(actions_array, minlength=3)
    action_dist = action_counts / len(actions_array)
    
    # Compute calibration diagnostics (Brier score and reliability curves)
    calibration_report = compute_full_calibration_report(
        probabilities_array,
        actions_array,
        n_bins=10,
    )
    
    results = {
        "data_name": data_name,
        "total_steps": len(actions_array),
        "action_distribution": {
            "HOLD": float(action_dist[0]),
            "BUY": float(action_dist[1]),
            "SELL": float(action_dist[2]),
        },
        "legal_sell_stats": legal_sell_stats,
        "probability_variance": {
            "HOLD_std": float(prob_stds[0]),
            "BUY_std": float(prob_stds[1]),
            "SELL_std": float(prob_stds[2]),
            "mean_std": float(np.mean(prob_stds)),
        },
        "margin_stats": {
            "mean": float(np.mean(margins_array)),
            "std": float(np.std(margins_array)),
            "min": float(np.min(margins_array)),
            "max": float(np.max(margins_array)),
        },
        "tiebreaker_stats": {
            "activation_count": int(np.sum(tiebreaker_array)),
            "activation_rate": float(np.mean(tiebreaker_array)),
        },
        "calibration": calibration_report,
        "acceptance_criteria": {
            "prob_std_positive": bool(np.mean(prob_stds) > 0),
            "legal_sell_rate_ok": bool(legal_sell_stats["legal_sell_rate"] >= 0.15),
        },
    }
    
    # Print summary
    print(f"\nResults for {data_name}:")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Action distribution: HOLD={action_dist[0]:.1%}, BUY={action_dist[1]:.1%}, SELL={action_dist[2]:.1%}")
    print(f"  Legal SELL rate: {legal_sell_stats['legal_sell_rate']:.1%} (target: ≥15%)")
    print(f"  Probability std (mean): {np.mean(prob_stds):.4f} (target: >0)")
    print(f"  Margin (mean±std): {np.mean(margins_array):.4f}±{np.std(margins_array):.4f}")
    print(f"  Tiebreaker activations: {np.sum(tiebreaker_array)}/{len(tiebreaker_array)} ({np.mean(tiebreaker_array):.1%})")
    
    # Print calibration diagnostics
    print(f"\n  Calibration Diagnostics:")
    print(f"    Brier score (overall): {calibration_report['brier_score']['overall']:.4f} (lower is better)")
    print(f"    Brier score per action:")
    for action_name, score in calibration_report['brier_score']['per_action'].items():
        if score is not None:
            print(f"      {action_name}: {score:.4f}")
        else:
            print(f"      {action_name}: N/A (no samples)")
    print(f"    Expected Calibration Error (ECE):")
    for action_name, curve in calibration_report['reliability_curves'].items():
        print(f"      {action_name}: {curve['expected_calibration_error']:.4f}")
    
    # Check acceptance
    all_pass = all(results["acceptance_criteria"].values())
    status = "✅ PASS" if all_pass else "❌ FAIL"
    print(f"\n  Acceptance: {status}")
    if not all_pass:
        for criterion, passed in results["acceptance_criteria"].items():
            if not passed:
                print(f"    ❌ {criterion}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Short-distance A/B diagnostics")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps per dataset")
    parser.add_argument("--real-data", type=str, default="btc_jpy_real_dataset.csv", help="Path to real data")
    parser.add_argument("--temperature", type=float, default=0.7, help="Softmax temperature")
    parser.add_argument("--tiebreaker-tau", type=float, default=0.05, help="Tiebreaker margin threshold")
    parser.add_argument("--disable-tiebreaker", action="store_true", help="Disable tiebreaker")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic action selection")
    parser.add_argument("--output", type=str, default="diagnostics_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)
    
    # Find model.zip in directory
    model_file = model_path / "model.zip"
    if not model_file.exists():
        print(f"Error: model.zip not found in {model_path}")
        sys.exit(1)
    
    # Create inference config
    config = InferenceConfig(
        temperature=args.temperature,
        tiebreaker_tau=args.tiebreaker_tau,
        enable_tiebreaker=not args.disable_tiebreaker,
        deterministic=args.deterministic,
    )
    
    # Generate datasets
    print("Generating synthetic datasets...")
    synth_up = generate_synthetic_uptrend(n_steps=args.steps + 100)  # +100 for warmup
    synth_down = generate_synthetic_downtrend(n_steps=args.steps + 100)
    
    # Load real data
    real_data_path = Path(args.real_data)
    if real_data_path.exists():
        print(f"Loading real data from {real_data_path}...")
        real_data = load_real_data(real_data_path, n_steps=args.steps + 100)
    else:
        print(f"Warning: Real data not found at {real_data_path}, skipping...")
        real_data = None
    
    # Run diagnostics
    all_results = []
    
    # Test 1: Synthetic uptrend (deterministic mode)
    print("\n" + "="*60)
    print("TEST 1: Synthetic Uptrend (Deterministic)")
    print("="*60)
    config_det = InferenceConfig(
        temperature=args.temperature,
        tiebreaker_tau=args.tiebreaker_tau,
        enable_tiebreaker=not args.disable_tiebreaker,
        deterministic=True,
    )
    result_up_det = run_diagnostic(model_file, synth_up, "Synthetic_Uptrend_Det", config_det, args.steps)
    all_results.append(result_up_det)
    
    # Test 2: Synthetic downtrend (deterministic mode)
    print("\n" + "="*60)
    print("TEST 2: Synthetic Downtrend (Deterministic)")
    print("="*60)
    result_down_det = run_diagnostic(model_file, synth_down, "Synthetic_Downtrend_Det", config_det, args.steps)
    all_results.append(result_down_det)
    
    # Test 3: Real data (deterministic mode)
    if real_data is not None:
        print("\n" + "="*60)
        print("TEST 3: Real Data (Deterministic)")
        print("="*60)
        result_real_det = run_diagnostic(model_file, real_data, "Real_Data_Det", config_det, args.steps)
        all_results.append(result_real_det)
    
    # Test 4: Synthetic uptrend (stochastic mode)
    print("\n" + "="*60)
    print("TEST 4: Synthetic Uptrend (Stochastic)")
    print("="*60)
    config_stoch = InferenceConfig(
        temperature=args.temperature,
        tiebreaker_tau=args.tiebreaker_tau,
        enable_tiebreaker=not args.disable_tiebreaker,
        deterministic=False,
    )
    result_up_stoch = run_diagnostic(model_file, synth_up, "Synthetic_Uptrend_Stoch", config_stoch, args.steps)
    all_results.append(result_up_stoch)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "model_path": str(model_path),
                "temperature": args.temperature,
                "tiebreaker_tau": args.tiebreaker_tau,
                "enable_tiebreaker": not args.disable_tiebreaker,
                "steps": args.steps,
            },
            "results": all_results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    all_pass = all(all(r["acceptance_criteria"].values()) for r in all_results)
    
    for result in all_results:
        status = "✅ PASS" if all(result["acceptance_criteria"].values()) else "❌ FAIL"
        print(f"  {result['data_name']}: {status}")
        print(f"    - Legal SELL rate: {result['legal_sell_stats']['legal_sell_rate']:.1%}")
        print(f"    - Prob std (mean): {result['probability_variance']['mean_std']:.4f}")
    
    print(f"\n{'='*60}")
    if all_pass:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
