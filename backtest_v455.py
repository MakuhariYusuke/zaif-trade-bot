import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.execution.pseudo_hft import PseudoHFTExecutionModel
from ztb.trading.signal.entry_system import IntegratedEntrySystem
from ztb.trading.types import MarketState


def run_backtest_v455():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps to run"
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to model")
    parser.add_argument("--warmup-steps", type=int, default=10000, help="Steps to run without Gate blocking to gather stats")
    args = parser.parse_args()

    # 1. Load Config
    # We'll use v451 config as base but add v455 specific settings
    config_path = os.path.join(
        project_root, "config", "v451", "sac_v451_optimized.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    # v455 Configuration
    v455_config = {
        "ewma_tau": 100.0,
        "n_min": 30.0,
        "fee_rate": 0.001,
        "c_spread": 0.3,
        "c_vol": 0.2,
        "c_imp": 0.5,
        "gamma": 0.5,
        "min_volume": 0.01,
        "latency_sec": 1.0,
        "order_size_btc": 0.01,
    }

    # Setup environment config
    env_config = config["training"]["environment"]["config"]
    env_config["adaptive_threshold_mode"] = True

    # Load data
    data_path = os.path.join(project_root, "data", "btc_jpy_1m_v451.csv")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Create environment
    env = HeavyTradingEnv(df, env_config)

    # Load model
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(
            project_root, "models", "sac_v451_phase7_regime_aware.zip"
        )
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = os.path.join(
                project_root, "checkpoints", "v451", "phase7", "best_model.zip"
            )

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    model = SAC.load(model_path, env=env)

    # Initialize v455 Components
    entry_system = IntegratedEntrySystem(v455_config)
    execution_model = PseudoHFTExecutionModel(v455_config)

    # Shadow Portfolio State
    shadow_jpy = 1000000.0
    shadow_btc = 0.0
    shadow_history = []

    # Tracking for Calibration Update
    last_entry_market_price = 0.0  # Pre-slippage price
    last_entry_action = 0.0
    last_entry_regime = "unknown"

    obs, _ = env.reset()
    done = False

    print("Starting v455 Backtest...")

    step_count = 0
    while not done:
        if args.steps and step_count >= args.steps:
            break
        step_count += 1

        current_step = env.current_step

        # 1. Get RL Action
        action_rl, _ = model.predict(obs, deterministic=True)

        # Extract scalar action for logic
        if isinstance(action_rl, np.ndarray):
            raw_action = float(action_rl[0])
        else:
            raw_action = float(action_rl)

        # 2. Construct MarketState
        # We need to access current row from df.
        # env.current_step corresponds to the index in df (roughly, depending on window size)
        # HeavyEnv usually aligns current_step with df index.
        try:
            row = df.iloc[current_step]
            market_data: MarketState = {
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "atr": float(row.get("atr", row.get("ATR", 0.0))),
                "volume": float(row["volume"]),
                "timestamp": str(row.name),
            }
        except Exception:
            # Fallback if index out of bounds or columns missing
            market_data = {
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "atr": 0.0,
                "volume": 0.0,
                "timestamp": None,
            }

        # Get Regime
        # We can try to get it from env if exposed, or use 'unknown'
        regime = "unknown"
        if hasattr(env, "_get_current_market_regime"):
            r = env._get_current_market_regime()
            regime = r.value if hasattr(r, "value") else str(r)

        # 3. Gate Check (Entry)
        # Determine if RL wants to buy (assuming discrete mapping or threshold)
        # HeavyEnv maps continuous > threshold to BUY.
        # We need to know the threshold.
        threshold = env.action_threshold
        negative_threshold = env.negative_action_threshold

        # Check if it's an entry signal
        is_buy_signal = raw_action > threshold
        is_sell_signal = raw_action < negative_threshold  # Short entry

        action_to_env = action_rl

        # If we are flat and get a signal, check Gate
        if shadow_btc == 0.0:
            if is_buy_signal:
                gate_res = entry_system.process_signal(
                    raw_action,
                    market_data,
                    regime,
                    order_size=v455_config["order_size_btc"],
                )
                # Gate Logic with Warm-up
                # If in warm-up, we ignore 'should_enter' being False, effectively forcing entry
                # to gather calibration data.
                should_enter = gate_res["should_enter"]
                if step_count < args.warmup_steps:
                    should_enter = True
                
                if not should_enter:
                    # Gate blocked entry
                    action_to_env = np.array([0.0], dtype=np.float32)
                else:
                    # Gate allowed entry (or forced by warm-up)
                    last_entry_action = raw_action
                    last_entry_regime = regime
            elif is_sell_signal:
                # Short entry logic
                gate_res = entry_system.process_signal(
                    raw_action,
                    market_data,
                    regime,
                    order_size=v455_config["order_size_btc"],
                )
                should_enter = gate_res["should_enter"]
                if step_count < args.warmup_steps:
                    should_enter = True

                if not should_enter:
                    # Gate blocked entry
                    action_to_env = np.array([0.0], dtype=np.float32)
                else:
                    # Gate allowed entry (or forced by warm-up)
                    last_entry_action = raw_action
                    last_entry_regime = regime

        # 4. Step Env
        # We pass the potentially modified action
        # Note: If we forced entry during warm-up, we pass the original action to Env.
        # If Gate blocked, we pass 0.0.
        obs, reward, terminated, truncated, info = env.step(action_to_env)
        done = terminated or truncated

        # 5. Shadow Execution
        # Note on Execution Timing:
        # We use 'market_data' fetched BEFORE env.step().
        # This assumes we execute at the Close of the bar that generated the signal.
        # This is consistent with PseudoHFT which simulates immediate execution (plus latency).
        # If Env executes at Next Open, there might be a slight mismatch, but for HFT simulation
        # we control the execution price via PseudoHFT anyway.
            discrete_act = 1  # Buy
        elif act_val < negative_threshold:
            discrete_act = 2  # Sell

        # Logic for Shadow Portfolio
        # State: shadow_btc (positive = long, negative = short, 0 = flat)

        # LONG ENTRY
        if discrete_act == 1 and shadow_btc == 0.0:
            requested_size = v455_config["order_size_btc"]

            exec_res = execution_model.simulate_execution(
                "buy",
                market_data["close"],
                requested_size,
                current_atr=market_data["atr"],
                current_volume=market_data["volume"],
                market_regime=regime,
                market_data=market_data,
            )

            cost = exec_res.executed_price * exec_res.executed_size
            shadow_jpy -= cost
            shadow_btc += exec_res.executed_size
            last_entry_market_price = market_data["close"]  # Store pre-slippage price

            # Fee
            fee = cost * v455_config["fee_rate"]
            shadow_jpy -= fee

        # SHORT ENTRY
        elif discrete_act == 2 and shadow_btc == 0.0:
            requested_size = v455_config["order_size_btc"]

            exec_res = execution_model.simulate_execution(
                "sell",
                market_data["close"],
                requested_size,
                current_atr=market_data["atr"],
                current_volume=market_data["volume"],
                market_regime=regime,
                market_data=market_data,
            )

            revenue = exec_res.executed_price * exec_res.executed_size
            shadow_jpy += revenue
            shadow_btc -= exec_res.executed_size  # Negative for short
            last_entry_market_price = market_data["close"]  # Store pre-slippage price

            # Fee
            fee = revenue * v455_config["fee_rate"]
            shadow_jpy -= fee

        # LONG EXIT (SELL)
        elif discrete_act == 2 and shadow_btc > 0.0:
            requested_size = abs(shadow_btc)  # Sell all

            exec_res = execution_model.simulate_execution(
                "sell",
                market_data["close"],
                requested_size,
                current_atr=market_data["atr"],
                current_volume=market_data["volume"],
                market_regime=regime,
                market_data=market_data,
            )

            revenue = exec_res.executed_price * exec_res.executed_size
            shadow_jpy += revenue
            shadow_btc = 0.0

            # Fee
            fee = revenue * v455_config["fee_rate"]
            shadow_jpy -= fee

            # Update Calibration Map
            # Calculate Gross PnL per unit (Pre-Slippage)
            # Gross PnL = ExitMarketPrice - EntryMarketPrice
            gross_pnl_per_unit = market_data["close"] - last_entry_market_price

            entry_system.update_outcome(
                last_entry_regime, last_entry_action, gross_pnl_per_unit, current_step
            )

        # SHORT EXIT (COVER/BUY)
        elif discrete_act == 1 and shadow_btc < 0.0:
            requested_size = abs(shadow_btc)  # Buy back all

            exec_res = execution_model.simulate_execution(
                "buy",
                market_data["close"],
                requested_size,
                current_atr=market_data["atr"],
                current_volume=market_data["volume"],
                market_regime=regime,
                market_data=market_data,
            )

            cost = exec_res.executed_price * exec_res.executed_size
            shadow_jpy -= cost
            shadow_btc = 0.0

            # Fee
            fee = cost * v455_config["fee_rate"]
            shadow_jpy -= fee

            # Update Calibration Map
            # Calculate Gross PnL per unit (Pre-Slippage) for Short
            # Gross PnL = EntryMarketPrice - ExitMarketPrice
            gross_pnl_per_unit = last_entry_market_price - market_data["close"]

            entry_system.update_outcome(
                last_entry_regime, last_entry_action, gross_pnl_per_unit, current_step
            )

        # Track Portfolio Value
        current_price = market_data["close"]
        total_val = shadow_jpy + shadow_btc * current_price
        shadow_history.append(total_val)

        if current_step % 1000 == 0:
            print(f"Step {current_step}, Shadow PV: {total_val:.0f}")

    print(f"Backtest Finished. Final Shadow PV: {shadow_history[-1]:.0f}")

    # Save Results
    results_dir = os.path.join(project_root, "backtest_results", "v455")
    os.makedirs(results_dir, exist_ok=True)

    pd.DataFrame({"portfolio_value": shadow_history}).to_csv(
        os.path.join(results_dir, "shadow_results.csv")
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    run_backtest_v455()
