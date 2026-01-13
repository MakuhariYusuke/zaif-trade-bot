import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None


# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.results_utils import save_backtest_results


class GateLogger:
    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.headers = [
            "step",
            "timestamp",
            "action",
            "threshold",
            "env_threshold",
            "regime",
            "gate_status",
            "gate_should_enter",
            "gate_ev",
            "gate_cost",
            "p_win_lcb",
            "p_win_mean",
            "p_win_ucb",
            "avg_win",
            "avg_loss",
            "n_eff",
            "atr",
            "atr_env",
            "block_reason",
            "gate_bin",
        ]
        self.buffer = []

        # Write header
        with open(self.filepath, "w") as f:
            f.write(",".join(self.headers) + "\n")

    def log(self, data) -> None:
        # data is a dict matching headers
        row = [str(data.get(h, "")) for h in self.headers]
        self.buffer.append(",".join(row))

        if len(self.buffer) >= 100:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        with open(self.filepath, "a") as f:
            f.write("\n".join(self.buffer) + "\n")
        self.buffer = []

    @staticmethod
    def get_bin(action: float) -> str:
        if action > 0.6:
            return "Strong_Buy"
        if action > 0.2:
            return "Buy"
        if action > -0.2:
            return "Neutral"
        if action > -0.6:
            return "Sell"
        return "Strong_Sell"


class BacktestReporter:
    """
    Simplified reporter for backtest statistics.
    Mimics ztb.training.unified_trainer.reporting.TrainingReporter structure.
    """

    def __init__(self) -> None:
        self.start_time = time.time()
        self.stats = {
            "total_steps": 0,
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "gate_blocked_count": 0,
            "gate_allowed_count": 0,
            "warmup_forced_count": 0,
            "action_distribution": {},
            "regime_distribution": {},
            "system_info": {},
            "performance": {},
        }
        self.portfolio_history = []
        self.trade_history = []

    def update_step(self, step, portfolio_value, action, regime, gate_status) -> None:
        self.stats["total_steps"] += 1
        self.portfolio_history.append(portfolio_value)

        # Action Dist
        act_key = "hold"
        if action > 0:
            act_key = "buy"
        elif action < 0:
            act_key = "sell"
        self.stats["action_distribution"][act_key] = (
            self.stats["action_distribution"].get(act_key, 0) + 1
        )

        # Regime Dist
        self.stats["regime_distribution"][regime] = (
            self.stats["regime_distribution"].get(regime, 0) + 1
        )

        # Gate Stats
        if gate_status == "blocked":
            self.stats["gate_blocked_count"] += 1
        elif gate_status == "allowed":
            self.stats["gate_allowed_count"] += 1
        elif gate_status == "forced":
            self.stats["warmup_forced_count"] += 1

    def record_trade(self, trade_type, pnl, entry_price, exit_price, size) -> None:
        self.stats["total_trades"] += 1
        if trade_type == "long":
            self.stats["long_trades"] += 1
        else:
            self.stats["short_trades"] += 1

        if pnl > 0:
            self.stats["winning_trades"] += 1
            self.stats["gross_profit"] += pnl
        else:
            self.stats["losing_trades"] += 1
            self.stats["gross_loss"] += abs(pnl)

        self.stats["net_pnl"] += pnl
        self.trade_history.append(
            {
                "type": trade_type,
                "pnl": pnl,
                "entry": entry_price,
                "exit": exit_price,
                "size": size,
            }
        )

    def finalize_stats(self) -> None:
        # Calculate Drawdown
        peak = -np.inf
        max_dd = 0.0
        max_dd_pct = 0.0

        for val in self.portfolio_history:
            if val > peak:
                peak = val
            dd = peak - val
            dd_pct = dd / peak if peak > 0 else 0.0

            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        self.stats["max_drawdown"] = max_dd
        self.stats["max_drawdown_percent"] = max_dd_pct

        # Sharpe (Simplified, assuming 1m steps)
        if len(self.portfolio_history) > 1:
            returns = pd.Series(self.portfolio_history).pct_change().dropna()
            if returns.std() > 0:
                # Annualize (assuming 1m bars -> 525600 mins/year)
                self.stats["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(
                    525600
                )
            else:
                self.stats["sharpe_ratio"] = 0.0

        # Profit Factor
        if self.stats["gross_loss"] > 0:
            self.stats["profit_factor"] = (
                self.stats["gross_profit"] / self.stats["gross_loss"]
            )
        else:
            self.stats["profit_factor"] = (
                float("inf") if self.stats["gross_profit"] > 0 else 0.0
            )

        # Avg Win/Loss
        if self.stats["winning_trades"] > 0:
            self.stats["avg_win"] = (
                self.stats["gross_profit"] / self.stats["winning_trades"]
            )
        if self.stats["losing_trades"] > 0:
            self.stats["avg_loss"] = (
                self.stats["gross_loss"] / self.stats["losing_trades"]
            )

        # Performance Metrics
        end_time = time.time()
        duration = end_time - self.start_time
        self.stats["performance"]["duration_seconds"] = duration
        if duration > 0:
            self.stats["performance"]["steps_per_second"] = (
                self.stats["total_steps"] / duration
            )

        # System Info
        self.stats["system_info"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }
        if psutil:
            try:
                self.stats["system_info"][
                    "memory_total"
                ] = psutil.virtual_memory().total
                self.stats["system_info"][
                    "memory_available"
                ] = psutil.virtual_memory().available
            except:
                pass

        # Action Diversity
        action_counts = list(self.stats["action_distribution"].values())
        if action_counts:
            total_actions = sum(action_counts)
            if total_actions > 0:
                ratios = [c / total_actions for c in action_counts]
                ideal_ratio = 1.0 / len(action_counts)
                diversity = 1.0 - sum(abs(r - ideal_ratio) for r in ratios) / 2.0
                self.stats["performance"]["action_diversity"] = diversity

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("BACKTEST REPORT SUMMARY")
        print("=" * 60)

        print(f"Total Steps: {self.stats['total_steps']}")
        print(f"Total Trades: {self.stats['total_trades']}")
        print(f"  Longs: {self.stats['long_trades']}")
        print(f"  Shorts: {self.stats['short_trades']}")
        print(
            f"Win Rate: {self.stats['winning_trades'] / max(1, self.stats['total_trades']):.2%}"
        )
        print(f"Profit Factor: {self.stats['profit_factor']:.2f}")
        print(f"Avg Win: {self.stats['avg_win']:.2f}")
        print(f"Avg Loss: {self.stats['avg_loss']:.2f}")
        print(f"Net PnL: {self.stats['net_pnl']:.2f}")
        print(
            f"Max Drawdown: {self.stats['max_drawdown']:.2f} ({self.stats['max_drawdown_percent']:.2%})"
        )
        print(f"Sharpe Ratio: {self.stats['sharpe_ratio']:.4f}")

        print("\nGate Statistics:")
        print(f"  Allowed: {self.stats['gate_allowed_count']}")
        print(f"  Blocked: {self.stats['gate_blocked_count']}")
        print(f"  Forced (Warmup): {self.stats['warmup_forced_count']}")

        print("\nAction Distribution:")
        for k, v in self.stats["action_distribution"].items():
            print(f"  {k}: {v}")

        print("\nPerformance Metrics:")
        for k, v in self.stats["performance"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        print("=" * 60)


def run_backtest_v455() -> None:
    print("DEBUG: Running MODIFIED v455 backtest script - VERSION 2")
    import argparse

    from stable_baselines3 import SAC
    from ztb.trading.environment.heavy_env import HeavyTradingEnv
    from ztb.trading.execution.pseudo_hft import PseudoHFTExecutionModel
    from ztb.trading.signal.entry_system import IntegratedEntrySystem
    from ztb.trading.types import MarketState
    from ztb.utils.config_utils import load_config_unified

    # Helper to load model
    def load_model(path, algorithm="SAC"):
        return SAC.load(path)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of steps to run"
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to model")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Steps to run without Gate blocking to gather stats",
    )
    args = parser.parse_args()

    # Initialize Reporter
    reporter = BacktestReporter()
    gate_logger = GateLogger(os.path.join(project_root, "backtest_gate_log.csv"))

    def log_gate_decision(
        step, ts, action, thresh, env_thresh, regime, status, res, atr, atr_env
    ):
        stats = res.get("stats", {})
        # Determine block reason
        reason = "allowed"
        prob_mode = v455_config.get("probability_mode", "lcb")

        if status == "blocked":
            if not math.isfinite(res.get("cost", 0)):
                reason = "cost_inf"
            elif res.get("cost", 0) > res.get("ev", 0):
                reason = "cost_gt_ev"
            elif stats.get("n_eff", 0) < v455_config["n_min"]:
                reason = "n_eff_low"
            else:
                # Check probability based on mode
                p_val = stats.get(f"p_win_{prob_mode}", 0.0)
                if p_val <= 0.5:
                    reason = f"{prob_mode}_low"
                else:
                    reason = "unknown"
        elif status == "forced":
            reason = "warmup_forced"

        gate_logger.log(
            {
                "step": step,
                "timestamp": ts,
                "action": action,
                "threshold": thresh,
                "env_threshold": env_thresh,
                "regime": regime,
                "gate_status": status,
                "gate_should_enter": res.get("should_enter", False),
                "gate_ev": res.get("ev", 0),
                "gate_cost": res.get("cost", 0),
                "p_win_lcb": stats.get("p_win_lcb", 0),
                "p_win_mean": stats.get("p_win_mean", 0),
                "p_win_ucb": stats.get("p_win_ucb", 0),
                "avg_win": stats.get("avg_win", 0),
                "avg_loss": stats.get("avg_loss", 0),
                "n_eff": stats.get("n_eff", 0),
                "atr": atr,
                "atr_env": atr_env,
                "block_reason": reason,
                "gate_bin": GateLogger.get_bin(res.get("normalized_action", action)),
            }
        )

    # 1. Load Config using unified utilities
    # Load base config (v451)
    config_path = os.path.join(
        project_root, "config", "v451", "sac_v451_optimized.json"
    )
    config = load_config_unified(config_path, required_keys=["training"])

    # Load v455 specific config
    v455_config_path = os.path.join(project_root, "config", "v455", "gate_config.json")
    v455_config = load_config_unified(v455_config_path)

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

    # Load model using unified utility
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
    model = load_model(model_path, algorithm="SAC")

    # Set environment for the model
    model.set_env(env)

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
    last_entry_threshold = 0.0

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
            # Use env.df which has preprocessed features (like ATR)
            row = env.df.iloc[current_step]

            # Resolve ATR
            atr_val = row.get("atr", row.get("ATR", row.get("ATR_simplified", 0.0)))

            # ATR Fallback
            if not math.isfinite(atr_val) or atr_val <= 1e-9:
                # Try High-Low
                hl = float(row["high"]) - float(row["low"])
                if hl > 0:
                    atr_val = hl
                else:
                    # Last resort: 0.1% of Close
                    atr_val = float(row["close"]) * 0.001

            # Volume Fallback
            vol_val = float(row["volume"])
            if not math.isfinite(vol_val) or vol_val <= 1e-9:
                vol_val = 1.0  # Default small volume

            market_data: MarketState = {
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "atr": float(atr_val),
                "volume": float(vol_val),
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

        # Use dynamic threshold if available (same logic as HeavyTradingEnv.step)
        if hasattr(env, "threshold_manager"):
            threshold = env.threshold_manager.get_threshold(
                volatility=market_data["atr"],
                current_price=market_data["close"],
                regime=regime,
                base_value=env.action_threshold,
                raw_action_value=raw_action,
            )
            negative_threshold = env.threshold_manager.get_threshold(
                volatility=market_data["atr"],
                current_price=market_data["close"],
                regime=regime,
                base_value=env.negative_action_threshold,
                raw_action_value=raw_action,
            )
        else:
            threshold = env.action_threshold
            negative_threshold = env.negative_action_threshold

        # Store strict thresholds for execution logic
        strict_threshold = threshold
        strict_negative_threshold = negative_threshold

        # Relax thresholds during warm-up to encourage exploration/data collection for CalibrationGate
        if step_count < args.warmup_steps:
            threshold *= 0.5
            negative_threshold *= 0.5

        # Check if it's an entry signal
        is_buy_signal = raw_action > threshold
        is_sell_signal = raw_action < negative_threshold  # Short entry

        if step_count % 1000 == 0:
            print(
                f"Step {step_count}: Action={raw_action:.3f}, Thresh={threshold:.3f}, NegThresh={negative_threshold:.3f}"
            )

        action_to_env = action_rl

        # If we are flat and get a signal, check Gate
        gate_status = "none"
        if shadow_btc == 0.0:
            if is_buy_signal:
                gate_res = entry_system.process_signal(
                    raw_action,
                    market_data,
                    regime,
                    threshold=threshold,
                    order_size=v455_config["order_size_btc"],
                )
                # Gate Logic with Warm-up
                should_enter = gate_res["should_enter"]
                cost = gate_res.get("cost", 0.0)

                if step_count < args.warmup_steps:
                    # Only force entry if cost is finite (valid market data)
                    if math.isfinite(cost):
                        should_enter = True
                        gate_status = "forced"

                if not should_enter:
                    # Gate blocked entry
                    gate_status = "blocked"
                    if step_count % 1000 == 0:
                        print(
                            f"DEBUG: Gate Blocked Buy. Step={step_count}, Reason={gate_res.get('reason', 'unknown')}"
                        )
                        if not math.isfinite(cost):
                            print(f"DEBUG: Cost Inf! MarketData={market_data}")
                        if step_count < args.warmup_steps:
                            print(
                                f"DEBUG: Warmup Blocked! Cost={cost}, MarketData={market_data}"
                            )
                    action_to_env = np.array([0.0], dtype=np.float32)
                else:
                    # Gate allowed entry (or forced by warm-up)
                    if gate_status != "forced":
                        gate_status = "allowed"

                    # Ensure Env executes this trade even if raw_action < strict_threshold
                    if raw_action < strict_threshold:
                        # Boost action to force Env execution
                        action_to_env = np.array(
                            [strict_threshold + 1e-4], dtype=np.float32
                        )

                    last_entry_action = raw_action
                    last_entry_regime = regime
                    last_entry_threshold = threshold
                    print(
                        f"DEBUG: Gate Allowed Buy. Step={step_count}, Action={raw_action:.3f}, Cost={cost:.1f}"
                    )

                # LOGGING
                log_gate_decision(
                    step_count,
                    market_data["timestamp"],
                    raw_action,
                    threshold,
                    strict_threshold,
                    regime,
                    gate_status,
                    gate_res,
                    market_data["atr"],
                    market_data["atr"],
                )

            elif is_sell_signal:
                # Short entry logic
                gate_res = entry_system.process_signal(
                    raw_action,
                    market_data,
                    regime,
                    threshold=negative_threshold,
                    order_size=v455_config["order_size_btc"],
                )
                should_enter = gate_res["should_enter"]
                cost = gate_res.get("cost", 0.0)

                if step_count < args.warmup_steps:
                    # Only force entry if cost is finite
                    if math.isfinite(cost):
                        should_enter = True
                        gate_status = "forced"

                if not should_enter:
                    # Gate blocked entry
                    gate_status = "blocked"
                    if step_count % 1000 == 0:
                        print(
                            f"DEBUG: Gate Blocked Sell. Step={step_count}, Reason={gate_res.get('reason', 'unknown')}"
                        )
                        if not math.isfinite(cost):
                            print(f"DEBUG: Cost Inf! MarketData={market_data}")
                    action_to_env = np.array([0.0], dtype=np.float32)
                else:
                    # Gate allowed entry (or forced by warm-up)
                    if gate_status != "forced":
                        gate_status = "allowed"

                    # Ensure Env executes this trade even if raw_action > strict_negative_threshold
                    if raw_action > strict_negative_threshold:
                        action_to_env = np.array(
                            [strict_negative_threshold - 1e-4], dtype=np.float32
                        )

                    last_entry_action = raw_action
                    last_entry_regime = regime
                    last_entry_threshold = negative_threshold
                    print(
                        f"DEBUG: Gate Allowed Sell. Step={step_count}, Action={raw_action:.3f}, Cost={cost:.1f}"
                    )

                # LOGGING
                log_gate_decision(
                    step_count,
                    market_data["timestamp"],
                    raw_action,
                    negative_threshold,
                    strict_negative_threshold,
                    regime,
                    gate_status,
                    gate_res,
                    market_data["atr"],
                    market_data["atr"],
                )

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

        # Execute BUY / SELL / COVER / CLOSE
        if isinstance(action_to_env, np.ndarray):
            act_val = float(action_to_env[0])
        else:
            act_val = float(action_to_env)

        discrete_act = 0
        if act_val > threshold:
            discrete_act = 1  # Buy
        elif act_val < negative_threshold:
            discrete_act = 2  # Sell

        # DEBUG
        if discrete_act != 0:
            print(
                f"DEBUG: Shadow Logic sees Action {discrete_act}. ShadowBTC={shadow_btc}"
            )

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

        # LONG EXIT
        elif discrete_act == 2 and shadow_btc > 0.0:
            requested_size = abs(shadow_btc)

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

            # Calculate PnL for Reporter
            # Net PnL is captured in shadow_jpy change, but for stats we want per-trade PnL
            # We can approximate or track it.
            # Simple way: (ExitPrice - EntryPrice) * Size - Fees
            # But we have fees on both sides.
            # Let's rely on shadow_jpy tracking for total PnL, but for trade stats:
            gross_pnl = (
                exec_res.executed_price - last_entry_market_price
            ) * requested_size
            # Note: last_entry_market_price is pre-slippage. exec_res.executed_price includes slippage.
            # Actually, let's use the executed prices if we tracked them, but we only tracked market price.
            # For simplicity in this reporter, we'll use the portfolio value change if we could,
            # but we have other cash flows.
            # Let's just use the gross pnl based on market prices for the calibration map,
            # and for the reporter we can try to be more precise if needed, but gross is fine for now.

            reporter.record_trade(
                "long",
                gross_pnl,
                last_entry_market_price,
                market_data["close"],
                requested_size,  # This is approx
            )

            # Update Calibration Map
            entry_system.update_outcome(
                last_entry_regime,
                last_entry_action,
                market_data["close"] - last_entry_market_price,
                current_step,
                threshold=last_entry_threshold,
            )

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
            last_entry_market_price = market_data["close"]

            # Fee
            fee = revenue * v455_config["fee_rate"]
            shadow_jpy -= fee

        # SHORT EXIT (COVER)
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

            # Calculate PnL
            gross_pnl = (
                last_entry_market_price - market_data["close"]
            ) * requested_size

            reporter.record_trade(
                "short",
                gross_pnl,
                last_entry_market_price,
                market_data["close"],
                requested_size,
            )

            # Update Calibration Map
            entry_system.update_outcome(
                last_entry_regime,
                last_entry_action,
                last_entry_market_price - market_data["close"],
                current_step,
                threshold=last_entry_threshold,
            )

        # Track Portfolio Value
        current_price = market_data["close"]
        total_val = shadow_jpy + shadow_btc * current_price
        shadow_history.append(total_val)

        reporter.update_step(current_step, total_val, raw_action, regime, gate_status)

        if current_step % 1000 == 0:
            print(f"Step {current_step}, Shadow PV: {total_val:.0f}")

    print(f"Backtest Finished. Final Shadow PV: {shadow_history[-1]:.0f}")

    gate_logger.flush()
    reporter.finalize_stats()
    reporter.print_summary()

    # Save Results using unified utility
    results_dir = os.path.join(project_root, "backtest_results", "v455")

    # Prepare metrics for unified saving
    metrics = {
        "total_return": (shadow_history[-1] / shadow_history[0]) - 1
        if shadow_history
        else 0,
        "final_portfolio_value": shadow_history[-1] if shadow_history else 0,
        "total_trades": reporter.stats["total_trades"],
        "winning_trades": reporter.stats["winning_trades"],
        "losing_trades": reporter.stats["losing_trades"],
        "win_rate": reporter.stats["winning_trades"]
        / max(1, reporter.stats["total_trades"]),
        "net_pnl": reporter.stats["net_pnl"],
        "max_drawdown": reporter.stats["max_drawdown"],
        "max_drawdown_percent": reporter.stats["max_drawdown_percent"],
        "sharpe_ratio": reporter.stats["sharpe_ratio"],
        "gate_blocked_count": reporter.stats["gate_blocked_count"],
        "gate_allowed_count": reporter.stats["gate_allowed_count"],
        "warmup_forced_count": reporter.stats["warmup_forced_count"],
    }

    # Add action distribution
    metrics.update(
        {f"action_{k}": v for k, v in reporter.stats["action_distribution"].items()}
    )
    metrics.update(
        {f"regime_{k}": v for k, v in reporter.stats["regime_distribution"].items()}
    )

    # Save using unified utility
    saved_files = save_backtest_results(
        portfolio_values=shadow_history,
        trade_history=reporter.trade_history,
        metrics=metrics,
        output_dir=results_dir,
        filename_prefix="v455_backtest",
        metadata={
            "v455_config": v455_config,
            "base_config": config_path,
            "model_path": model_path,
            "data_path": data_path,
            "gate_log_path": os.path.join(project_root, "backtest_gate_log.csv"),
        },
    )

    print(f"Results saved to {results_dir}: {list(saved_files.keys())}")


if __name__ == "__main__":
    run_backtest_v455()
