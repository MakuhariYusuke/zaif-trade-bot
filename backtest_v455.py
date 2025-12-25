import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.signal.entry_system import IntegratedEntrySystem
from ztb.trading.execution.pseudo_hft import PseudoHFTExecutionModel
from ztb.trading.types import MarketState
from ztb.trading.constants import ACTION_BUY, ACTION_SELL, ACTION_HOLD

def run_backtest_v455():
    # 1. Load Config
    # We'll use v451 config as base but add v455 specific settings
    config_path = os.path.join(project_root, "config", "v451", "sac_v451_optimized.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # v455 Configuration
    v455_config = {
        'ewma_tau': 100.0,
        'n_min': 30.0,
        'fee_rate': 0.001,
        'c_spread': 0.3,
        'c_vol': 0.2,
        'c_imp': 0.5,
        'gamma': 0.5,
        'min_volume': 0.01,
        'latency_sec': 1.0,
        'order_size_btc': 0.01
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
    model_path = os.path.join(project_root, "models", "sac_v451_phase7_regime_aware.zip")
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = os.path.join(project_root, "checkpoints", "v451", "phase7", "best_model.zip")
    
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
    last_entry_price = 0.0
    last_entry_action = 0.0
    last_entry_regime = "unknown"

    obs, _ = env.reset()
    done = False
    
    print("Starting v455 Backtest...")
    
    while not done:
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
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'atr': float(row.get('atr', row.get('ATR', 0.0))),
                'volume': float(row['volume']),
                'timestamp': row.name
            }
        except Exception:
            # Fallback if index out of bounds or columns missing
            market_data = {
                'high': 0.0, 'low': 0.0, 'close': 0.0, 'atr': 0.0, 'volume': 0.0, 'timestamp': None
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
        is_sell_signal = raw_action < negative_threshold # Short entry? Or exit?
        # Assuming Long-Only for simplicity or check env config. 
        # HeavyEnv usually supports Long/Short if configured.
        
        action_to_env = action_rl
        
        # If we are flat and get a signal, check Gate
        if shadow_btc == 0.0:
            if is_buy_signal:
                gate_res = entry_system.process_signal(raw_action, market_data, regime, order_size=v455_config['order_size_btc'])
                if not gate_res['should_enter']:
                    # Gate blocked entry
                    # We must force Env to HOLD to keep sync
                    # But we can't easily change the continuous action to "Hold" without knowing the mapping perfectly.
                    # Usually 0.0 is Hold.
                    action_to_env = np.array([0.0], dtype=np.float32)
                else:
                    # Gate allowed entry
                    last_entry_action = raw_action
                    last_entry_regime = regime
            elif is_sell_signal:
                # Short entry logic if supported... assuming Long-Only for now or symmetric
                pass

        # 4. Step Env
        # We pass the potentially modified action
        obs, reward, terminated, truncated, info = env.step(action_to_env)
        done = terminated or truncated
        
        # 5. Shadow Execution
        # Check if Env position changed
        # We need to track Env's position to detect trades
        # But wait, if we blocked entry, Env shouldn't have traded.
        # If we allowed entry, Env might have traded.
        
        # Ideally we want to calculate OUR OWN execution price using PseudoHFT
        # and update OUR shadow portfolio.
        # But we need to know IF a trade happened.
        # We can check if action_to_env was Buy/Sell and we were flat/long.
        
        # Simplified logic:
        # If we sent a Buy/Sell action (not 0.0), assume execution happened if Env logic allows.
        # But Env has internal checks (funds, etc.).
        # Let's rely on `info` or `env.position` change?
        # `env.position` is reliable.
        
        # Note: This requires us to know previous position.
        # But we are running in a loop.
        
        # Actually, since we want to test v455 logic, maybe we should just run the logic 
        # and NOT rely on Env's internal position for PnL, but we DO need Env to track position 
        # so that observations are correct.
        
        # Let's assume Env execution is "perfect" (price = close), and we apply slippage on top.
        # Or we use PseudoHFT to get the price.
        
        # Detect Trade
        # We need to store previous position of Env?
        # Env.position is updated in step().
        # But we don't have "prev_position" easily unless we tracked it.
        # Let's track it.
        
        # Actually, `info` might contain trade info?
        # HeavyEnv info usually contains 'portfolio_value', 'pnl', etc.
        
        # Let's just implement a simple state machine here matching Env.
        # If action_to_env was BUY and shadow_btc == 0:
        #    Execute BUY
        # If action_to_env was SELL and shadow_btc > 0:
        #    Execute SELL
        
        # Execute BUY
        if isinstance(action_to_env, np.ndarray):
            act_val = float(action_to_env[0])
        else:
            act_val = float(action_to_env)
            
        discrete_act = 0
        if act_val > threshold: discrete_act = 1 # Buy
        elif act_val < negative_threshold: discrete_act = 2 # Sell
        
        if discrete_act == 1 and shadow_btc == 0.0:
            # BUY
            requested_size = v455_config['order_size_btc'] # Fixed size for test
            # Check funds? Assuming infinite for backtest or check shadow_jpy
            
            exec_res = execution_model.simulate_execution(
                'buy', market_data['close'], requested_size, 
                current_atr=market_data['atr'], current_volume=market_data['volume'], market_regime=regime,
                market_data=market_data
            )
            
            cost = exec_res.executed_price * exec_res.executed_size
            if shadow_jpy >= cost:
                shadow_jpy -= cost
                shadow_btc += exec_res.executed_size
                last_entry_price = exec_res.executed_price
                # Fee? PseudoHFT returns fee in result?
                # ExecutionResult has fee field.
                # But PseudoHFT currently returns 0.0 fee (comment says handled by FeeModel).
                # Let's add fee manually: 0.1%
                fee = cost * v455_config['fee_rate']
                shadow_jpy -= fee
        
        elif discrete_act == 2 and shadow_btc > 0.0:
            # SELL
            requested_size = shadow_btc # Sell all
            
            exec_res = execution_model.simulate_execution(
                'sell', market_data['close'], requested_size,
                current_atr=market_data['atr'], current_volume=market_data['volume'], market_regime=regime,
                market_data=market_data
            )
            
            revenue = exec_res.executed_price * exec_res.executed_size
            shadow_jpy += revenue
            shadow_btc = 0.0
            
            # Fee
            fee = revenue * v455_config['fee_rate']
            shadow_jpy -= fee
            
            # Update Calibration Map
            # Calculate Gross PnL per unit
            # (ExitPrice - EntryPrice) / EntryPrice? Or absolute?
            # CalibrationMap expects "Gross PnL per unit (JPY/BTC)"
            gross_pnl_per_unit = exec_res.executed_price - last_entry_price
            
            entry_system.update_outcome(last_entry_regime, last_entry_action, gross_pnl_per_unit, current_step)

        # Track Portfolio Value
        current_price = market_data['close']
        total_val = shadow_jpy + shadow_btc * current_price
        shadow_history.append(total_val)
        
        if current_step % 1000 == 0:
            print(f"Step {current_step}, Shadow PV: {total_val:.0f}")

    print(f"Backtest Finished. Final Shadow PV: {shadow_history[-1]:.0f}")
    
    # Save Results
    results_dir = os.path.join(project_root, "backtest_results", "v455")
    os.makedirs(results_dir, exist_ok=True)
    
    pd.DataFrame({'portfolio_value': shadow_history}).to_csv(os.path.join(results_dir, "shadow_results.csv"))
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    run_backtest_v455()
