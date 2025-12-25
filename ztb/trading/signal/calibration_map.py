import numpy as np
from scipy.stats import beta
from typing import Dict, Any, Tuple, Optional, List, cast
import math
from ztb.trading.environment.constants import EPSILON
from ztb.trading.signal.types import CalibrationStats, CalibrationStatsBundle, FusedSignal, GateResult
from ztb.trading.types import MarketState

class CalibrationMap:
    """
    Calibration Map for HFT Strategy (v455).
    Manages historical performance stats (Win Rate, AvgWin, AvgLoss)
    using hierarchical fallback (Specific -> Regime -> Global) and EWMA.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.tau = float(config.get('ewma_tau', 100.0)) # Time constant for EWMA
        self.n_min = float(config.get('n_min', 30.0))   # Min effective sample size for trust
        
        # Storage for stats: {key: {sum_w, sum_w_sq, w_avg_win, w_avg_loss, ...}}
        # Keys: 
        # - Level 1: f"{regime}_{action_bin}"
        # - Level 2: f"{regime}"
        # - Level 3: "global"
        self.stats: Dict[str, Dict[str, float]] = {}
        
        # Initialize global stats
        self._init_stats("global")

    def _init_stats(self, key: str) -> None:
        if key not in self.stats:
            self.stats[key] = {
                # alpha/beta removed as they are redundant with w_sum_wins/losses + prior
                'sum_w': 0.0,
                'sum_w_sq': 0.0,
                'w_sum_win_amt': 0.0, # Weighted sum of win amounts (gross)
                'w_sum_loss_amt': 0.0, # Weighted sum of loss amounts (gross, absolute)
                'w_sum_wins': 0.0,    # Weighted count of wins
                'w_sum_losses': 0.0,  # Weighted count of losses
                'last_update_step': 0.0 # For time-decay
            }

    def _get_bin(self, action: float) -> str:
        """Bin the RL action into discrete categories."""
        if action > 0.6: return "Strong_Buy"
        if action > 0.2: return "Buy"
        if action > -0.2: return "Neutral"
        if action > -0.6: return "Sell"
        return "Strong_Sell"

    def update(self, regime: str, action: float, gross_pnl: float, step: int) -> None:
        """
        Update stats with new trade result.
        gross_pnl: Gross PnL per unit (JPY/BTC).
        """
        action_bin = self._get_bin(action)
        keys = [f"{regime}_{action_bin}", regime, "global"]
        
        is_win = gross_pnl > 0
        abs_pnl = abs(gross_pnl)
        
        for key in keys:
            self._init_stats(key)
            stats = self.stats[key]
            
            dt = step - stats['last_update_step']
            decay = math.exp(-dt / self.tau) if dt > 0 else 1.0
            
            # Decay existing stats
            stats['sum_w'] *= decay
            stats['sum_w_sq'] *= (decay * decay) # Squared weights decay with square of factor
            stats['w_sum_win_amt'] *= decay
            stats['w_sum_loss_amt'] *= decay
            stats['w_sum_wins'] *= decay
            stats['w_sum_losses'] *= decay
            
            # Add new observation (weight = 1.0)
            w = 1.0
            stats['sum_w'] += w
            stats['sum_w_sq'] += w * w
            
            if is_win:
                stats['w_sum_win_amt'] += w * abs_pnl
                stats['w_sum_wins'] += w
            else:
                stats['w_sum_loss_amt'] += w * abs_pnl
                stats['w_sum_losses'] += w
                
            stats['last_update_step'] = float(step)

    def get_stats(self, regime: str, action: float) -> CalibrationStatsBundle:
        """
        Retrieve stats with hierarchical fallback.
        Returns dictionary with p_win_lcb, avg_win, avg_loss, n_eff, etc.
        """
        action_bin = self._get_bin(action)
        key_l1 = f"{regime}_{action_bin}"
        key_l2 = regime
        key_l3 = "global"
        
        stats_l1 = self._compute_metrics(key_l1)
        stats_l2 = self._compute_metrics(key_l2)
        stats_l3 = self._compute_metrics(key_l3)
        
        # Determine fallback: L2 if trusted, else L3.
        if stats_l2['n_eff'] >= self.n_min:
            stats_fallback = stats_l2
        else:
            stats_fallback = stats_l3
            
        return {
            'l1': stats_l1,
            'fallback': stats_fallback,
            'n_min': self.n_min
        }

    def _compute_metrics(self, key: str) -> CalibrationStats:
        if key not in self.stats:
            # Return default/prior values
            return {
                'p_win_lcb': 0.0, # Conservative
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'n_eff': 0.0
            }
            
        s = self.stats[key]
        
        # n_eff
        n_eff = (s['sum_w'] ** 2) / (s['sum_w_sq'] + EPSILON)
        
        # p_win_LCB (Beta Inverse)
        alpha_post = 2.0 + s['w_sum_wins']
        beta_post = 2.0 + s['w_sum_losses']
        
        p_win_lcb = beta.ppf(0.05, alpha_post, beta_post)
        
        # AvgWin / AvgLoss
        avg_win = s['w_sum_win_amt'] / (s['w_sum_wins'] + EPSILON)
        avg_loss = s['w_sum_loss_amt'] / (s['w_sum_losses'] + EPSILON)
        
        return {
            'p_win_lcb': float(p_win_lcb),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'n_eff': n_eff
        }

class CalibrationGate:
    """
    Calibration Gate for HFT Strategy.
    Evaluates entry opportunity using CalibrationMap and CostModel.
    """
    def __init__(self, config: Dict[str, Any], calibration_map: CalibrationMap) -> None:
        self.config = config
        self.calibration_map = calibration_map
        # Cost parameters
        self.fee_rate = float(config.get('fee_rate', 0.0)) # e.g. 0.001 for 0.1%
        self.c_spread = float(config.get('c_spread', 0.3))
        self.c_vol = float(config.get('c_vol', 0.2))
        self.c_imp = float(config.get('c_imp', 0.5))
        self.gamma = float(config.get('gamma', 0.5))
        self.min_volume = float(config.get('min_volume', 0.01))
        self.latency_sec = float(config.get('latency_sec', 1.0))
        self.order_size_btc = float(config.get('order_size_btc', 0.01))

    def evaluate(self, fused_signal: FusedSignal, market_data: MarketState, order_size: Optional[float] = None) -> GateResult:
        """
        Evaluate entry.
        fused_signal: {rl_action, regime, ...}
        market_data: {high, low, close, atr, volume, ...}
        order_size: Size of the order to estimate impact. Defaults to config size if None.
        """
        rl_action = fused_signal['rl_action']
        regime = fused_signal['regime']
        
        # Use passed order size or default
        size_to_use = order_size if order_size is not None else self.order_size_btc

        # 1. Get Stats
        stats_bundle = self.calibration_map.get_stats(regime, rl_action)
        stats_l1 = stats_bundle['l1']
        stats_fb = stats_bundle['fallback']
        n_min = stats_bundle['n_min']
        
        # 2. Estimate Cost
        cost = self._estimate_cost(market_data, rl_action, size_to_use)
        
        # 3. Calculate EV for L1 and Fallback
        ev_l1 = self._calculate_ev(stats_l1, cost)
        ev_fb = self._calculate_ev(stats_fb, cost)
        
        # 4. Blend EV
        n_eff = stats_l1['n_eff']
        
        if n_min > EPSILON:
            lambda_val = min(1.0, n_eff / n_min)
        else:
            # If n_min is 0, we fully trust L1 if it has any data, otherwise... 
            # Actually if n_min=0, we should probably just trust L1.
            lambda_val = 1.0

        ev_final = lambda_val * ev_l1 + (1.0 - lambda_val) * ev_fb
        
        return {
            'should_enter': ev_final > 0,
            'ev': ev_final,
            'ev_l1': ev_l1,
            'ev_fb': ev_fb,
            'lambda_val': lambda_val,
            'cost': cost,
            'stats': stats_l1, # Return L1 stats for logging
            'stats_fallback': stats_fb
        }

    def _calculate_ev(self, stats: CalibrationStats, cost: float) -> float:
        p_win = stats['p_win_lcb']
        avg_win = stats['avg_win']
        avg_loss = stats['avg_loss']
        
        # EV = p * Win - (1-p) * Loss - Cost
        return p_win * avg_win - (1.0 - p_win) * avg_loss - cost

    def _estimate_cost(self, market_data: MarketState, action: float, order_size: float) -> float:
        """
        Estimate Round-Trip Cost (Fee + Slippage).
        Unit: JPY/BTC
        """
        # Extract market data
        high = market_data.get('high', 0.0)
        low = market_data.get('low', 0.0)
        close = market_data.get('close', 0.0) # Use close as proxy for price
        atr = market_data.get('atr', 0.0)
        volume = market_data.get('volume', 0.0)
        
        # Fail-closed check for missing data
        # Added check for close <= EPSILON as Fee depends on it
        if (high <= EPSILON or 
            low <= EPSILON or 
            atr <= EPSILON or 
            volume <= EPSILON or 
            close <= EPSILON):
            # Return infinite cost to prevent entry
            return float('inf')

        # 1. Fee (Round-trip approx)
        # Fee = Price * 2 * FeeRate
        fee_roundtrip = close * 2 * self.fee_rate
        
        # 2. Slippage (One-way)
        # Spread Proxy
        spread_proxy = self.c_spread * (high - low)
        
        # Volatility Risk (Latency assumed 1 sec for now, or config)
        vol_risk = self.c_vol * atr * math.sqrt(self.latency_sec / 60.0)
        
        # Market Impact
        impact = self.c_imp * atr * ((order_size / max(volume, self.min_volume)) ** self.gamma)
        
        slippage_one_way = spread_proxy + vol_risk + impact
        slippage_roundtrip = slippage_one_way * 2
        
        return fee_roundtrip + slippage_roundtrip
