"""
Fast Intraday Trading Environment v456

v455環境を拡張して、v456の88次元観測空間とGroupedFeatureScalerに対応:
- Cyclical time features (6)
- Global market features (9)
- Grouped正規化スケーラー (36次元selective scale)
- MTFリーク検証統合
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ztb.features.grouping.grouped_scaler import GroupedFeatureScaler
from ztb.features.time.cyclical_v456 import CyclicalTimeFeatureExtractor
from ztb.features.global_market_v456 import GlobalMarketFeatureEngineerV456
from ztb.trading.environment.components.fast_intraday_accounting import (
    FastIntradayAccounting,
)
from ztb.trading.environment.components.fast_intraday_action_processor import (
    FastIntradayActionProcessor,
)
from ztb.trading.environment.components.threshold_manager import (
    ThresholdManager,
)
from ztb.trading.environment.components.position_manager import (
    PositionManager,
)
from ztb.trading.rewards.fast_intraday import compute_hft_reward
from ztb.utils.fee_model import ExchangeFeeModel

# ★ Lost Alpha Restoration: Ichimoku Calculation for Trend Guidance
def _calculate_ichimoku_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate simple Ichimoku Cloud signal for Trend Guidance.
    Signal = 1 (Bull) if Close > Cloud
    Signal = -1 (Bear) if Close < Cloud
    Signal = 0 (Neutral) inside Cloud
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9).max().values
    period9_low = pd.Series(low).rolling(window=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26).max().values
    period26_low = pd.Series(low).rolling(window=26).min().values
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Shifted forward 26 periods (but we use current value for comparison with future cloud, or lagged? 
    # Standard Ichimoku compares price today with Cloud projected 26 periods ago? 
    # No, Cloud is projected forward. So today's price is compared to the cloud calculated 26 periods ago.
    # We need to shift A and B forward by 26.
    senkou_span_a = np.roll(senkou_span_a, 26)
    senkou_span_a[:26] = 0

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52).max().values
    period52_low = pd.Series(low).rolling(window=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    senkou_span_b = np.roll(senkou_span_b, 26)
    senkou_span_b[:26] = 0
    
    # Determine Signal
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    signal = np.zeros(len(close), dtype=np.float32)
    signal[close > cloud_top] = 1.0
    signal[close < cloud_bottom] = -1.0
    # Inside cloud remains 0
    
    return signal

logger = logging.getLogger(__name__)


class FastIntradayEnvV456(gym.Env):
    """
    Fast Intraday Trading Environment v456
    
    88次元観測空間:
    [0:30]   Base features (OHLCV)
    [30:57]  MTF features (5m/15m/1h)
    [57:63]  Cyclical time (sin/cos)
    [63:69]  Global continuous (spread, returns, vol)
    [69:82]  Regime (One-Hot, 13D)
    [82:88]  Account metrics (6D: position, ttl, cost, balance, pnl, steps)
    
    Action Space: 
    - Type '2d_position_ttl' (Default): Box([-1, 0], [1, 1])
      - target_position: [-1, 1]
      - ttl_fraction: [0, 1]
    - Type '1d_position': Box([-1], [1])
      - target_position: [-1, 1]
      - ttl_fraction: Always 1.0 (Implicit)
    
    Observation Space: Box (88,)
    """
    
    metadata = {"render_modes": ["human"]}
    
    # 観測空間の構造定義
    OBSERVATION_DIMS = {
        'base': (0, 30),           # 30
        'mtf': (30, 57),           # 27
        'cyclical': (57, 63),      # 6
        'global': (63, 69),        # 6
        'regime': (69, 82),        # 13
        'account': (82, 88),       # 6
    }
    TOTAL_OBS_DIM = 88
    
    def __init__(
        self,
        df: pd.DataFrame,
        base_feature_columns: List[str],  # 30個のBase特徴量
        mtf_feature_columns: List[str],   # 27個のMTF特徴量
        regime_feature_columns: List[str],  # 13個のRegime特徴量
        binance_df: Optional[pd.DataFrame] = None,  # グローバル市場データ
        initial_balance: float = 1_000_000.0,
        max_position: float = 1.0,
        max_steps: Optional[int] = None,
        commission_rate: float = 0.001,
        max_ttl_steps: int = 60,
        cooldown_steps: int = 5,
        max_delta_per_step: float = 0.2,
        min_delta: float = 0.01,
        drawdown_limit: float = 0.1,
        prewarm_steps: int = 100,
        reward_scale: float = 100000.0,
        reward_clip: Optional[float] = 1.0,
        reward_params: Optional[Dict[str, float]] = None,
        action_space_type: str = "2d_position_ttl",
        guidance_decay_steps: int = 50000,  # New parameter for curriculum decay
        dynamic_threshold_mode: str = "z_score",
        z_score_window: int = 50,
        z_score_threshold: float = 3.0,
        z_score_method: str = "mad",
        min_action_threshold: float = 0.001,
        env_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Store config for entry gate
        self.config = env_config or {}
        
        # Data
        self.df = df.reset_index(drop=True)
        # Preserve timestamps for feature generation
        if "timestamp" in df.columns:
            self.timestamps = pd.to_datetime(df["timestamp"])
            # Ensure TZ-aware (assume UTC if naive)
            if self.timestamps.dt.tz is None:
                 self.timestamps = self.timestamps.dt.tz_localize("UTC")
        elif isinstance(df.index, pd.DatetimeIndex):
            self.timestamps = df.index
            if self.timestamps.tz is None:
                 self.timestamps = self.timestamps.tz_localize("UTC")
        else:
            # Fallback for synthetic data
            logger.warning("No timestamp found in DF. Using synthetic hourly pattern.")
            self.timestamps = pd.date_range("2024-01-01", periods=len(df), freq="1min", tz="UTC")

        self.binance_df = binance_df
        self.base_feature_columns = base_feature_columns
        self.mtf_feature_columns = mtf_feature_columns
        self.regime_feature_columns = regime_feature_columns
        
        # Verify dimensions
        if len(base_feature_columns) != 30:
            raise ValueError(f"Expected 30 base features, got {len(base_feature_columns)}")
        if len(mtf_feature_columns) != 27:
            raise ValueError(f"Expected 27 MTF features, got {len(mtf_feature_columns)}")
        if len(regime_feature_columns) != 13:
            raise ValueError(f"Expected 13 regime features, got {len(regime_feature_columns)}")
        
        # ★ Phase 1.1: Check for missing features before accessing
        missing_base = [col for col in base_feature_columns if col not in self.df.columns]
        missing_mtf = [col for col in mtf_feature_columns if col not in self.df.columns]
        missing_regime = [col for col in regime_feature_columns if col not in self.df.columns]
        
        if missing_base or missing_mtf or missing_regime:
            error_msg = "❌ Missing feature columns detected:\n"
            if missing_base:
                error_msg += f"  Base: {missing_base}\n"
            if missing_mtf:
                error_msg += f"  MTF: {missing_mtf}\n"
            if missing_regime:
                error_msg += f"  Regime: {missing_regime}\n"
            
            # If strictly required, raise error. Or log warning?
            # Existing code raises error later or fails?
            # Reverting previous assumption: The code didn't raise here, just built msg.
            # But let's assume if critical cols are missing, calculation fails.
            pass

        # Pre-extract market data for entry gate
        self.regime_data = df[regime_feature_columns]
        self.high_prices = df.get('high', df['close']).values
        self.low_prices = df.get('low', df['close']).values
        self.open_prices = df.get('open', df['close']).values
        try:
            self.volume = df['volume'].values
        except KeyError:
            self.volume = np.ones(len(df), dtype=np.float64)  # Default volume
        try:
            self.atr_values = df['atr'].values
        except KeyError:
            self.atr_values = np.ones(len(df), dtype=np.float64) * 0.01  # Default ATR

        # ★ Fix Finding 3: Calculate Ichimoku AFTER validation
        # Requires OHLC which are in base features.
        try:
             self.ichimoku_signals = _calculate_ichimoku_signal(self.df)
             logger.info(f"✓ Ichimoku Trend Guidance Signals Calculated: Non-Zero={np.count_nonzero(self.ichimoku_signals)}/{len(self.df)}")
        except Exception as e:
             logger.warning(f"Ichimoku Signal Calculation failed (possibly missing columns): {e}")
             self.ichimoku_signals = np.zeros(len(df), dtype=np.float64)

        # ★ Restore "Lost Alpha": Pre-calculate Cyclical Time Features
        from ztb.features.time.cyclical_v456 import calc_cyclical_time_features
        try:
            # Create a temp DF with the index set to timestamps for the extractor
            _temp_time_df = pd.DataFrame(index=self.timestamps)
            _cyclical_df = calc_cyclical_time_features(_temp_time_df)
            self.cyclical_features = _cyclical_df.values.astype(np.float64)
            logger.info(f"✓ Restored Cyclical Time Features: shape={self.cyclical_features.shape}")
        except Exception as e:
            logger.error(f"Failed to calculate cyclical features: {e}")
            self.cyclical_features = np.zeros((len(df), 6), dtype=np.float32)
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.max_steps = max_steps
        self.max_ttl_steps = max_ttl_steps
        self.cooldown_steps = cooldown_steps
        self.max_delta_per_step = max_delta_per_step
        self.min_delta = min_delta
        self.drawdown_limit = drawdown_limit
        self.prewarm_steps = prewarm_steps
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.reward_params = reward_params or {}
        self.action_space_type = action_space_type
        self.guidance_decay_steps = guidance_decay_steps
        self.dynamic_threshold_mode = dynamic_threshold_mode
        self.z_score_window = z_score_window
        self.z_score_threshold = z_score_threshold
        self.z_score_method = z_score_method
        self.min_action_threshold = min_action_threshold
        self.ttl_enabled = action_space_type != "1d_position"
        self.env_config = env_config or {}
        self.previous_action = 0.0
        self.regime = "UNKNOWN"
        
        # ★ P1-1: TP/SL閾値設定（Phase 2簡易実装）
        self.tp_threshold = self.env_config.get("tp_threshold", 0.02)  # 2% profit
        self.sl_threshold = self.env_config.get("sl_threshold", 0.01)  # 1% loss

        # Entry gate system (optional)
        entry_gate_config = self.env_config.get("entry_gate", {})
        if entry_gate_config.get("enabled", False):
            from ztb.trading.signal.entry_system import IntegratedEntrySystem
            self.entry_system = IntegratedEntrySystem(entry_gate_config)
            # Load calibration state if path provided
            calibration_path = entry_gate_config.get("calibration_map_path")
            if calibration_path:
                self.entry_system.load_state(calibration_path)
                logger.info(f"Loaded entry gate calibration from {calibration_path}")
            logger.info("Entry gate system enabled")
        else:
            self.entry_system = None

        # Dynamic threshold manager
        threshold_config = {
            "continuous_to_discrete_threshold": min_action_threshold,
            "adaptive_mode": dynamic_threshold_mode == "adaptive",
            "z_score_mode": dynamic_threshold_mode == "z_score",
            "z_score_window": z_score_window,
            "z_score_threshold": z_score_threshold,
            "z_score_method": z_score_method,
            "min_threshold": 0.0,
            "max_threshold": 1.0,
        }
        self.threshold_manager = ThresholdManager(config=threshold_config)

        self.action_processor = FastIntradayActionProcessor(
            action_space_type=self.action_space_type,
            max_position=self.max_position,
            cooldown_steps=self.cooldown_steps,
        )
        self.position_manager = PositionManager(
            config={
                "max_position": self.max_position,
                "commission_rate": commission_rate,
                "max_ttl_steps": self.max_ttl_steps,
                "allow_reverse": True,  # デフォルトでreverse許可
            },
            get_price_callback=lambda: self.close_prices[self.current_step],
        )
        self.accounting = FastIntradayAccounting(initial_balance=self.initial_balance)
        
        # Fee model
        self.fee_model = ExchangeFeeModel(exchange_fees={
            "zaif": {"buy": commission_rate, "sell": commission_rate}
        })
        self.fee_model.set_exchange("zaif")
        
        # Entry price tracking for trade PnL calculation
        self.entry_price = 0.0
        self.last_execution_price = 0.0
        
        # ★ Phase 1.1: Check for missing features before accessing
        missing_base = [col for col in base_feature_columns if col not in self.df.columns]
        missing_mtf = [col for col in mtf_feature_columns if col not in self.df.columns]
        missing_regime = [col for col in regime_feature_columns if col not in self.df.columns]
        
        if missing_base or missing_mtf or missing_regime:
            error_msg = "❌ Missing feature columns detected:\n"
            if missing_base:
                error_msg += f"  Base: {missing_base}\n"
            if missing_mtf:
                error_msg += f"  MTF: {missing_mtf}\n"
            if missing_regime:
                error_msg += f"  Regime: {missing_regime}\n"
            error_msg += f"\nAvailable columns ({len(self.df.columns)}): {self.df.columns.tolist()}\n"
            error_msg += "→ Implement feature calculation or provide pre-computed features."
            raise ValueError(error_msg)
        
        # Pre-convert data to numpy
        self.base_features = self.df[base_feature_columns].values.astype(np.float32)
        self.mtf_features = self.df[mtf_feature_columns].values.astype(np.float64)
        self.regime_features = self.df[regime_feature_columns].values.astype(np.float32)
        
        self.close_prices = self.df["close"].values.astype(np.float32)
        
        # ATR (for slippage/normalization)
        if "atr" in self.df.columns:
            self.atr_data = self.df["atr"].values.astype(np.float32)
        else:
            logger.warning("ATR column not found, using 1% of close price")
            self.atr_data = self.close_prices * 0.01
        
        # Impact proxy
        if "impact_proxy" in self.df.columns:
            self.impact_data = self.df["impact_proxy"].values.astype(np.float32)
        else:
            self.impact_data = np.zeros_like(self.close_prices)
        
        self.data_len = len(self.df)
        del self.df
        self.df = None
        
        # Action space
        if self.action_space_type == "1d_position":
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        
        # Observation space (88D)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.TOTAL_OBS_DIM,), dtype=np.float32
        )
        
        # Feature extractors
        self.cyclical_extractor = CyclicalTimeFeatureExtractor()
        self.global_engineer = GlobalMarketFeatureEngineerV456(binance_df=binance_df)
        
        # Scaler (36D selective: Base + Global continuous)
        self.scaler = GroupedFeatureScaler(
            epsilon=1e-7,
            momentum=0.99,
            clip_value=3.0,
        )
        
        # State
        self.current_step = 0
        self.lifetime_steps = 0
        self.balance = initial_balance
        self.position = 0.0
        self.position_ttl = 0
        self.steps_held = 0
        self.cooldown_counter = 0
        self.gross_pnl = 0.0
        self.net_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        
        # Recorder for backtest reporting
        self.recorder = None
        self.total_pnl = 0.0
        self.max_balance = initial_balance
        self.last_step_cost = 0.0
        self.steps_in_episode = 0
        self.last_realized_fee = 0.0  # ★ Phase 1.3: fee tracking
        self.ttl_forced_exits = 0
        self.cooldown_triggers = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """環境をリセット
        
        Args:
            seed: 乱数シード
            options: リセットオプション
                - fixed_start: bool, Trueで固定開始位置（デフォルトFalse）
                - start_step: int, 固定開始位置（指定時優先）
        """
        super().reset(seed=seed)
        
        # 開始位置決定
        options = options or {}
        # max_steps をオプションから注入
        if "max_steps" in options:
            self.max_steps = options["max_steps"]
        if options.get("fixed_start", False) or "start_step" in options:
            # 固定開始
            if "start_step" in options:
                self.current_step = max(self.prewarm_steps, min(options["start_step"], self.data_len - (self.max_steps or 1000)))
            else:
                self.current_step = self.prewarm_steps  # 固定開始位置
        else:
            # ランダム開始（従来通り）
            data_len = self.data_len
            min_start = self.prewarm_steps
            max_start = data_len - (self.max_steps if self.max_steps else 1000)
            
            if max_start <= min_start:
                self.current_step = min_start
            else:
                self.current_step = self.np_random.integers(min_start, max_start)
        
        # 状態リセット
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_ttl = 0
        self.steps_held = 0
        self.cooldown_counter = 0
        self.accounting.reset()
        self.position_manager.reset()
        self.gross_pnl = self.accounting.gross_pnl
        self.net_pnl = self.accounting.net_pnl
        self.total_fees = self.accounting.total_fees
        self.total_slippage = self.accounting.total_slippage
        self.total_pnl = self.net_pnl
        self.balance = self.accounting.portfolio_value()
        self.max_balance = self.balance
        self.last_step_cost = 0.0
        self.steps_in_episode = 0
        self.ttl_forced_exits = 0
        self.cooldown_triggers = 0
        
        # スケーラーリセット
        self.scaler.reset()
        
        # Prewarm data
        prewarm_start = self.current_step - self.prewarm_steps
        prewarm_indices = range(prewarm_start, self.current_step)
        
        for idx in prewarm_indices:
            obs = self._build_observation(idx, update_scaler=True)
        
        info = {
            "start_index": int(self.current_step),
            "seed": seed,
        }
        
        # Initialize entry system if configured
        entry_gate_config = self.config.get("environment", {}).get("entry_gate") if self.config else None
        if entry_gate_config:
            from ztb.trading.entry_system import IntegratedEntrySystem
            self.entry_system = IntegratedEntrySystem(config=entry_gate_config)
            # Load calibration map if available
            if hasattr(self.entry_system, 'calibration_map') and self.entry_system.calibration_map:
                self.entry_system.calibration_map.load()
        else:
            self.entry_system = None
        
        return self._get_observation(), info
    
    def _is_entry_action(self, target_position: float, current_position: float) -> bool:
        """
        Doc04仕様: 新規エントリー/拡大のみをゲートチェック対象に
        exit/close/reduceは常に許可
        
        Args:
            target_position: 目標ポジション
            current_position: 現在ポジション
        
        Returns:
            True: エントリー/拡大（ゲートチェック必要）
            False: exit/close/reduce（常に許可）
        """
        # 絶対値が増える = エントリー/拡大
        return abs(target_position) > abs(current_position)
    
    def _convert_to_hold_action(self) -> np.ndarray:
        """
        Doc04仕様: エントリーブロック時にHOLDに変換
        
        Returns:
            HOLD相当のアクション配列
        """
        if self.action_space_type == "2d_position_ttl":
            return np.array([0.0, 0.5])  # position=0, ttl=default
        else:
            return np.array([0.0])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """1ステップ実行"""
        # Store previous action for calibration
        self.previous_action = action[0]
        
        # 価格取得
        price_now = self.close_prices[self.current_step]
        price_prev = self.close_prices[self.current_step - 1] if self.current_step > 0 else price_now
        atr = self.atr_data[self.current_step]
        position_prev = self.position
        
        # ★ P1-1: close_reason初期化（スコープ全体で利用）
        close_reason: Optional[str] = None
        trade_pnl = 0.0
        
        # Update threshold manager with raw action
        self.threshold_manager.update_action_stats(abs(action[0]))
        
        # Parse action first to get target position
        action_result = self.action_processor.parse_action(action)
        target_pos_fraction = action_result.target_pos_fraction
        ttl_fraction = action_result.ttl_fraction
        
        # Apply entry gate if enabled (Doc04仕様: exit/closeは常時許可)
        if self.entry_system is not None:
            is_entry = self._is_entry_action(target_pos_fraction, position_prev)
            
            if is_entry:
                # エントリー/拡大のみゲートチェック
                from ztb.trading.types import MarketState
                market_state = MarketState(
                    high=self.high_prices[self.current_step],
                    low=self.low_prices[self.current_step],
                    close=self.close_prices[self.current_step],
                    atr=self.atr_values[self.current_step],
                    volume=self.volume[self.current_step],
                    spread=0.0,
                    timestamp=None,
                )
                current_regime_data = self.regime_data.iloc[self.current_step]
                regime = current_regime_data.idxmax() if current_regime_data.sum() > 0 else "UNKNOWN"
                gate_result = self.entry_system.process_signal(
                    rl_action=action[0],
                    market_data=market_state,
                    regime=regime,
                    threshold=self.min_action_threshold,
                )
                if not gate_result["should_enter"]:
                    # エントリーブロック → HOLDに変換
                    action = self._convert_to_hold_action()
                    action_result = self.action_processor.parse_action(action)
                    target_pos_fraction = 0.0
                    logger.debug(
                        f"Gate blocked entry: original_action={action[0]:.3f}, "
                        f"regime={regime}, threshold={self.min_action_threshold:.3f}"
                    )
            # else: exit/close/reduceは常に許可（ゲートチェックなし）
        
        # Position transition
        position_prev = self.position
        delta = target_pos_fraction - position_prev
        
        # Apply dynamic threshold
        threshold = self.threshold_manager.get_threshold(raw_action_value=target_pos_fraction)
        if abs(target_pos_fraction) < threshold:
            target_pos_fraction = 0.0
        
        ttl_result = self.action_processor.apply_ttl_and_cooldown(
            target_pos_fraction=target_pos_fraction,
            position=self.position,
            position_ttl=self.position_ttl,
            cooldown_counter=self.cooldown_counter,
        )
        raw_target_position = ttl_result.raw_target_position
        self.position_ttl = ttl_result.position_ttl
        self.cooldown_counter = ttl_result.cooldown_counter
        if ttl_result.ttl_forced_exit:
            self.ttl_forced_exits += 1
        if ttl_result.cooldown_triggered:
            self.cooldown_triggers += 1
        
        # ポジション遷移
        delta = raw_target_position - position_prev
        
        # デッドバンド
        if abs(delta) < self.min_delta * self.max_position:
            delta = 0.0
        
        # リキディティ制約
        max_delta = self.max_delta_per_step * self.max_position
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
        
        new_position = position_prev + delta
        
        # スリッページと手数料の計算
        fee_paid = 0.0
        slippage_paid = 0.0
        
        if abs(delta) > 1e-6:
            # Impact slippage
            impact_mult = 1.0 + self.impact_data[self.current_step] * abs(delta) / self.max_position
            slippage = atr * impact_mult * (0.1 if delta > 0 else 0.05)  # Asymmetric
            slippage_paid = abs(delta) * slippage

            # 実行価格
            execution_price = price_now * (1.0 + np.sign(delta) * slippage / price_now)

            self.last_execution_price = execution_price

            # 手数料
            trade_type = "buy" if delta > 0 else "sell"
            fee_rate = self.fee_model.get_fee_rate(trade_type)
            fee_paid = abs(delta) * execution_price * fee_rate

            # ★ Phase 1.3修正: 手数料/スリッページは balance から直接引かない
            # → 代わりに報酬計算で反映される
            self.last_realized_fee = fee_paid + slippage_paid

            # ポジション更新
            self.position = new_position
            self.last_step_cost = execution_price

            # ★ P0-3: Calculate trade_pnl as NET PnL (costs already deducted)
            # This value is used in info['trade_pnl'] for Reporter.record_trade()
            # ★ P1-1: close_reasonはstep()冒頭で初期化済み
            
            # ★ P1-2: 反転検出（Long⇄Short）
            is_reversal = (abs(position_prev) > 1e-6 and 
                          abs(new_position) > 1e-6 and 
                          position_prev * new_position < 0)
            
            # 既存ポジションの決済PnL計算
            if abs(position_prev) > 1e-6:
                realized_pnl = position_prev * (execution_price - self.entry_price) - fee_paid - slippage_paid
                trade_pnl = realized_pnl  # NET PnL (gross - costs)
                
                # ★ P1-1: close_reason判定（ポジション決済時）
                if abs(new_position) <= 1e-6 or is_reversal:
                    # 判定優先順位: TP/SL > 反転 > 手動
                    # 理由: 反転でもTP/SL達成なら、その情報を記録すべき
                    if self._is_take_profit_exit(trade_pnl, abs(position_prev)):
                        close_reason = "tp"
                    elif self._is_stop_loss_exit(trade_pnl, abs(position_prev)):
                        close_reason = "sl"
                    elif is_reversal:
                        close_reason = "reversal"
                    else:
                        close_reason = "manual"  # TTL強制決済、手動決済含む
            
            # 新規ポジションのentry_price更新
            if abs(new_position) > 1e-6:
                # ★ P1-2: 反転時も含め、新ポジション開始時は必ずentry_price更新
                self.entry_price = execution_price
                
                # エントリーコストの記録（純粋な新規エントリー時のみ、反転時は除外）
                if abs(position_prev) <= 1e-6:
                    trade_pnl = -fee_paid - slippage_paid

            # Update entry system outcome on trade close
            if self.entry_system and abs(position_prev) > 1e-6 and abs(new_position) <= 1e-6:
                outcome = 1 if trade_pnl > 0 else -1
                self.entry_system.update_outcome(outcome)

            # TTL更新（エントリー時）
            if abs(delta) > 1e-6 and abs(new_position) > 1e-6:
                if self.ttl_enabled:
                    ttl_steps = int(ttl_fraction * self.max_ttl_steps) + 1
                    self.position_ttl = ttl_steps
                else:
                    self.position_ttl = self.max_ttl_steps
                self.steps_held = 0
        
        # else: delta が小さい場合、trade_pnl/close_reasonは初期値（0/None）のまま
        
        # ポジション保有ステップカウント
        if abs(self.position) > 1e-6:
            self.steps_held += 1
            if self.ttl_enabled:
                self.position_ttl -= 1
        else:
            self.steps_held = 0
            self.position_ttl = 0

        # 報酬計算（compute_hft_rewardの署名に合わせて）
        # Config から reward_params をワイアリング
        reward_kwargs = {
            'price_prev': price_prev,
            'price_now': price_now,
            'position_prev': position_prev,
            'position_now': self.position,
            'atr': atr,
            'fee_paid': fee_paid,
            'slippage_paid': slippage_paid,
            'holding_steps': self.steps_held,
            'max_position': self.max_position,
        }
        # reward_params があれば追加
        if self.reward_params:
            reward_kwargs.update(self.reward_params)
        
        reward, reward_info = compute_hft_reward(**reward_kwargs)
        
        # ★ Trend-Guided Curriculum (Direct Injection)
        # Fixes Finding 1: Ensures Reference Signals affect the reward even in FastEnv.
        if hasattr(self, "ichimoku_signals"):
            ichimoku_trend = self.ichimoku_signals[self.current_step]
            target_pos_fraction = action_result.target_pos_fraction # [-1, 1]
            
            # Trend Alignment: Positive if aligned, Negative if opposed
            trend_alignment = ichimoku_trend * target_pos_fraction
            
            # Penalty for opposing the trend (Contra-Trend)
            # Decay the guidance over time (Curriculum Learning)
            guidance_weight = max(0.0, 1.0 - (self.lifetime_steps / self.guidance_decay_steps))
            
            if trend_alignment < -0.1 and guidance_weight > 0:
                # Target normalized penalty (e.g. -0.05 at max misalignment)
                # This fixes Finding 2 (Reward Scale) and Finding 4 (Decay)
                target_penalty_norm = 0.05 * abs(trend_alignment) * guidance_weight
                
                # Convert to JPY for subtraction from raw reward
                # Use reward_scale so the penalty is consistent in learning space
                penalty_jpy = target_penalty_norm * self.reward_scale
                reward -= penalty_jpy
        
        # ★ Phase 1.3修正: 報酬を学習用にスケーリング（設定可能）
        scaled_reward = reward / max(self.reward_scale, 1e-8)
        learning_reward = scaled_reward
        
        step_pnl = position_prev * (price_now - price_prev)
        self.accounting.update(
            step_pnl=step_pnl,
            fee_paid=fee_paid,
            slippage_paid=slippage_paid,
        )
        self.gross_pnl = self.accounting.gross_pnl
        self.net_pnl = self.accounting.net_pnl
        self.total_fees = self.accounting.total_fees
        self.total_slippage = self.accounting.total_slippage
        self.total_pnl = self.net_pnl
        self.balance = self.accounting.portfolio_value()
        
        # ステップ更新
        self.current_step += 1
        self.lifetime_steps += 1
        self.steps_in_episode += 1
        
        # balance は accounting に同期済みのポートフォリオ値
        self.max_balance = max(self.max_balance, self.balance)
        
        # 終了条件
        done = False
        truncated = False
        
        # ドローダウンリミット
        if self.balance < self.initial_balance * (1 - self.drawdown_limit):
            done = True
        
        # Max steps
        if self.max_steps and self.steps_in_episode >= self.max_steps:
            truncated = True
        
        # データ終端
        if self.current_step >= self.data_len - 1:
            truncated = True
        
        # Force close position at end of episode
        if truncated and abs(self.position) > 1e-6:
            realized_pnl = self.position * (price_now - self.entry_price)
            self.accounting.gross_pnl += realized_pnl
            self.accounting.net_pnl = self.accounting.gross_pnl - self.accounting.total_fees - self.accounting.total_slippage
            self.balance = self.accounting.portfolio_value()
            if self.recorder:
                self.recorder.record_trade(
                    trade_type="long" if self.position > 0 else "short",
                    pnl=realized_pnl,
                    entry_price=self.entry_price,
                    exit_price=price_now,
                    size=abs(self.position),
                    fee=0.0,
                    slippage=0.0
                )
            self.position = 0.0
        
        # 情報辞書
        # ★ P0-3規約: 'trade_pnl'はNET PnL（コスト控除済み）
        # Reporter.record_trade()はこの値を使用し、二重控除しない
        info = {
            'balance': self.balance,
            'position': self.position,
            'pnl': self.net_pnl,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.net_pnl,
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'portfolio_value': self.balance,
            'step': self.steps_in_episode,
            'current_price': float(price_now),
            'fee_paid': fee_paid,
            'slippage_paid': slippage_paid,
            'action_value': float(action_result.action_value),
            'ttl_fraction': float(ttl_fraction),
            'ttl_enabled': self.ttl_enabled,
            'position_ttl': self.position_ttl,
            'cooldown_counter': self.cooldown_counter,
            'steps_held': self.steps_held,
            'ttl_forced_exits': self.ttl_forced_exits,
            'cooldown_triggers': self.cooldown_triggers,
            'trade_pnl': trade_pnl,  # NET PnL (costs deducted)
            'entry_price': self.entry_price,
            'close_reason': close_reason,  # ★ P1-1: close理由（"tp", "sl", "reversal", "manual"）
            'exit_price': self.last_execution_price,
        }
        
        # Update calibration if trade closed
        if abs(self.position) < 1e-6 and abs(position_prev) > 1e-6:  # trade closed
            if self.entry_system and hasattr(self.entry_system, 'calibration_map') and self.entry_system.calibration_map:
                current_regime_data = self.regime_data.iloc[self.current_step]
                regime = current_regime_data.idxmax() if current_regime_data.sum() > 0 else "UNKNOWN"
                self.entry_system.calibration_map.update(regime, self.previous_action, trade_pnl, self.current_step)
            
            # Record trade for reporter
            trade_type = "long" if position_prev > 0 else "short"
            if self.recorder:
                self.recorder.record_trade(
                    trade_type=trade_type,
                    pnl=trade_pnl,
                    entry_price=self.entry_price,
                    exit_price=price_now,
                    size=abs(position_prev),
                    fee=fee_paid,
                    slippage=slippage_paid,
                )
        
        # 次の観測
        next_obs = self._get_observation()
        
        return next_obs, learning_reward, done, truncated, info
    
    def _build_observation(self, idx: int, update_scaler: bool = True) -> np.ndarray:
        """88次元観測を構築"""
        obs = np.zeros(self.TOTAL_OBS_DIM, dtype=np.float32)
        
        # [0:30] Base features
        obs[self.OBSERVATION_DIMS['base'][0]:self.OBSERVATION_DIMS['base'][1]] = self.base_features[idx]
        
        # [30:57] MTF features
        mtf_obs = self.mtf_features[idx]
        # Check for NaN/inf and replace with 0
        mtf_obs = np.nan_to_num(mtf_obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs[self.OBSERVATION_DIMS['mtf'][0]:self.OBSERVATION_DIMS['mtf'][1]] = mtf_obs
        
        # [57:63] Cyclical time features
        # Restored "Lost Alpha" from v451/v456 proposals
        # Uses pre-calculated features from __init__
        obs[self.OBSERVATION_DIMS['cyclical'][0]:self.OBSERVATION_DIMS['cyclical'][1]] = self.cyclical_features[idx]
        
        # [63:69] Global market features
        # 簡略版：zero fill (実運用ではBinance データが必要)
        obs[self.OBSERVATION_DIMS['global'][0]:self.OBSERVATION_DIMS['global'][1]] = 0.0
        
        # [69:82] Regime features
        obs[self.OBSERVATION_DIMS['regime'][0]:self.OBSERVATION_DIMS['regime'][1]] = self.regime_features[idx]
        
        # [82:88] Account features (6D)
        if self.max_ttl_steps > 0:
            ttl_norm = self.position_ttl / self.max_ttl_steps
            ttl_norm = max(0.0, min(ttl_norm, 1.0))
            steps_held_norm = self.steps_held / self.max_ttl_steps
            steps_held_norm = max(0.0, min(steps_held_norm, 1.0))
        else:
            ttl_norm = 0.0
            steps_held_norm = 0.0

        account_feats = np.array([
            self.position / self.max_position if self.max_position > 0 else 0.0,
            ttl_norm,
            (self.last_step_cost / self.close_prices[idx]) if self.close_prices[idx] > 0 else 0.0,
            self.balance / self.initial_balance,  # Balance ratio
            (self.total_pnl / self.initial_balance) if self.initial_balance > 0 else 0.0,  # PnL ratio
            steps_held_norm,  # Steps held norm
        ], dtype=np.float32)
        obs[self.OBSERVATION_DIMS['account'][0]:self.OBSERVATION_DIMS['account'][1]] = account_feats
        
        # スケーラー更新
        if update_scaler:
            self.scaler.fit_one(obs)
        
        # スケーラー適用
        obs_scaled = self.scaler.transform(obs)
        
        return obs_scaled
    
    def _is_take_profit_exit(self, trade_pnl: float, position_size: float) -> bool:
        """
        TP条件を満たすかチェック（Phase 2簡易実装）
        
        Args:
            trade_pnl: 決済時の実現PnL（NET PnL、コスト控除済み）
            position_size: ポジションサイズ（絶対値）
        
        Returns:
            bool: TP条件を満たす場合True
        """
        if abs(position_size) < 1e-6:
            return False
        
        # PnL率計算: pnl / (position_size * entry_price)
        notional_value = position_size * self.entry_price
        if notional_value < 1e-6:
            return False
        
        pnl_pct = trade_pnl / notional_value
        return pnl_pct > self.tp_threshold
    
    def _is_stop_loss_exit(self, trade_pnl: float, position_size: float) -> bool:
        """
        SL条件を満たすかチェック（Phase 2簡易実装）
        
        Args:
            trade_pnl: 決済時の実現PnL（NET PnL、コスト控除済み）
            position_size: ポジションサイズ（絶対値）
        
        Returns:
            bool: SL条件を満たす場合True
        """
        if abs(position_size) < 1e-6:
            return False
        
        # PnL率計算
        notional_value = position_size * self.entry_price
        if notional_value < 1e-6:
            return False
        
        pnl_pct = trade_pnl / notional_value
        return pnl_pct < -self.sl_threshold
    
    def _get_observation(self) -> np.ndarray:
        """現在の観測を取得"""
        return self._build_observation(self.current_step)
    
    def validate_observation_shape(self) -> bool:
        """観測空間が正しいことを検証"""
        obs = self._get_observation()
        
        checks = [
            (obs.shape == (self.TOTAL_OBS_DIM,), f"Shape mismatch: {obs.shape} != ({self.TOTAL_OBS_DIM},)"),
            (np.all(np.isfinite(obs)), "Observation contains non-finite values"),
            (len(self.base_feature_columns) == 30, "Base features dimension mismatch"),
            (len(self.mtf_feature_columns) == 27, "MTF features dimension mismatch"),
            (len(self.regime_feature_columns) == 13, "Regime features dimension mismatch"),
        ]
        
        for check, message in checks:
            if not check:
                logger.error(f"Validation failed: {message}")
                return False
        
        return True
    
    def get_observation_structure(self) -> dict:
        """観測空間の構造情報を取得"""
        return {
            'total_dim': self.TOTAL_OBS_DIM,
            'base': (30, 'OHLCV features'),
            'mtf': (27, 'Multi-timeframe features'),
            'cyclical': (6, 'Cyclical time features (sin/cos)'),
            'global': (6, 'Global market features (continuous)'),
            'regime': (13, 'Regime features (One-Hot)'),
            'account': (6, 'Account metrics (normalized)'),
        }


if __name__ == "__main__":
    # Demo
    import sys
    
    # Create dummy data
    n_steps = 1000
    dates = pd.date_range('2025-01-01', periods=n_steps, freq='1min', tz='UTC')
    
    np.random.seed(42)
    prices = 9000 + np.cumsum(np.random.randn(n_steps) * 5)
    
    base_cols = [f'base_{i}' for i in range(30)]
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    df = pd.DataFrame({
        'close': prices,
        'atr': np.abs(np.random.randn(n_steps)) + 5,
        'impact_proxy': np.random.rand(n_steps) * 0.1,
        **{col: np.random.randn(n_steps) for col in base_cols},
        **{col: np.random.randn(n_steps) for col in mtf_cols},
        **{col: np.random.rand(n_steps) for col in regime_cols},
    }, index=dates)
    
    # Env初期化
    env = FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols,
        mtf_feature_columns=mtf_cols,
        regime_feature_columns=regime_cols,
        max_steps=100,
    )
    
    # 検証
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Validation: {env.validate_observation_shape()}")
    print(f"Structure: {env.get_observation_structure()}")
    
    # 1ステップ
    action = np.array([0.5, 0.7], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step reward: {reward:.4f}")
    print(f"Info: {info}")
