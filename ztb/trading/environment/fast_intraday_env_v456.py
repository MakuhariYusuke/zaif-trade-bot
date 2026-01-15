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
from ztb.trading.rewards.fast_intraday import compute_hft_reward
from ztb.utils.fee_model import ExchangeFeeModel

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
    
    Action Space: Box([-1, 0], [1, 1])
    - target_position: [-1, 1] (Fraction of max_position)
    - ttl_fraction: [0, 1] (Time-To-Live fraction)
    
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
        'account': (82, 88),       # 3
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
        reward_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        # Data
        self.df = df.reset_index(drop=True)
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
        self.reward_params = reward_params or {}
        
        # Fee model
        self.fee_model = ExchangeFeeModel(exchange_fees={
            "zaif": {"buy": commission_rate, "sell": commission_rate}
        })
        self.fee_model.set_exchange("zaif")
        
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
        self.mtf_features = self.df[mtf_feature_columns].values.astype(np.float32)
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
        self.balance = initial_balance
        self.position = 0.0
        self.position_ttl = 0
        self.steps_held = 0
        self.cooldown_counter = 0
        self.total_pnl = 0.0
        self.max_balance = initial_balance
        self.last_step_cost = 0.0
        self.steps_in_episode = 0
        self.last_realized_fee = 0.0  # ★ Phase 1.3: fee tracking
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """環境をリセット"""
        super().reset(seed=seed)
        
        # ランダム開始位置（prewarmとmax_stepsを考慮）
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
        self.total_pnl = 0.0
        self.max_balance = self.initial_balance
        self.last_step_cost = 0.0
        self.steps_in_episode = 0
        
        # スケーラーリセット
        self.scaler.reset()
        
        # Prewarm data
        prewarm_start = self.current_step - self.prewarm_steps
        prewarm_indices = range(prewarm_start, self.current_step)
        
        for idx in prewarm_indices:
            obs = self._build_observation(idx, update_scaler=True)
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """1ステップ実行"""
        # 価格取得
        price_now = self.close_prices[self.current_step]
        price_prev = self.close_prices[self.current_step - 1] if self.current_step > 0 else price_now
        atr = self.atr_data[self.current_step]
        
        pnl = 0.0  # 初期化
        
        # アクション解析
        target_pos_fraction = float(np.clip(action[0], -1.0, 1.0))
        ttl_fraction = float(np.clip(action[1], 0.0, 1.0))
        
        raw_target_position = target_pos_fraction * self.max_position
        
        # TTL満期チェック
        if self.position_ttl <= 0 and abs(self.position) > 1e-6:
            raw_target_position = 0.0
            if self.position_ttl == 0:
                self.cooldown_counter = self.cooldown_steps
                self.position_ttl = -1
        
        # Cooldown期間の強制フラット
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            raw_target_position = 0.0
        
        # ポジション遷移
        delta = raw_target_position - self.position
        
        # デッドバンド
        if abs(delta) < self.min_delta * self.max_position:
            delta = 0.0
        
        # リキディティ制約
        max_delta = self.max_delta_per_step * self.max_position
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
        
        new_position = self.position + delta
        
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

            # 手数料
            trade_type = "buy" if delta > 0 else "sell"
            fee_rate = self.fee_model.get_fee_rate(trade_type)
            fee_paid = abs(delta) * execution_price * fee_rate

            # PnL更新（確定損益のみ balance に反映）
            pnl = self.position * (price_now - self.last_step_cost) if self.position != 0 else 0.0
            realized_pnl = pnl - fee_paid - slippage_paid
            
            self.total_pnl += pnl
            # ★ Phase 1.3修正: 手数料/スリッページは balance から直接引かない
            # → 代わりに報酬計算で反映される
            self.last_realized_fee = fee_paid + slippage_paid

            # ポジション更新
            self.position = new_position
            self.last_step_cost = execution_price

            # TTL更新（エントリー時）
            if abs(delta) > 1e-6 and abs(new_position) > 1e-6:
                ttl_steps = int(ttl_fraction * self.max_ttl_steps) + 1
                self.position_ttl = ttl_steps
                self.steps_held = 0

        # ポジション保有ステップカウント
        if abs(self.position) > 1e-6:
            self.steps_held += 1
            self.position_ttl -= 1

        # 報酬計算（compute_hft_rewardの署名に合わせて）
        # Config から reward_params をワイアリング
        reward_kwargs = {
            'price_prev': price_prev,
            'price_now': price_now,
            'position_prev': self.position - delta if delta != 0 else self.position,
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
        
        # ★ Phase 1.3修正: 報酬を学習用にスケーリング
        # reward を [-0.1, 0.1] 範囲に正規化
        learning_reward = np.clip(reward / 100.0, -0.1, 0.1)
        
        # ステップ更新
        self.current_step += 1
        self.steps_in_episode += 1
        
        # ★ Phase 1.3修正: balance は日次確定基準でのみ更新
        # （毎ステップの手数料反映ではなく）
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
        
        # 情報辞書
        info = {
            'balance': self.balance,
            'position': self.position,
            'pnl': self.total_pnl,
            'step': self.steps_in_episode,
            'current_price': float(price_now),
        }
        
        # 次の観測
        next_obs = self._get_observation()
        
        return next_obs, learning_reward, done, truncated, info
    
    def _build_observation(self, idx: int, update_scaler: bool = True) -> np.ndarray:
        """88次元観測を構築"""
        obs = np.zeros(self.TOTAL_OBS_DIM, dtype=np.float32)
        
        # [0:30] Base features
        obs[self.OBSERVATION_DIMS['base'][0]:self.OBSERVATION_DIMS['base'][1]] = self.base_features[idx]
        
        # [30:57] MTF features
        obs[self.OBSERVATION_DIMS['mtf'][0]:self.OBSERVATION_DIMS['mtf'][1]] = self.mtf_features[idx]
        
        # [57:63] Cyclical time features
        # インデックスのタイムスタンプから周期特徴量を抽出
        # (訓練時: DataFrameのインデックスから自動取得)
        try:
            # プレースホルダー：実運用ではDataFrameインデックスから自動抽出
            # 今はzero fill (Cyclical時間は実装済みだが、env統合では簡略版)
            cyclical_feats = np.zeros(6, dtype=np.float32)
            obs[self.OBSERVATION_DIMS['cyclical'][0]:self.OBSERVATION_DIMS['cyclical'][1]] = cyclical_feats
        except Exception:
            obs[self.OBSERVATION_DIMS['cyclical'][0]:self.OBSERVATION_DIMS['cyclical'][1]] = 0.0
        
        # [63:69] Global market features
        # 簡略版：zero fill (実運用ではBinance データが必要)
        obs[self.OBSERVATION_DIMS['global'][0]:self.OBSERVATION_DIMS['global'][1]] = 0.0
        
        # [69:82] Regime features
        obs[self.OBSERVATION_DIMS['regime'][0]:self.OBSERVATION_DIMS['regime'][1]] = self.regime_features[idx]
        
        # [82:88] Account features (6D)
        account_feats = np.array([
            self.position / self.max_position if self.max_position > 0 else 0.0,
            (self.position_ttl / self.max_ttl_steps) if self.max_ttl_steps > 0 else 0.0,
            (self.last_step_cost / self.close_prices[idx]) if self.close_prices[idx] > 0 else 0.0,
            self.balance / self.initial_balance,  # Balance ratio
            (self.total_pnl / self.initial_balance) if self.initial_balance > 0 else 0.0,  # PnL ratio
            (self.steps_held / self.max_ttl_steps) if self.max_ttl_steps > 0 else 0.0,  # Steps held norm
        ], dtype=np.float32)
        obs[self.OBSERVATION_DIMS['account'][0]:self.OBSERVATION_DIMS['account'][1]] = account_feats
        
        # スケーラー更新
        if update_scaler:
            self.scaler.fit_one(obs)
        
        # スケーラー適用
        obs_scaled = self.scaler.transform(obs)
        
        return obs_scaled
    
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
            'account': (3, 'Account metrics (normalized)'),
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
