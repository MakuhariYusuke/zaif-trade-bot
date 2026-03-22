"""P8: LiteTradingEnv — 連続行動空間の軽量トレーディング環境.

365# §3.3(A) で指摘された「±0.3333 閾値による硬直離散化」を排除し、
SAC Actor の連続出力 [-1, +1] をそのままポジションサイズに変換する。

設計ポリシー:
  - HeavyTradingEnv の市場理論機構 (15種) を全排除
  - action [-1, +1] → target_position = action * max_position_size
  - 報酬 = step PnL (ポジション × 価格変動) - 取引コスト
  - 閾値なし・ActionValidator なし → Ghost action 問題が構造的に発生しない
  - ObservationBuilder / DataManager は flow 再利用 (ただし依存直接は最小化)

参照: 365# §7 P8, §3.3(A), §10.2
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LiteEnvConfig:
    """LiteTradingEnv 用の最小設定.

    HeavyTradingEnv の EnvironmentConfig (80+ fields) とは独立。
    必要最小限のフィールドのみ定義する。
    """

    max_position_size: float = 0.01  # BTC
    initial_portfolio_value: float = 10_000_000.0  # JPY
    transaction_cost_rate: float = 0.001  # 0.1 %
    reward_scaling: float = 1.0
    feature_columns: list[str] | None = None
    random_start: bool = True
    random_start_buffer: int = 100  # 先頭 N 行はスキップ候補
    embed_action_masks: bool = False  # P7 互換 (LiteEnv では常に全合法)
    max_steps_per_episode: int | None = None  # None = データ末尾まで

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LiteEnvConfig":
        """dict → LiteEnvConfig (不明キーは無視)."""
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------


class LiteTradingEnv(gym.Env):
    """連続行動空間 [-1, +1] の軽量トレーディング環境.

    action: float in [-1, +1]
      - +1 → max long  (+max_position_size)
      - -1 → max short (-max_position_size)
      -  0 → flat

    各ステップで target_position = action * max_position_size に向けてポジションを調整し、
    差分に対して取引コストを課す。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        config: LiteEnvConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or LiteEnvConfig()
        self._setup_data(df)
        self._setup_spaces()
        self._reset_state()

    # ------------------------------------------------------------------
    #  Setup helpers
    # ------------------------------------------------------------------

    def _setup_data(self, df: pd.DataFrame) -> None:
        """DataFrame からの価格配列 / 特徴量行列の構築."""
        self.df = df.reset_index(drop=True)

        # 価格配列 (close 優先)
        if "close" in df.columns:
            self._price_array = df["close"].to_numpy(dtype=np.float64)
        elif "price" in df.columns:
            self._price_array = df["price"].to_numpy(dtype=np.float64)
        else:
            raise ValueError("DataFrame must contain 'close' or 'price' column")

        self.n_steps = len(df)

        # 特徴量行列
        feature_cols = self.config.feature_columns
        if feature_cols is None:
            # デフォルト: 価格系列 + volume のみ
            candidates = ["open", "high", "low", "close", "volume"]
            feature_cols = [c for c in candidates if c in df.columns]

        self._feature_columns = feature_cols
        self._feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)

        # NaN → 0 (学習安定性)
        nan_mask = np.isnan(self._feature_matrix)
        if nan_mask.any():
            logger.warning(
                "LiteTradingEnv: %d NaN values replaced with 0 in feature matrix",
                int(nan_mask.sum()),
            )
            self._feature_matrix = np.nan_to_num(self._feature_matrix, nan=0.0)

    def _setup_spaces(self) -> None:
        """Gymnasium の action_space / observation_space を定義."""
        obs_dim = self._feature_matrix.shape[1]
        # P7 互換: embed_action_masks が True でも、LiteEnv では常に全合法なので
        # 定数 [1, 1, 1] を追加するのみ (HeavyEnv 互換の obs_dim 計算)
        if self.config.embed_action_masks:
            obs_dim += 3

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

    def _reset_state(self) -> None:
        """内部状態の初期化."""
        self.current_step = 0
        self.position: float = 0.0  # BTC
        self.portfolio_value: float = self.config.initial_portfolio_value
        self.total_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.trades_count: int = 0
        self._prev_portfolio_value: float = self.config.initial_portfolio_value
        self._episode_start_step: int = 0

    # ------------------------------------------------------------------
    #  Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()

        if self.config.random_start and self.n_steps > self.config.random_start_buffer + 10:
            max_start = min(
                self.config.random_start_buffer,
                self.n_steps // 4,
            )
            self.current_step = self.np_random.integers(0, max(1, max_start))
        else:
            self.current_step = 0

        self._episode_start_step = self.current_step
        obs = self._get_observation()
        return obs, self._get_info()

    def step(
        self, action: np.ndarray | float,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # action のスカラー化
        if isinstance(action, np.ndarray):
            action_val = float(action.flat[0])
        else:
            action_val = float(action)

        # クリップ
        action_val = np.clip(action_val, -1.0, 1.0)

        # 現在価格と次の価格
        current_price = self._price_array[self.current_step]

        # ポジション目標
        target_position = action_val * self.config.max_position_size
        position_delta = target_position - self.position

        # 取引コスト (ポジション変更分にのみ課す)
        trade_cost = abs(position_delta) * current_price * self.config.transaction_cost_rate
        self.total_fees += trade_cost

        if abs(position_delta) > 1e-10:
            self.trades_count += 1

        # ポジション更新
        old_position = self.position
        self.position = target_position

        # ステップ進行
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # max_steps_per_episode による打ち切り
        truncated = False
        if self.config.max_steps_per_episode is not None:
            steps_in_episode = self.current_step - self._episode_start_step
            if steps_in_episode >= self.config.max_steps_per_episode:
                truncated = True

        # step PnL: ポジション × 価格変動 - 取引コスト
        if not done:
            next_price = self._price_array[self.current_step]
            price_change = next_price - current_price
            step_pnl = self.position * price_change - trade_cost
        else:
            step_pnl = -trade_cost  # 最終ステップは価格変動なし

        self.total_pnl += step_pnl
        self.portfolio_value += step_pnl

        # 報酬 (RewardKernel による一元化)
        reward_params = RewardParams(
            reward_scaling=self.config.reward_scaling,
            hold_penalty_multiplier=self.config.hold_penalty_multiplier,
            trade_frequency_bonus=self.config.trade_frequency_bonus,
            bankruptcy_penalty=self.config.bankruptcy_penalty,
        )

        # LiteEnv では連続値を離散アクション(HOLD/BUY/SELL)にマッピングしてボーナスを適用
        # 0.001 程度の微小な動きは HOLD とみなす
        if abs(action_val) < 1e-4:
            action_type = ACTION_HOLD
        elif action_val > 0:
            action_type = ACTION_BUY
        else:
            action_type = ACTION_SELL

        reward = RewardKernel.calculate_basic_reward(
            pnl=step_pnl,
            action=action_type,
            params=reward_params,
            old_position=old_position,
            current_position=self.position,
            portfolio_value=self.portfolio_value,
        )

        # 破産チェック (報酬計算は Kernel 側で処理済み)
        if self.portfolio_value <= 0:
            done = True

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "step_pnl": step_pnl,
            "trade_cost": trade_cost,
            "position_delta": position_delta,
            "old_position": old_position,
            "action_value": action_val,
            "current_price": current_price,
        })

        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    #  Observation / Info
    # ------------------------------------------------------------------

    def _get_observation(self) -> NDArray[np.float32]:
        idx = min(self.current_step, self._feature_matrix.shape[0] - 1)
        obs = self._feature_matrix[idx].copy()

        if self.config.embed_action_masks:
            # LiteEnv: 全アクションが常に合法
            masks = np.ones(3, dtype=np.float32)
            obs = np.concatenate([obs, masks])

        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self.current_step,
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "trades_count": self.trades_count,
        }

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def get_action_masks(self) -> NDArray[np.bool_]:
        """HeavyTradingEnv 互換の action_masks (常に全合法)."""
        return np.ones(3, dtype=np.bool_)

    @property
    def price_at_current_step(self) -> float:
        """現在ステップの価格."""
        idx = min(self.current_step, len(self._price_array) - 1)
        return float(self._price_array[idx])

    def gross_roi(self) -> float:
        """エピソード開始からの ROI."""
        return self.portfolio_value / self.config.initial_portfolio_value - 1.0
