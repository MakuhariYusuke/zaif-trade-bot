"""366# M2: Bayesian Regime Filter (Hamilton 1989 拡張).

Hamilton (1989) Markov-Switching Model のオンラインベイズフィルタリング。
FillTestRegimeDetector の閾値ベース分類を補完し、
4 状態の **事後確率ベクトル** を提供する。

利点
----
- ヒステリシスの上位互換: ベイズ filtering 自体が状態遷移の
  smooth な確率推移を実現する
- 確率的 offset 混合: 離散的なレジーム切替の代わりに
  ``Σ P(s) · offset_mult(s)`` で連続的な offset 乗数を計算可能
- 遷移予測: 遷移行列 A から「次の状態」の予測分布を提供

モデル
------
- **状態**: S = {trending_up, trending_down, ranging, high_vol} (4 状態)
- **観測**: y_t = (mid_price_t - mid_price_{t-1}) / mid_price_{t-1} (return)
- **Emission**: f(y_t | s_t = j) = N(y_t; μ_j, σ_j) — 各状態固有の正規分布
- **遷移行列**: A[i,j] = P(s_t = j | s_{t-1} = i) — 対角優勢 (sticky)

更新式 (Hamilton filter):
    π_pred[j] = Σ_i A[i,j] · π_prev[i]
    π_post[j] ∝ f(y_t | s_t=j) · π_pred[j]

計算量: O(S²) per update = O(16) = O(1)

既存基盤
--------
- ``WelfordOnlineVar`` (T5): O(1) σ 追跡 → emission σ のオンライン更新
- ``FillTestRegimeDetector._classify()``: 閾値ベースのフォールバック
- ``v444_regime_analyzer._calculate_transition_matrix()``: A 行列計算ロジック
- ``VolatilityRegimeClassifier`` (M3): σ-Clustering との連携

References
----------
Hamilton, J. D. (1989). "A New Approach to the Economic Analysis
of Nonstationary Time Series and the Business Cycle."
Econometrica, 57(2), 357-384.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BayesianRegimeFilter",
    "BayesianRegimeConfig",
    "BayesianRegimeResult",
    "RegimeState",
]


# =====================================================================
# 定数・状態
# =====================================================================

class RegimeState(IntEnum):
    """4 状態レジーム (Hamilton model).

    IntEnum で numpy indexing に直接使用可能。
    """

    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    HIGH_VOL = 3


_N_STATES: Final[int] = len(RegimeState)

# FillTestRegime 文字列との相互変換マップ
_STATE_TO_REGIME_STR: Final[dict[RegimeState, str]] = {
    RegimeState.TRENDING_UP: "trending_up",
    RegimeState.TRENDING_DOWN: "trending_down",
    RegimeState.RANGING: "ranging",
    RegimeState.HIGH_VOL: "high_vol",
}

_REGIME_STR_TO_STATE: Final[dict[str, RegimeState]] = {
    v: k for k, v in _STATE_TO_REGIME_STR.items()
}


# =====================================================================
# 設定
# =====================================================================

@dataclass
class EmissionParams:
    """各状態の Emission (正規分布) パラメータ.

    μ: 平均 return
    σ: return の標準偏差 (> 0)
    """

    mu: float = 0.0
    sigma: float = 1e-4

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


@dataclass
class BayesianRegimeConfig:
    """Bayesian Regime Filter の設定.

    Attributes:
        emission_params: 各状態の emission パラメータ (μ, σ)
        transition_stickiness: 対角要素の割合 (0.5-0.99), 高いほど状態が持続
        prior: 初期事前分布 (None → 一様分布)
        emission_floor: emission probability の下限 (numerical stability)
        adaptive_emission: True なら observation から emission パラメータを
            オンライン更新 (Exponential smoothing)
        emission_lr: adaptive emission の学習率 (0.0-1.0)
        reestimate_interval: A 行列再推定の最小更新間隔 (observations 数)
        offset_multipliers: 各状態の offset 乗数 (確率加重で混合)
    """

    emission_params: dict[RegimeState, EmissionParams] = field(
        default_factory=lambda: {
            # Zaif BTC_JPY: 120 秒サイクルの典型的な return 分布
            # trending: μ ≈ ±0.05% (5bps/cycle), σ ≈ 0.03%
            # ranging: μ ≈ 0, σ ≈ 0.01%
            # high_vol: μ ≈ 0, σ ≈ 0.10%
            RegimeState.TRENDING_UP: EmissionParams(mu=5e-4, sigma=3e-4),
            RegimeState.TRENDING_DOWN: EmissionParams(mu=-5e-4, sigma=3e-4),
            RegimeState.RANGING: EmissionParams(mu=0.0, sigma=1e-4),
            RegimeState.HIGH_VOL: EmissionParams(mu=0.0, sigma=1e-3),
        }
    )
    transition_stickiness: float = 0.90
    prior: np.ndarray | None = None
    emission_floor: float = 1e-300
    adaptive_emission: bool = True
    emission_lr: float = 0.01
    reestimate_interval: int = 100
    offset_multipliers: dict[RegimeState, float] = field(
        default_factory=lambda: {
            RegimeState.TRENDING_UP: 1.0,
            RegimeState.TRENDING_DOWN: 1.0,
            RegimeState.RANGING: 0.8,
            RegimeState.HIGH_VOL: 1.5,
        }
    )


# =====================================================================
# 結果
# =====================================================================

@dataclass
class BayesianRegimeResult:
    """Bayesian filter の更新結果.

    Attributes:
        posterior: 事後確率ベクトル shape=(4,)
        map_state: Maximum A Posteriori (最尤状態)
        map_probability: MAP 状態の確率
        predicted: 1-step 予測確率 (次サイクルの事前分布)
        offset_multiplier: 確率加重 offset 乗数
        observation: 入力された return 値
    """

    posterior: np.ndarray
    map_state: RegimeState
    map_probability: float
    predicted: np.ndarray
    offset_multiplier: float
    observation: float

    @property
    def regime_probabilities(self) -> dict[str, float]:
        """FillTestRegime 互換の確率辞書."""
        return {
            _STATE_TO_REGIME_STR[state]: float(self.posterior[state])
            for state in RegimeState
        }

    @property
    def map_regime_str(self) -> str:
        """MAP 状態の FillTestRegime 文字列."""
        return _STATE_TO_REGIME_STR[self.map_state]

    def to_dict(self) -> dict[str, object]:
        """JSON serializable dict."""
        return {
            "map_regime": self.map_regime_str,
            "map_probability": round(self.map_probability, 4),
            "offset_multiplier": round(self.offset_multiplier, 4),
            "probabilities": {
                k: round(v, 4) for k, v in self.regime_probabilities.items()
            },
            "observation": round(self.observation, 8),
        }


# =====================================================================
# Bayesian Regime Filter
# =====================================================================

class BayesianRegimeFilter:
    """Hamilton (1989) Markov-Switching Online Bayesian Filter.

    Usage:
        filter = BayesianRegimeFilter()
        result = filter.update(return_value)
        print(result.regime_probabilities)
        print(result.offset_multiplier)  # 確率加重 offset 乗数
    """

    def __init__(self, config: BayesianRegimeConfig | None = None) -> None:
        self._config = config or BayesianRegimeConfig()
        self._n_states = _N_STATES

        # 遷移行列 A[i,j] = P(s_t=j | s_{t-1}=i)
        self._transition: np.ndarray = self._build_transition_matrix(
            self._config.transition_stickiness
        )

        # Emission パラメータ (numpy array, 高速化用)
        self._emission_mu = np.array(
            [self._config.emission_params[s].mu for s in RegimeState],
            dtype=np.float64,
        )
        self._emission_sigma = np.array(
            [self._config.emission_params[s].sigma for s in RegimeState],
            dtype=np.float64,
        )

        # 事後確率 (初期 = 一様 or 指定)
        if self._config.prior is not None:
            self._posterior = np.array(self._config.prior, dtype=np.float64).copy()
            self._posterior /= self._posterior.sum()
        else:
            self._posterior = np.ones(self._n_states, dtype=np.float64) / self._n_states

        # Offset 乗数
        self._offset_mults = np.array(
            [self._config.offset_multipliers[s] for s in RegimeState],
            dtype=np.float64,
        )

        # 統計
        self._update_count: int = 0
        # C1 fix: deque で上限を設け無限メモリ増加を防止
        _max_history = max(self._config.reestimate_interval * 3, 300)
        self._regime_history: deque[int] = deque(maxlen=_max_history)

    # -----------------------------------------------------------------
    # 遷移行列
    # -----------------------------------------------------------------

    def _build_transition_matrix(self, stickiness: float) -> np.ndarray:
        """対角優勢の遷移行列を構築.

        Args:
            stickiness: 対角要素の割合 (0.5-0.99)

        Returns:
            shape=(4,4) の遷移確率行列 (行和=1)
        """
        stickiness = max(0.5, min(0.99, stickiness))
        off_diag = (1.0 - stickiness) / (self._n_states - 1)
        A = np.full(
            (self._n_states, self._n_states), off_diag, dtype=np.float64
        )
        np.fill_diagonal(A, stickiness)
        return A

    # -----------------------------------------------------------------
    # Emission (Gaussian likelihood)
    # -----------------------------------------------------------------

    def _emission_likelihood(self, y: float) -> np.ndarray:
        """各状態の emission probability を計算.

        f(y | s=j) = N(y; μ_j, σ_j)

        Args:
            y: 観測値 (return)

        Returns:
            shape=(4,) の likelihood ベクトル
        """
        z = (y - self._emission_mu) / self._emission_sigma
        log_lik = -0.5 * z * z - np.log(self._emission_sigma) - 0.5 * np.log(2.0 * np.pi)
        lik = np.exp(log_lik)
        # Floor で numerical underflow 防止
        np.maximum(lik, self._config.emission_floor, out=lik)
        return lik

    # -----------------------------------------------------------------
    # Core: Bayesian update
    # -----------------------------------------------------------------

    def update(self, observation: float) -> BayesianRegimeResult:
        """新しい observation (return) でフィルタを更新.

        Hamilton filter:
            1. 予測: π_pred = A^T · π_post_prev
            2. 尤度: lik[j] = N(y; μ_j, σ_j)
            3. 更新: π_post[j] ∝ lik[j] · π_pred[j]

        Args:
            observation: return 値 = (price_t - price_{t-1}) / price_{t-1}

        Returns:
            BayesianRegimeResult with posterior probabilities.
        """
        if not math.isfinite(observation):
            # NaN/Inf → 事後確率を変更せず返す
            return self._make_result(observation)

        # Step 1: 予測 (one-step prediction)
        predicted = self._transition.T @ self._posterior

        # Step 2: Emission likelihood
        likelihood = self._emission_likelihood(observation)

        # Step 3: ベイズ更新
        unnorm = likelihood * predicted
        total = unnorm.sum()
        if total > 0:
            self._posterior = unnorm / total
        else:
            # 完全な underflow → 一様分布にリセット
            self._posterior = np.ones(self._n_states, dtype=np.float64) / self._n_states
            logger.warning("[BayesianRegime] posterior underflow — reset to uniform")

        self._update_count += 1

        # MAP 状態を履歴に追加 (A 行列再推定用)
        map_idx = int(np.argmax(self._posterior))
        self._regime_history.append(map_idx)

        # Adaptive emission (emission パラメータのオンライン更新)
        if self._config.adaptive_emission:
            self._adapt_emission(observation)

        # 定期的な A 行列再推定
        if (
            self._config.reestimate_interval > 0
            and self._update_count % self._config.reestimate_interval == 0
            and len(self._regime_history) >= self._config.reestimate_interval
        ):
            self._reestimate_transition()

        # M1 fix: predicted を再利用して二重計算を回避
        return self._make_result(observation, predicted=predicted)

    # -----------------------------------------------------------------
    # 結果生成
    # -----------------------------------------------------------------

    def _make_result(
        self, observation: float, *, predicted: np.ndarray | None = None,
    ) -> BayesianRegimeResult:
        """現在の posterior から BayesianRegimeResult を生成."""
        map_idx = int(np.argmax(self._posterior))
        if predicted is None:
            predicted = self._transition.T @ self._posterior
        offset_mult = float(np.dot(self._posterior, self._offset_mults))
        return BayesianRegimeResult(
            posterior=self._posterior.copy(),
            map_state=RegimeState(map_idx),
            map_probability=float(self._posterior[map_idx]),
            predicted=predicted.copy(),
            offset_multiplier=offset_mult,
            observation=observation,
        )

    # -----------------------------------------------------------------
    # Adaptive emission
    # -----------------------------------------------------------------

    def _adapt_emission(self, y: float) -> None:
        """Exponential smoothing で emission パラメータをオンライン更新.

        MAP 状態の μ, σ を observation に向けて微調整する。
        他状態は変更しない (MAP に基づく soft assignment)。
        """
        lr = self._config.emission_lr
        for j in range(self._n_states):
            resp = float(self._posterior[j])  # responsibility
            if resp < 0.01:
                continue
            eff_lr = lr * resp  # responsibility-weighted learning rate
            # μ の更新
            self._emission_mu[j] += eff_lr * (y - self._emission_mu[j])
            # σ の更新 (exponential smoothing of |y - μ|)
            new_dev = abs(y - self._emission_mu[j])
            self._emission_sigma[j] += eff_lr * (
                new_dev - self._emission_sigma[j]
            )
            # σ の下限
            self._emission_sigma[j] = max(self._emission_sigma[j], 1e-8)

    # -----------------------------------------------------------------
    # A 行列再推定
    # -----------------------------------------------------------------

    def _reestimate_transition(self) -> None:
        """regime_history から遷移行列を再推定.

        v444_regime_analyzer._calculate_transition_matrix() のロジックを
        numpy 化して再利用。Laplace smoothing (α=1) で zero 遷移を防止。
        """
        history = self._regime_history
        n = len(history)
        if n < 10:
            return

        # 遷移カウント
        counts = np.ones(
            (self._n_states, self._n_states), dtype=np.float64
        )  # Laplace smoothing α=1
        for i in range(n - 1):
            counts[history[i], history[i + 1]] += 1

        # 行和で正規化
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        new_A = counts / row_sums

        # 急激な変化を避けるため、古い A と指数平均
        alpha = 0.3
        self._transition = (1 - alpha) * self._transition + alpha * new_A

        # 行和の正規化 (数値安定性)
        row_sums2 = self._transition.sum(axis=1, keepdims=True)
        self._transition /= np.maximum(row_sums2, 1e-12)

        logger.debug(
            "[BayesianRegime] transition matrix reestimated "
            f"(n={n}, diag={np.diag(self._transition).tolist()})"
        )

    # -----------------------------------------------------------------
    # プロパティ
    # -----------------------------------------------------------------

    @property
    def posterior(self) -> np.ndarray:
        """現在の事後確率ベクトル (read-only copy)."""
        return self._posterior.copy()

    @property
    def transition_matrix(self) -> np.ndarray:
        """現在の遷移行列 (read-only copy)."""
        return self._transition.copy()

    @property
    def emission_mu(self) -> np.ndarray:
        """現在の emission μ ベクトル (read-only copy)."""
        return self._emission_mu.copy()

    @property
    def emission_sigma(self) -> np.ndarray:
        """現在の emission σ ベクトル (read-only copy)."""
        return self._emission_sigma.copy()

    @property
    def update_count(self) -> int:
        """累計更新回数."""
        return self._update_count

    @property
    def map_state(self) -> RegimeState:
        """現在の MAP 状態."""
        return RegimeState(int(np.argmax(self._posterior)))

    # -----------------------------------------------------------------
    # 確率加重ユーティリティ
    # -----------------------------------------------------------------

    def weighted_value(self, values: dict[RegimeState, float]) -> float:
        """事後確率で加重平均した値を計算.

        Args:
            values: 各状態の値 (例: offset 乗数, boost 係数)

        Returns:
            Σ P(s) · value(s)
        """
        result = 0.0
        for state in RegimeState:
            result += float(self._posterior[state]) * values.get(state, 0.0)
        return result

    # -----------------------------------------------------------------
    # State persistence (121# A4 互換)
    # -----------------------------------------------------------------

    def get_state(self) -> dict:
        """永続化用の状態辞書."""
        return {
            "posterior": self._posterior.tolist(),
            "transition": self._transition.tolist(),
            "emission_mu": self._emission_mu.tolist(),
            "emission_sigma": self._emission_sigma.tolist(),
            "update_count": self._update_count,
            "regime_history_tail": self._regime_history[-200:]
            if len(self._regime_history) > 200
            else list(self._regime_history),
        }

    def restore_state(self, state: dict) -> bool:
        """永続化された状態から復元."""
        try:
            post = np.array(state["posterior"], dtype=np.float64)
            if post.shape != (self._n_states,):
                return False
            # H2 fix: 全要素0の場合は復元失敗
            if post.sum() <= 0:
                return False
            self._posterior = post / post.sum()

            trans = np.array(state["transition"], dtype=np.float64)
            if trans.shape != (self._n_states, self._n_states):
                return False
            self._transition = trans

            # H1 fix: emission 配列の shape 検証
            mu = np.array(state["emission_mu"], dtype=np.float64)
            sigma = np.array(state["emission_sigma"], dtype=np.float64)
            if mu.shape != (self._n_states,) or sigma.shape != (self._n_states,):
                return False
            self._emission_mu = mu
            self._emission_sigma = sigma
            self._update_count = int(state.get("update_count", 0))
            tail = list(state.get("regime_history_tail", []))
            _max_history = max(self._config.reestimate_interval * 3, 300)
            self._regime_history = deque(tail, maxlen=_max_history)

            logger.info(
                f"[BayesianRegime] state restored: "
                f"update_count={self._update_count}, "
                f"map={self.map_state.name}"
            )
            return True
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[BayesianRegime] state restore failed: {e}")
            return False

    def reset(self) -> None:
        """内部状態をリセット (一様事前に戻す)."""
        if self._config.prior is not None:
            self._posterior = np.array(self._config.prior, dtype=np.float64).copy()
            self._posterior /= self._posterior.sum()
        else:
            self._posterior = np.ones(self._n_states, dtype=np.float64) / self._n_states
        self._transition = self._build_transition_matrix(
            self._config.transition_stickiness
        )
        self._emission_mu = np.array(
            [self._config.emission_params[s].mu for s in RegimeState],
            dtype=np.float64,
        )
        self._emission_sigma = np.array(
            [self._config.emission_params[s].sigma for s in RegimeState],
            dtype=np.float64,
        )
        self._update_count = 0
        _max_history = max(self._config.reestimate_interval * 3, 300)
        self._regime_history = deque(maxlen=_max_history)
