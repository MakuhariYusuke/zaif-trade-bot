"""366# M4: GLFT Fill Probability Model (Guéant-Lehalle-Fernandez-Tapia 2013).

到着率モデル A(δ) = A · exp(-k·δ) を fill_records から推定し、
最適 offset δ* を算出する。

理論
----
GLFT (2013) の fill probability は offset δ に対して指数減衰:
    A(δ) = A · exp(-k·δ)

最適 offset (有限期間 AS):
    δ* = 1/k + q·γ·σ²·τ / k

ここで:
  - A: 基準到着強度 (δ=0 での fill 確率)
  - k: decay 定数 (offset 感度)
  - q: 在庫偏向
  - γ: risk aversion
  - σ: volatility
  - τ: time horizon

既存基盤
--------
- ``_apply_as_reservation_shift()`` の AS δ* が 80% 実装済み
- ``effective_offset_used`` フィールドで offset 実績を記録済み
- ``filled`` フラグで fill/timeout を識別可能

References
----------
Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013).
"Dealing with the inventory risk: a solution to the market making problem."
Mathematics and Financial Economics, 7(4), 477-507.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FillProbabilityModel",
    "FillProbEstimate",
    "estimate_fill_probability_params",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: 最小サンプル数 (これ未満では推定しない)
MIN_SAMPLES: int = 20
#: Offset ビン数 (等分位)
DEFAULT_N_BINS: int = 10
#: k のフォールバック (推定失敗時)
DEFAULT_K: float = 50.0
#: A のフォールバック (推定失敗時)
DEFAULT_A: float = 0.8
#: k の上限 (数値安定性)
MAX_K: float = 500.0
#: k の下限
MIN_K: float = 1.0


@dataclass(frozen=True)
class FillProbEstimate:
    """到着率パラメータ推定結果."""

    A: float  # noqa: N815 — 理論記号
    k: float
    n_samples: int
    r_squared: float
    is_fallback: bool = False


@dataclass
class FillProbabilityModel:
    """GLFT 到着率モデル.

    fill_records から A, k を推定し、offset → fill probability の
    予測と最適 δ* の算出を提供する。
    """

    A: float = DEFAULT_A  # noqa: N815
    k: float = DEFAULT_K
    _estimate: FillProbEstimate | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_fill_prob(self, offset_ratio: float) -> float:
        """offset 比率 → fill probability [0, 1] を予測.

        Parameters
        ----------
        offset_ratio:
            spread_offset_ratio (e.g. 0.10 = spread の 10%)

        Returns
        -------
        float
            推定 fill probability
        """
        if offset_ratio < 0:
            offset_ratio = 0.0
        prob = self.A * math.exp(-self.k * offset_ratio)
        return min(max(prob, 0.0), 1.0)

    def optimal_delta(
        self,
        q: float = 0.0,
        gamma: float = 0.01,
        sigma: float = 0.001,
        tau: float = 60.0,
    ) -> float:
        """GLFT 最適 offset δ* を算出.

        δ* = 1/k + q·γ·σ²·τ / k

        Parameters
        ----------
        q: 在庫偏向 [-1, +1]
        gamma: risk aversion coefficient
        sigma: volatility (fraction)
        tau: time horizon (seconds)

        Returns
        -------
        float
            最適 offset ratio
        """
        if self.k <= 0:
            return 0.0
        base = 1.0 / self.k
        inventory_term = abs(q) * gamma * sigma * sigma * tau / self.k
        return base + inventory_term

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        offsets: np.ndarray,
        filled: np.ndarray,
        *,
        n_bins: int = DEFAULT_N_BINS,
    ) -> FillProbEstimate:
        """fill_records の offset/filled データから A, k を推定.

        Parameters
        ----------
        offsets:
            各注文の effective_offset_used (ratio)
        filled:
            各注文の fill フラグ (bool array)
        n_bins:
            ビン分割数

        Returns
        -------
        FillProbEstimate
        """
        estimate = estimate_fill_probability_params(
            offsets, filled, n_bins=n_bins,
        )
        self.A = estimate.A
        self.k = estimate.k
        self._estimate = estimate
        return estimate

    @property
    def last_estimate(self) -> FillProbEstimate | None:
        """直近の推定結果."""
        return self._estimate


def estimate_fill_probability_params(
    offsets: np.ndarray,
    filled: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> FillProbEstimate:
    """Offset vs fill rate から A, k を OLS 対数回帰で推定.

    log(fill_rate) = log(A) - k·δ → OLS for (log(A), -k)

    Parameters
    ----------
    offsets:
        各注文の offset ratio (float array)
    filled:
        fill フラグ (bool or 0/1 array)
    n_bins:
        等分位ビン数

    Returns
    -------
    FillProbEstimate
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    filled = np.asarray(filled, dtype=bool)

    if len(offsets) != len(filled):
        raise ValueError(
            f"offsets ({len(offsets)}) と filled ({len(filled)}) の長さが不一致"
        )

    # フィルタ: offset が非負のレコードのみ
    valid = offsets >= 0
    offsets = offsets[valid]
    filled = filled[valid]

    n_samples = len(offsets)
    if n_samples < MIN_SAMPLES:
        logger.warning(
            f"サンプル不足 ({n_samples} < {MIN_SAMPLES})、フォールバック値を使用"
        )
        return FillProbEstimate(
            A=DEFAULT_A, k=DEFAULT_K, n_samples=n_samples,
            r_squared=0.0, is_fallback=True,
        )

    # ビン分割: 等分位でオフセットをビン化
    actual_bins = min(n_bins, n_samples // 2)
    if actual_bins < 2:
        return FillProbEstimate(
            A=DEFAULT_A, k=DEFAULT_K, n_samples=n_samples,
            r_squared=0.0, is_fallback=True,
        )

    try:
        bin_edges = np.percentile(
            offsets, np.linspace(0, 100, actual_bins + 1),
        )
        # 重複エッジの除去
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return FillProbEstimate(
                A=DEFAULT_A, k=DEFAULT_K, n_samples=n_samples,
                r_squared=0.0, is_fallback=True,
            )

        bin_indices = np.digitize(offsets, bin_edges[1:-1])
        bin_centers: list[float] = []
        bin_fill_rates: list[float] = []

        for i in range(len(bin_edges) - 1):
            mask = bin_indices == i
            if mask.sum() < 3:  # 最低3サンプル/ビン
                continue
            center = float(np.mean(offsets[mask]))
            rate = float(np.mean(filled[mask]))
            if rate > 0:  # log(0) を避ける
                bin_centers.append(center)
                bin_fill_rates.append(rate)

        if len(bin_centers) < 2:
            return FillProbEstimate(
                A=DEFAULT_A, k=DEFAULT_K, n_samples=n_samples,
                r_squared=0.0, is_fallback=True,
            )

        # OLS: log(fill_rate) = log(A) - k·δ
        x = np.array(bin_centers)
        y = np.log(np.array(bin_fill_rates))

        # 加重なし OLS (y = a + b·x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))

        if ss_xx < 1e-20:
            return FillProbEstimate(
                A=DEFAULT_A, k=DEFAULT_K, n_samples=n_samples,
                r_squared=0.0, is_fallback=True,
            )

        b = ss_xy / ss_xx  # slope = -k
        a = y_mean - b * x_mean  # intercept = log(A)

        # R² 計算
        y_pred = a + b * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0

        # パラメータ抽出
        estimated_A = math.exp(a)
        estimated_k = -b  # slope = -k → k = -slope

        # サニティチェック
        if estimated_k < MIN_K:
            logger.info(
                f"推定 k={estimated_k:.2f} < {MIN_K}、"
                f"フロア適用"
            )
            estimated_k = MIN_K
        if estimated_k > MAX_K:
            logger.info(
                f"推定 k={estimated_k:.2f} > {MAX_K}、"
                f"キャップ適用"
            )
            estimated_k = MAX_K
        estimated_A = min(max(estimated_A, 0.01), 1.0)

        return FillProbEstimate(
            A=estimated_A,
            k=estimated_k,
            n_samples=n_samples,
            r_squared=r_squared,
        )

    except Exception:
        logger.exception("fill probability パラメータ推定に失敗")
        return FillProbEstimate(
            A=DEFAULT_A, k=DEFAULT_K, n_samples=n_samples,
            r_squared=0.0, is_fallback=True,
        )
