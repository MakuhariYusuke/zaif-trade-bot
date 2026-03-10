"""365# P3: SAC Sidecar 型定義 — directional_bias interface.

SAC の連続出力 [-1, +1] を方向性バイアスとして定義する型群。

設計根拠 (365# §2.2):
  SAC は BUY/SELL/HOLD の注文指示ではなく directional_bias を出力する。
  fill_test の offset 計算に非対称ブーストとして注入し、
  Asymmetric Maker として機能させる。

閾値設計 (365# §2.2):
  bias > +0.3  → BUY_BIAS  (買い方向に攻撃的)
  bias < -0.3  → SELL_BIAS (売り方向に攻撃的)
  |bias| ≤ 0.3 → NEUTRAL   (オフセット変更なし)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class SidecarDirection(IntEnum):
    """SAC sidecar の方向性分類."""

    BUY_BIAS = 1
    NEUTRAL = 0
    SELL_BIAS = -1


# ── 閾値定数 ──────────────────────────────────────────────
# 365# §2.2: ±0.3 で BUY/SELL 判定
BIAS_THRESHOLD: float = 0.3

# 365# §10.1: 初期ブースト値
# 0.1 bps は保守的すぎる可能性あり → 0.3 bps から開始
DEFAULT_SIDECAR_BOOST_BPS: float = 0.3

# シグナルの有効期限 (秒)
# 372# fix: retrain_interval=7200s なので TTL も合わせる。
# 600s では retrain 間の ~92% で stale 判定 → sidecar 事実上無効だった。
# retrain_interval + 10min buffer = 7800s に延長。
# scheduler crash 時は 2h10m で signal 自然失効 → safe fallback。
DEFAULT_SIGNAL_TTL_SEC: float = 7800.0


@dataclass(frozen=True, slots=True)
class SidecarSignal:
    """SAC sidecar のシグナル出力.

    365# §5.3 sidecar_signal.json に準拠。
    fill_test 側は ``directional_bias`` のみを使用。
    他のフィールドは診断・デバッグ用。
    """

    timestamp: str
    """ISO 8601 形式のタイムスタンプ (e.g. '2026-03-10T12:00:00+09:00')."""

    directional_bias: float
    """SAC Actor の連続出力 [-1.0 .. +1.0]."""

    model_version: str = ""
    """モデルバージョン識別子 (e.g. 'sac_sidecar_v460_20260310_1200')."""

    confidence: float = 1.0
    """推論の確信度 [0.0 .. 1.0]。将来的に boost 重み付けに使用。"""

    regime_hint: str = ""
    """SAC が認識した regime ヒント (診断用)."""

    features_snapshot: dict[str, float] = field(default_factory=dict)
    """推論時の特徴量スナップショット (診断用)."""

    training_metrics: dict[str, float] = field(default_factory=dict)
    """直近の訓練メトリクス (診断用)."""

    def __post_init__(self) -> None:
        """値域バリデーション."""
        if not (-1.0 <= self.directional_bias <= 1.0):
            raise ValueError(
                f"directional_bias must be in [-1.0, 1.0], "
                f"got {self.directional_bias}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], "
                f"got {self.confidence}"
            )

    @property
    def direction(self) -> SidecarDirection:
        """bias 値から方向性を分類."""
        return classify_bias(self.directional_bias)


def classify_bias(
    bias: float,
    threshold: float = BIAS_THRESHOLD,
) -> SidecarDirection:
    """directional_bias を BUY_BIAS / SELL_BIAS / NEUTRAL に分類.

    Args:
        bias: [-1.0, +1.0] の連続値
        threshold: 分類閾値 (デフォルト 0.3)

    Returns:
        SidecarDirection
    """
    if bias > threshold:
        return SidecarDirection.BUY_BIAS
    if bias < -threshold:
        return SidecarDirection.SELL_BIAS
    return SidecarDirection.NEUTRAL


def compute_sidecar_offset_bps(
    bias: float,
    side: str,
    boost_bps: float = DEFAULT_SIDECAR_BOOST_BPS,
    threshold: float = BIAS_THRESHOLD,
    confidence: float = 1.0,
) -> float:
    """side に対する sidecar オフセット調整量 (bps) を計算.

    365# §2.2 フロー:
      BUY_BIAS:  buy_offset  += boost,  sell_offset -= boost
      SELL_BIAS: sell_offset += boost,  buy_offset  -= boost
      NEUTRAL:   no change

    正のオフセット = より攻撃的 (mid に近い指値)
    負のオフセット = より保守的 (mid から離れた指値)

    Args:
        bias: directional_bias [-1.0, +1.0]
        side: 'buy' or 'sell'
        boost_bps: 基本ブースト値 (bps)
        threshold: 分類閾値
        confidence: 確信度 [0.0, 1.0] — boost に乗算

    Returns:
        オフセット調整量 (bps)。正=攻撃的、負=保守的。
    """
    direction = classify_bias(bias, threshold)
    if direction == SidecarDirection.NEUTRAL:
        return 0.0

    # confidence-weighted boost (365# §9.1: confidence-weighted で影響制限)
    effective_boost = boost_bps * min(max(confidence, 0.0), 1.0)

    if direction == SidecarDirection.BUY_BIAS:
        # BUY bias: buy を攻撃的に、sell を保守的に
        return effective_boost if side == "buy" else -effective_boost
    else:
        # SELL bias: sell を攻撃的に、buy を保守的に
        return effective_boost if side == "sell" else -effective_boost
