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

import math
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

# 365# §10.1 / 375#-376# 修正: 初期ブースト値
# 375#: max_boost_bps=3.0 は median spread 超過で自殺的
# 376#: 0.15 bps を絶対上限。ladder 検証 0.1/0.15/0.2
DEFAULT_SIDECAR_BOOST_BPS: float = 0.15

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
    """side に対する sidecar オフセット調整量 (bps) を計算 (v1: 離散分類方式).

    365# §2.2 フロー:
      BUY_BIAS:  buy_offset  += boost,  sell_offset -= boost
      SELL_BIAS: sell_offset += boost,  buy_offset  -= boost
      NEUTRAL:   no change

    正のオフセット = より攻撃的 (mid に近い指値)
    負のオフセット = より保守的 (mid から離れた指値)

    .. deprecated:: 374#
        v2 (compute_sidecar_offset_bps_v2) で置換予定。
        v1 は classify_bias() で [-1,+1] を 3 値に離散化するため
        SAC 連続出力の情報量を ~95% 損失する (375# §3 指摘)。

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


# ── 374# Phase 3.1: Proportional Boost ─────────────────────────
# 375#/376# 修正:
#   - max_boost_bps: 3.0→0.15 (3.0 は median spread 超過で自殺的)
#   - dead_zone: 0.10 (|bias|≤0.10 は noise → 0 出力)
#   - shaping: linear (初期), quadratic/sigmoid は α 検証後
#   - 377# ladder 検証: 0.10 → 0.15 → 0.20 bps step-up
# ────────────────────────────────────────────────────────────────

# dead zone デフォルト: |bias| ≤ 0.10 ではオフセット変更しない
DEFAULT_SIDECAR_DEAD_ZONE: float = 0.10

# shaping 関数の有効値
_VALID_SHAPING: frozenset[str] = frozenset({"linear", "quadratic", "sigmoid"})


def _shaping_fn(
    normalized: float,
    shaping: str,
) -> float:
    """正規化されたバイアス [0, 1] を shaping 関数で変換.

    Args:
        normalized: dead_zone 除去・正規化済みの |bias| [0.0 .. 1.0]
        shaping: "linear" | "quadratic" | "sigmoid"

    Returns:
        変換後の値 [0.0 .. ~1.0]
    """
    if shaping == "linear":
        return normalized
    if shaping == "quadratic":
        return normalized * normalized
    if shaping == "sigmoid":
        return math.tanh(3.0 * normalized)
    # unreachable if caller validates, but defensive
    return normalized


def compute_sidecar_offset_bps_v2(
    bias: float,
    side: str,
    max_boost_bps: float = DEFAULT_SIDECAR_BOOST_BPS,
    dead_zone: float = DEFAULT_SIDECAR_DEAD_ZONE,
    confidence: float = 1.0,
    shaping: str = "linear",
) -> float:
    """374# Phase 3.1: SAC 連続値を比例的にオフセットへ変換.

    v1 (compute_sidecar_offset_bps) は classify_bias() で [-1,+1] を
    BUY/SELL/NEUTRAL の 3 値に離散化し、全 BUY_BIAS に同一 boost を適用していた。
    → SAC 情報量の ~95% を損失。

    v2 は SAC 出力 [-1,+1] をそのまま比例計算に使用:
      f(b) = max_boost × shaping((|b| - dead_zone) / (1 - dead_zone)) × confidence
      sign = +1 if bias と side が同方向, -1 if 逆方向

    方向ルール (v1 と同一):
      bias > 0 (BUY方向):  buy → +offset (攻撃的), sell → -offset (保守的)
      bias < 0 (SELL方向): sell → +offset (攻撃的), buy → -offset (保守的)

    375#/376# 安全制約:
      - max_boost_bps ≤ 0.20 bps (hard ceiling, median spread 超過防止)
      - dead_zone = 0.10 (低確信ノイズ除去)
      - 377# ladder 検証: 0.10 → 0.15 → 0.20 bps で段階的に引き上げ

    Args:
        bias: directional_bias [-1.0, +1.0] — SAC Actor 出力
        side: 'buy' or 'sell'
        max_boost_bps: 最大ブースト (bps)。375# hard ceiling=0.20
        dead_zone: |bias| がこの値以下では 0.0 を返す [0.0, 1.0)
        confidence: 確信度 [0.0, 1.0] — boost に乗算
        shaping: 変換関数 "linear" | "quadratic" | "sigmoid"

    Returns:
        オフセット調整量 (bps)。正=攻撃的、負=保守的。

    Raises:
        ValueError: shaping が不正値の場合
    """
    # --- バリデーション ---
    if shaping not in _VALID_SHAPING:
        raise ValueError(
            f"shaping must be one of {sorted(_VALID_SHAPING)}, got '{shaping}'"
        )

    abs_bias = abs(bias)

    # dead zone: 低確信ノイズ除去
    if abs_bias <= dead_zone:
        return 0.0

    # 正規化: dead_zone ~ 1.0 → 0.0 ~ 1.0
    denominator = 1.0 - dead_zone
    if denominator <= 0.0:
        # dead_zone >= 1.0 は事実上 always-neutral
        return 0.0
    normalized = min((abs_bias - dead_zone) / denominator, 1.0)

    # shaping 適用
    shaped = _shaping_fn(normalized, shaping)

    # confidence & max_boost 適用
    clamped_confidence = min(max(confidence, 0.0), 1.0)
    magnitude = max_boost_bps * shaped * clamped_confidence

    # 方向決定: bias の符号 × side の一致で攻撃/保守を決定
    # bias > 0 (BUY方向): buy=攻撃的(+), sell=保守的(-)
    # bias < 0 (SELL方向): sell=攻撃的(+), buy=保守的(-)
    if bias > 0.0:
        return magnitude if side == "buy" else -magnitude
    else:
        return magnitude if side == "sell" else -magnitude
