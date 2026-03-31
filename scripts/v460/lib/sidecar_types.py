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

from collections.abc import Mapping
import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class SidecarDirection(IntEnum):
    """SAC sidecar の方向性分類."""

    BUY_BIAS = 1
    NEUTRAL = 0
    SELL_BIAS = -1


class PPOSidecarAction(StrEnum):
    """PPO sidecar の離散 side selection 出力."""

    BUY = "buy"
    SELL = "sell"
    SKIP = "skip"


# ── 閾値定数 ──────────────────────────────────────────────
# 365# §2.2: ±0.3 で BUY/SELL 判定
BIAS_THRESHOLD: float = 0.3

# 365# §10.1 / 375#-376# 修正: 初期ブースト値
# 375#: max_boost_bps=3.0 は median spread 超過で自殺的
# 376#: 0.15 bps を絶対上限。ladder 検証 0.1/0.15/0.2
DEFAULT_SIDECAR_BOOST_BPS: float = 0.15

# 675# PPO sidecar の side override 安定化閾値
DEFAULT_PPO_MIN_OVERRIDE_CONFIDENCE: float = 0.55
DEFAULT_PPO_MIN_ACTION_MARGIN: float = 0.10

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


_PPO_ACTION_NAMES: tuple[str, ...] = (
    PPOSidecarAction.BUY.value,
    PPOSidecarAction.SELL.value,
    PPOSidecarAction.SKIP.value,
)


def normalize_ppo_action_probabilities(
    probabilities: Mapping[str, float] | dict[str, float],
) -> dict[str, float]:
    """PPO sidecar の action probability dict を正規化する."""
    normalized: dict[str, float] = {}
    total = 0.0
    for action_name in _PPO_ACTION_NAMES:
        value = float(probabilities.get(action_name, 0.0))
        if value < 0.0:
            raise ValueError(
                f"action_probabilities[{action_name!r}] must be >= 0.0, got {value}"
            )
        normalized[action_name] = value
        total += value
    if total <= 0.0:
        raise ValueError("action_probabilities must contain a positive total weight")
    return {name: value / total for name, value in normalized.items()}


def resolve_ppo_sidecar_action(
    probabilities: Mapping[str, float] | dict[str, float],
) -> str:
    """最大確率 action から PPO sidecar の選択を返す."""
    normalized = normalize_ppo_action_probabilities(probabilities)
    return max(
        _PPO_ACTION_NAMES,
        key=lambda action_name: (
            normalized[action_name],
            -_PPO_ACTION_NAMES.index(action_name),
        ),
    )


@dataclass(frozen=True, slots=True)
class PPOSidecarSignal:
    """PPO sidecar の離散 side selection シグナル."""

    timestamp: str
    action: str
    action_probabilities: dict[str, float] = field(default_factory=dict)
    model_version: str = ""
    confidence: float = 1.0
    regime_hint: str = ""
    features_snapshot: dict[str, float] = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_action = str(self.action).lower()
        if normalized_action not in _PPO_ACTION_NAMES:
            raise ValueError(
                f"action must be one of {_PPO_ACTION_NAMES}, got {self.action!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        object.__setattr__(self, "action", normalized_action)
        normalized_probs = normalize_ppo_action_probabilities(self.action_probabilities)
        object.__setattr__(self, "action_probabilities", normalized_probs)

    @classmethod
    def from_probabilities(
        cls,
        *,
        timestamp: str,
        action_probabilities: Mapping[str, float] | dict[str, float],
        model_version: str = "",
        confidence: float | None = None,
        regime_hint: str = "",
        features_snapshot: dict[str, float] | None = None,
        training_metrics: dict[str, float] | None = None,
    ) -> "PPOSidecarSignal":
        """確率分布から action / confidence を補完して signal を組み立てる."""
        normalized_probs = normalize_ppo_action_probabilities(action_probabilities)
        resolved_action = resolve_ppo_sidecar_action(normalized_probs)
        resolved_confidence = (
            normalized_probs[resolved_action] if confidence is None else confidence
        )
        return cls(
            timestamp=timestamp,
            action=resolved_action,
            action_probabilities=normalized_probs,
            model_version=model_version,
            confidence=resolved_confidence,
            regime_hint=regime_hint,
            features_snapshot=features_snapshot or {},
            training_metrics=training_metrics or {},
        )

    @property
    def selected_side(self) -> str | None:
        """skip でなければ buy/sell を返す."""
        return None if self.action == PPOSidecarAction.SKIP.value else self.action

    @property
    def action_margin(self) -> float:
        """上位2 action の確率差. side override の安定度診断に使う."""
        ordered = sorted(self.action_probabilities.values(), reverse=True)
        if len(ordered) < 2:
            return 0.0
        return float(ordered[0] - ordered[1])


def should_activate_ppo_sidecar(
    signal: PPOSidecarSignal,
    *,
    min_confidence: float = DEFAULT_PPO_MIN_OVERRIDE_CONFIDENCE,
    min_action_margin: float = DEFAULT_PPO_MIN_ACTION_MARGIN,
) -> bool:
    """PPO sidecar を live gate に反映してよいか判定する.

    金融工学的には、top probability だけでなく 2位との差も見ないと
    microstructure noise による flip-flop override が増える。
    そのため confidence と action_margin の両方を通過条件にする。
    """
    return (
        signal.confidence >= min_confidence
        and signal.action_margin >= min_action_margin
    )


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
