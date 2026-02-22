"""058# Skip Gate: PnL 予測ベースの注文スキップ判定.

Ridge PnL regressor を使って注文前に期待 PnL を予測し、
負のPnLが見込まれる注文をスキップする。

TODO(123# Gemini review): 取引コアロジック（ML判定）のため ztb/models/ への
  移動を推奨。ただし 20+ ファイル import 変更が伴うため fill_test 非稼働の
  タイミング (ph3-pre) で実施。→ 118# E11 として追跡。

Usage:
    from scripts.v460.ml.skip_gate import SkipGate

    gate = SkipGate.load("models/v460/skip_gate.pkl")
    should_skip, pred_pnl = gate.evaluate(features_dict)
"""

from __future__ import annotations

import json
import logging
import pickle
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("models/v460/skip_gate.pkl")

#: skip gate で使用する特徴量カラム (順序固定)
#: 071# OB 特徴量を除去 — 072# トグルで復元可能
_BASE_FEATURE_COLS: list[str] = [
    # base features
    "side_buy",
    "hour_sin",
    "hour_cos",
    "spread_jpy",
    "offset_ratio",
    "regime_trending",
    "regime_ranging",
    "regime_high_vol",
    # trade-based micro features
    "trade_count_60s",
    "buy_ratio",
    "trade_flow_imbalance_60s",
    "avg_trade_size",
    "price_velocity_60s",
    "vpin_60s",
    # interaction features (trade-based)
    "side_aligned_tfi",
    "side_aligned_velocity",
]

#: 072# OB 復元時に追加される特徴量
_OB_FEATURE_COLS: list[str] = [
    "spread_bps_ob",
    "depth_imbalance_ob",
    "side_aligned_imbalance",
]


def get_gate_feature_cols(use_ob: bool = False) -> list[str]:
    """072# SkipGate 特徴量カラムを取得.

    Args:
        use_ob: True で OB 特徴量 (spread_bps_ob, depth_imbalance_ob,
                side_aligned_imbalance) を含める.

    Returns:
        特徴量カラム名のリスト.
    """
    cols = list(_BASE_FEATURE_COLS)
    if use_ob:
        cols.extend(_OB_FEATURE_COLS)
    return cols


# 後方互換: デフォルトは OB なし (071# 状態)
GATE_FEATURE_COLS = get_gate_feature_cols(use_ob=False)


@dataclass
class SkipGateConfig:
    """Skip gate 設定."""

    threshold_bps: float = 0.0  # PnL 予測がこれ以下ならスキップ (mode=pnl)
    enabled: bool = True
    # 118# A3: side 別有効/無効 (sell 逆選別対策)
    buy_enabled: bool = True
    sell_enabled: bool = True
    max_skip_rate: float = 0.7  # 連続スキップ率の上限
    # 061#: AS 分類器モード
    mode: str = "pnl"  # "pnl" (Ridge PnL) or "as" (AS probability)
    as_threshold: float = 0.6  # AS 確率がこれ以上ならスキップ (mode=as)
    # 068# §3.3: side 別閾値 (None は共通 as_threshold を使用)
    as_threshold_buy: Optional[float] = None
    as_threshold_sell: Optional[float] = None
    # 072#: OB 特徴量トグル (ph2 通過後に True へ)
    use_ob_features: bool = False
    # 088# 動的閾値較正: 目標 skip 率ベースで閾値を自動調整
    adaptive_threshold: bool = False        # True で動的較正を有効化
    target_skip_rate_buy: float = 0.10      # buy 側の目標 skip 率
    target_skip_rate_sell: float = 0.20     # sell 側の目標 skip 率 (sell 劣後を補正)
    adaptive_window: int = 50               # 較正に使う直近サイクル数
    adaptive_min_samples: int = 20          # 較正開始に必要な最小サンプル数
    adaptive_step: float = 0.05             # 100# 1 回の閾値調整ステップ (0.02→0.05)
    adaptive_floor: float = 0.35            # 閾値の下限 (過剰 skip 防止)
    adaptive_ceiling: float = 0.80          # 閾値の上限 (skip 無効化防止)
    # 141# P1-04: regime 別 PnL 閾値オーバーライド (regime_name -> threshold_bps)
    # 例: {"high_vol": 0.2, "unknown": 0.3} — 指定なしのレジームは threshold_bps を使用
    regime_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class SkipDecision:
    """スキップ判定の結果."""

    should_skip: bool
    predicted_pnl_bps: float
    threshold_bps: float
    features_used: int
    reason: str = ""
    model_used: str = ""  # 068# §3.2: "primary" or "fallback"
    # 084# P(AS) 可観測性: 確率と使用閾値を直接記録
    as_probability: Optional[float] = None   # AS 分類器の P(AS) (0.0-1.0)
    threshold_used: Optional[float] = None   # 実際に適用された閾値 (side別含む)


class SkipGate:
    """予測ベースのスキップゲート.

    2つのモード:
    - mode="pnl": Ridge 回帰で post_fill_30s_pnl (bps) を予測し、
      予測値が閾値以下ならスキップ。
    - mode="as" (061#): AS 分類器 (LR) で P(adverse_selection) を予測し、
      確率が閾値以上ならスキップ。

    059# NEW-02: Pipeline (Imputer+Scaler+Model) を保持。
    後方互換: 旧形式 (model+scaler) でも動作。

    071# OB 特徴量を除去 — 価格ベースのみで判断 (v459 方式回帰).
    072# OB トグル追加 — use_ob_features=True で OB 特徴量を復元可能.
    """

    def __init__(
        self,
        model: Ridge,
        scaler: Any,
        feature_cols: list[str],
        config: SkipGateConfig | None = None,
        metadata: dict[str, Any] | None = None,
        pipeline: Pipeline | None = None,
        score_calibrator: Any | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.config = config or SkipGateConfig()
        self.metadata = metadata or {}
        # 100# P1-1: per-side skip 履歴 (cross-side 干渉を排除)
        self._recent_skips_buy: list[bool] = []
        self._recent_skips_sell: list[bool] = []
        self._pipeline = pipeline  # 059# NEW-02: 完全 Pipeline
        # 088# 動的較正用: side 別 P(AS) 履歴
        self._pas_history_buy: list[float] = []
        self._pas_history_sell: list[float] = []
        # 138# P1-03: score calibrator (isotonic regression)
        self._score_calibrator = score_calibrator

    def evaluate(
        self,
        features: dict[str, float],
        *,
        side: str | None = None,
        regime: str | None = None,
    ) -> SkipDecision:
        """注文前にスキップ判定.

        Args:
            features: 特徴量辞書. GATE_FEATURE_COLS のキーを含む.
            side: "buy" or "sell". 068# §3.3 side 別閾値に使用.
            regime: 141# P1-04 レジーム名. regime_thresholds に一致すればオーバーライド.

        Returns:
            SkipDecision.
        """
        if not self.config.enabled:
            return SkipDecision(
                should_skip=False,
                predicted_pnl_bps=0.0,
                threshold_bps=self.config.threshold_bps,
                features_used=0,
                reason="gate_disabled",
            )

        # 118# A3: side 別無効化 (sell 逆選別対策)
        if side == "buy" and not self.config.buy_enabled:
            return SkipDecision(
                should_skip=False, predicted_pnl_bps=0.0,
                threshold_bps=self.config.threshold_bps,
                features_used=0, reason="buy_gate_disabled",
            )
        if side == "sell" and not self.config.sell_enabled:
            return SkipDecision(
                should_skip=False, predicted_pnl_bps=0.0,
                threshold_bps=self.config.threshold_bps,
                features_used=0, reason="sell_gate_disabled",
            )

        # 059# NEW-04: 未提供特徴量は NaN (Pipeline の Imputer が処理)
        x = np.full(len(self.feature_cols), np.nan)
        n_used = 0
        for i, col in enumerate(self.feature_cols):
            if col in features:
                x[i] = features[col]
                n_used += 1

        if n_used < 3:
            return SkipDecision(
                should_skip=False,
                predicted_pnl_bps=0.0,
                threshold_bps=self.config.threshold_bps,
                features_used=n_used,
                reason="insufficient_features",
            )

        # 予測 — 059# NEW-02: Pipeline 優先、後方互換あり
        x_df = pd.DataFrame([x], columns=self.feature_cols)

        # 061#: mode に応じた予測・判定
        as_prob: Optional[float] = None  # 084# P(AS) 直接記録用
        threshold_used: Optional[float] = None  # 084# 使用閾値記録用
        if self.config.mode == "as":
            # AS 分類器モード: P(AS) を予測
            if self._pipeline is not None:
                pred_prob = float(self._pipeline.predict_proba(x_df)[0, 1])
            else:
                x_scaled = self.scaler.transform(x_df)
                pred_prob = float(self.model.predict_proba(x_scaled)[0, 1])
            pred_pnl = -pred_prob * 10  # AS確率→疑似PnL (互換表示用)
            as_prob = pred_prob  # 084# 生の確率を保持
            # 068# §3.3: side 別閾値
            threshold = self.config.as_threshold
            if side == "buy" and self.config.as_threshold_buy is not None:
                threshold = self.config.as_threshold_buy
            elif side == "sell" and self.config.as_threshold_sell is not None:
                threshold = self.config.as_threshold_sell
            threshold_used = threshold  # 084# side別解決後の閾値
            # 088# 動的較正: 目標 skip 率ベースで閾値を自動調整
            if self.config.adaptive_threshold and side is not None:
                threshold = self._calibrate_threshold(side, pred_prob, threshold)
                threshold_used = threshold  # 較正後の値で上書き
            should_skip = pred_prob >= threshold
        else:
            # PnL 回帰モード (既存)
            if self._pipeline is not None:
                pred_pnl = float(self._pipeline.predict(x_df)[0])
            else:
                x_scaled = self.scaler.transform(x_df)
                pred_pnl = float(self.model.predict(x_scaled)[0])
            # 138# P1-03: score calibration — isotonic regression で校正
            if self._score_calibrator is not None:
                cal = self._score_calibrator
                if hasattr(cal, "is_fitted") and cal.is_fitted:
                    pred_pnl = cal.calibrate(pred_pnl)
            # 141# P1-04: regime 別 PnL 閾値オーバーライド
            base_threshold = self.config.threshold_bps
            if regime and self.config.regime_thresholds:
                base_threshold = self.config.regime_thresholds.get(
                    regime, base_threshold
                )
            # 125# 動的較正: mode=pnl でも target_skip_rate にあわせて閾値を調整
            if self.config.adaptive_threshold and side is not None:
                threshold_used = self._calibrate_pnl_threshold(
                    side, pred_pnl, base_threshold=base_threshold,
                )
            else:
                threshold_used = base_threshold
            should_skip = pred_pnl < threshold_used

        # 100# P1-1: side 別連続スキップ率チェック (cross-side 干渉排除)
        side_skips = (
            self._recent_skips_buy if side == "buy"
            else self._recent_skips_sell if side == "sell"
            else None
        )
        if side_skips is not None:
            recent_rate = (
                sum(side_skips) / len(side_skips) if side_skips else 0.0
            )
        else:
            recent_rate = 0.0
        if recent_rate > self.config.max_skip_rate and should_skip:
            should_skip = False
            reason = f"skip_rate_limit({recent_rate:.0%}>{self.config.max_skip_rate:.0%})"
        else:
            reason = "skip" if should_skip else "pass"

        # 最終決定を side 別履歴に記録 (force-pass 反映後)
        if side_skips is not None:
            side_skips.append(should_skip)
            if len(side_skips) > 20:
                del side_skips[: len(side_skips) - 20]

        return SkipDecision(
            should_skip=should_skip,
            predicted_pnl_bps=pred_pnl,
            threshold_bps=self.config.threshold_bps,
            features_used=n_used,
            reason=reason,
            model_used="primary",
            as_probability=as_prob,
            threshold_used=threshold_used,
        )

    def _calibrate_threshold(
        self, side: str, current_prob: float, base_threshold: float,
    ) -> float:
        """088# 動的閾値較正: 目標 skip 率を達成するよう閾値を調整.

        直近 N 件の P(AS) 分布から、目標 skip 率に対応する分位点を推定し、
        base_threshold をステップ幅で近づける。

        Args:
            side: "buy" or "sell".
            current_prob: 今回の P(AS) 予測値 (履歴に追加).
            base_threshold: 静的設定の閾値 (初期値 / フォールバック).

        Returns:
            較正後の閾値.
        """
        cfg = self.config
        history = self._pas_history_buy if side == "buy" else self._pas_history_sell
        target_rate = (
            cfg.target_skip_rate_buy if side == "buy" else cfg.target_skip_rate_sell
        )

        # 履歴に追加
        history.append(current_prob)
        # ウィンドウ超過分を除去
        max_len = cfg.adaptive_window
        if len(history) > max_len:
            del history[: len(history) - max_len]

        # ウォームアップ: サンプル不足時は静的閾値を使用
        if len(history) < cfg.adaptive_min_samples:
            return base_threshold

        # 目標 skip 率に対応する P(AS) 分位点を算出
        # skip 率 = P(AS) >= threshold の割合 → threshold = (1 - target_rate) 分位点
        sorted_probs = sorted(history)
        quantile_idx = int(len(sorted_probs) * (1.0 - target_rate))
        quantile_idx = min(quantile_idx, len(sorted_probs) - 1)
        target_threshold = sorted_probs[quantile_idx]

        # ステップ幅で段階的に近づける (急激な変動防止)
        if base_threshold > target_threshold + cfg.adaptive_step:
            new_threshold = base_threshold - cfg.adaptive_step
        elif base_threshold < target_threshold - cfg.adaptive_step:
            new_threshold = base_threshold + cfg.adaptive_step
        else:
            new_threshold = target_threshold

        # クランプ: floor / ceiling で制約
        new_threshold = max(cfg.adaptive_floor, min(cfg.adaptive_ceiling, new_threshold))

        if abs(new_threshold - base_threshold) > 0.001:
            logger.debug(
                f"[skip_gate] 088# adaptive threshold: {side} "
                f"{base_threshold:.3f}→{new_threshold:.3f} "
                f"(target_skip={target_rate:.0%}, "
                f"quantile={target_threshold:.3f}, n={len(history)})"
            )

        # side 別閾値を更新 (次回 evaluate で反映)
        if side == "buy":
            self.config.as_threshold_buy = new_threshold
        else:
            self.config.as_threshold_sell = new_threshold

        return new_threshold

    # 125# PnL 回帰用の per-side 閾値キャッシュ
    _pnl_threshold_buy: float | None = None
    _pnl_threshold_sell: float | None = None
    # 125# PnL 回帰用の per-side 予測値履歴
    _pred_pnl_history_buy: list[float] | None = None
    _pred_pnl_history_sell: list[float] | None = None

    def _calibrate_pnl_threshold(
        self, side: str, current_pnl: float,
        base_threshold: float | None = None,
    ) -> float:
        """125# PnL 回帰用の動的閾値較正.

        mode=pnl で target_skip_rate を達成するよう threshold_bps を調整する。
        ロジック: 直近 N 件の予測 PnL 分布で target_skip_rate に対応する
        分位点 (下位) を閾値とする。

        Args:
            side: "buy" or "sell".
            current_pnl: 今回の予測 PnL (bps).
            base_threshold: 142# 起点閾値 (regime 別オーバーライド後の値)。
                None の場合は config.threshold_bps を使用。

        Returns:
            較正後の閾値 (bps). predicted_pnl < threshold → skip.
        """
        cfg = self.config
        if base_threshold is None:
            base_threshold = cfg.threshold_bps
        target_rate = (
            cfg.target_skip_rate_buy if side == "buy" else cfg.target_skip_rate_sell
        )

        # 遅延初期化 (pickleの後方互換性)
        if self._pred_pnl_history_buy is None:
            self._pred_pnl_history_buy = []
        if self._pred_pnl_history_sell is None:
            self._pred_pnl_history_sell = []
        if self._pnl_threshold_buy is None:
            self._pnl_threshold_buy = base_threshold
        if self._pnl_threshold_sell is None:
            self._pnl_threshold_sell = base_threshold

        history = (
            self._pred_pnl_history_buy if side == "buy"
            else self._pred_pnl_history_sell
        )

        # 履歴に追加
        history.append(current_pnl)
        max_len = cfg.adaptive_window
        if len(history) > max_len:
            del history[: len(history) - max_len]

        # ウォームアップ: サンプル不足時は静的閾値を使用
        if len(history) < cfg.adaptive_min_samples:
            return base_threshold

        # target_skip_rate に対応する PnL 分位点 (下位)
        # skip 率 = predicted_pnl < threshold → threshold = target_rate 分位点
        sorted_pnls = sorted(history)
        quantile_idx = int(len(sorted_pnls) * target_rate)
        quantile_idx = min(quantile_idx, len(sorted_pnls) - 1)
        target_threshold = sorted_pnls[quantile_idx]

        # 既存の side 閾値をベースに段階的に近づける
        current_th = (
            self._pnl_threshold_buy if side == "buy" else self._pnl_threshold_sell
        )
        if current_th is None:
            current_th = base_threshold

        step = cfg.adaptive_step  # bps 単位のステップ (PnL 空間)
        if current_th > target_threshold + step:
            new_threshold = current_th - step
        elif current_th < target_threshold - step:
            new_threshold = current_th + step
        else:
            new_threshold = target_threshold

        if abs(new_threshold - current_th) > 0.001:
            logger.debug(
                f"[skip_gate] 125# adaptive pnl threshold: {side} "
                f"{current_th:.3f}→{new_threshold:.3f} bps "
                f"(target_skip={target_rate:.0%}, "
                f"quantile={target_threshold:.3f}, n={len(history)})"
            )

        # side 別閾値を更新
        if side == "buy":
            self._pnl_threshold_buy = new_threshold
        else:
            self._pnl_threshold_sell = new_threshold

        return new_threshold

    def save(self, path: Optional[Path] = None) -> Path:
        """モデルを pickle 保存 (SHA256 ハッシュ付き)."""
        p = path or _DEFAULT_MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "config": self.config,
            "metadata": self.metadata,
            "pipeline": self._pipeline,  # 059# NEW-02
        }
        data = pickle.dumps(payload)
        digest = hashlib.sha256(data).hexdigest()
        with open(p, "wb") as f:
            f.write(data)
        # ハッシュファイル保存 — 059# P2-8
        hash_path = p.with_suffix(p.suffix + ".sha256")
        hash_path.write_text(digest)
        logger.info(f"SkipGate saved to {p} (sha256={digest[:12]}...)")
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SkipGate":
        """pickle からロード (ハッシュ検証付き)."""
        p = path or _DEFAULT_MODEL_PATH
        with open(p, "rb") as f:
            data = f.read()
        # 059# P2-8: ハッシュ検証
        hash_path = p.with_suffix(p.suffix + ".sha256")
        if hash_path.exists():
            expected = hash_path.read_text().strip()
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"SkipGate pickle hash mismatch: expected={expected[:12]}... "
                    f"actual={actual[:12]}... — ファイルが改竄された可能性"
                )
        payload = pickle.loads(data)  # noqa: S301
        gate = cls(
            model=payload["model"],
            scaler=payload["scaler"],
            feature_cols=payload["feature_cols"],
            config=payload.get("config", SkipGateConfig()),
            metadata=payload.get("metadata", {}),
            pipeline=payload.get("pipeline"),  # 059# NEW-02: 後方互換
        )
        logger.info(f"SkipGate loaded from {p} ({len(gate.feature_cols)} features)")
        return gate


def build_features_from_market_state(
    *,
    side: str,
    spread_jpy: float,
    offset_ratio: float,
    regime: str,
    recent_trades: list[dict] | None = None,
    trade_window_sec: int = 60,
    market_timestamp: float | None = None,
    # 072# OB トグル: ph2 通過後に有効化
    best_bid: float | None = None,
    best_ask: float | None = None,
    bid_vol_5: float | None = None,
    ask_vol_5: float | None = None,
    use_ob_features: bool = False,
) -> dict[str, float]:
    """現在のマーケット状態から skip gate 用特徴量を構築.

    071# OB 特徴量除去版: 価格と取引データのみで判断.
    072# OB トグル追加: use_ob_features=True + best_bid/ask/vol 指定で
         OB 特徴量 (spread_bps_ob, depth_imbalance_ob,
         side_aligned_imbalance) を追加生成.

    Args:
        side: "buy" or "sell".
        spread_jpy: 現在のスプレッド (JPY).
        offset_ratio: spread_offset_ratio.
        regime: "trending", "ranging", "high_vol", etc.
        recent_trades: 直近の約定リスト [{ts, price, amount, side}, ...].
        trade_window_sec: 約定統計のウィンドウ (秒). recent_trades をフィルタ.
        market_timestamp: UNIX timestamp (秒). 省略時は datetime.now() のローカル時刻.
        best_bid: 最良買気配 (JPY). OB 特徴量生成に使用.
        best_ask: 最良売気配 (JPY). OB 特徴量生成に使用.
        bid_vol_5: bid 側 depth 5 の合計数量. OB 特徴量生成に使用.
        ask_vol_5: ask 側 depth 5 の合計数量. OB 特徴量生成に使用.
        use_ob_features: True で OB 特徴量を生成.

    Returns:
        特徴量辞書. use_ob_features=False → 16 cols, True → 19 cols.
    """
    # 059# P1-5: 時刻特徴を明示的に制御
    if market_timestamp is not None:
        now = datetime.fromtimestamp(market_timestamp)
    else:
        now = datetime.now()
    hour = now.hour + now.minute / 60.0

    features: dict[str, float] = {}

    # Base features
    features["side_buy"] = 1.0 if side == "buy" else 0.0
    features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    features["spread_jpy"] = spread_jpy
    features["offset_ratio"] = offset_ratio
    features["regime_trending"] = 1.0 if regime == "trending" else 0.0
    features["regime_ranging"] = 1.0 if regime == "ranging" else 0.0
    features["regime_high_vol"] = 1.0 if regime == "high_vol" else 0.0

    # Trade features — 059# P1-6: trade_window_sec でフィルタ
    if recent_trades and market_timestamp is not None:
        cutoff = market_timestamp - trade_window_sec
        recent_trades = [
            t for t in recent_trades if t.get("ts", 0) >= cutoff
        ]
    if recent_trades:
        n_trades = len(recent_trades)
        buy_vol = sum(
            t.get("amount", 0)
            for t in recent_trades
            if t.get("side", "").lower() == "buy"
        )
        sell_vol = sum(
            t.get("amount", 0)
            for t in recent_trades
            if t.get("side", "").lower() != "buy"
        )
        total_vol = buy_vol + sell_vol

        features["trade_count_60s"] = float(n_trades)
        features["buy_ratio"] = buy_vol / total_vol if total_vol > 0 else 0.5
        features["trade_flow_imbalance_60s"] = (
            (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        )
        features["avg_trade_size"] = total_vol / n_trades if n_trades > 0 else 0.0

        # Price velocity
        if n_trades >= 2:
            first_p = recent_trades[0].get("price", 0)
            last_p = recent_trades[-1].get("price", 0)
            features["price_velocity_60s"] = (
                (last_p - first_p) / first_p * 10000 if first_p > 0 else 0.0
            )
        else:
            features["price_velocity_60s"] = 0.0

        features["vpin_60s"] = (
            abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.5
        )
    else:
        features["trade_count_60s"] = 0.0
        features["buy_ratio"] = 0.5
        features["trade_flow_imbalance_60s"] = 0.0
        features["avg_trade_size"] = 0.0
        features["price_velocity_60s"] = 0.0
        features["vpin_60s"] = 0.5

    # Interaction features (trade-based only, 071# OB 除去)
    side_sign = 1.0 if side == "buy" else -1.0
    features["side_aligned_tfi"] = (
        features["trade_flow_imbalance_60s"] * side_sign
    )
    features["side_aligned_velocity"] = (
        features["price_velocity_60s"] * side_sign
    )

    # 072# OB 特徴量 (トグル有効時のみ)
    if use_ob_features:
        if (
            best_bid is not None
            and best_ask is not None
            and best_ask > best_bid > 0
        ):
            mid = (best_bid + best_ask) / 2
            features["spread_bps_ob"] = (
                (best_ask - best_bid) / mid * 10_000
            )
        else:
            features["spread_bps_ob"] = float("nan")

        if (
            bid_vol_5 is not None
            and ask_vol_5 is not None
            and (bid_vol_5 + ask_vol_5) > 0
        ):
            depth_imb = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5)
            features["depth_imbalance_ob"] = depth_imb
        else:
            depth_imb = 0.0
            features["depth_imbalance_ob"] = float("nan")

        features["side_aligned_imbalance"] = depth_imb * side_sign

    return features


def warm_start_skip_gate_thresholds(
    gate: "SkipGate",
    fill_records_dir: Path | str,
    *,
    window: int = 50,
) -> None:
    """096# SkipGate 閾値較正の warm start: 直近レコードから P(AS) 履歴を復元.

    起動直後に adaptive_threshold が static fallback に固定される問題を解消。
    直近の fill records から skip_gate_as_prob を読み取り、
    _pas_history_buy / _pas_history_sell を初期化する。

    124# モデル世代検証: metadata.trained_at 以降のレコードのみ使用し、
    旧モデルの P(AS) が新モデルの P(target) 分布と混在するのを防止。
    モデル交代直後は新レコードが不足するため静的閾値から再較正される。

    Args:
        gate: 初期化対象の SkipGate.
        fill_records_dir: fill_records_*.jsonl のディレクトリ.
        window: 復元するサンプル数 (直近 N 件).
    """
    import glob

    # 124# モデル世代検証: trained_at 以前のレコードは旧モデル由来。
    # 旧 P(AS) を新 P(target) の adaptive calibrator に注入すると closed_form 誤較正。
    model_trained_ts: float | None = None
    trained_at_str = gate.metadata.get("trained_at", "")
    if trained_at_str:
        try:
            model_trained_ts = datetime.fromisoformat(str(trained_at_str)).timestamp()
        except (ValueError, TypeError):
            pass

    prob_records: list[tuple[str, float]] = []  # (side, p_as)
    files = sorted(glob.glob(str(Path(fill_records_dir) / "*.jsonl")))
    # 096# パフォーマンス: 必要な件数に達したら早期終了（最新ファイルから逆順読み）
    need = window * 2
    stale_skipped = 0
    for f in reversed(files):
        with open(f) as fh:
            file_records: list[tuple[str, float]] = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p_as = r.get("skip_gate_as_prob")
                side = r.get("side")
                if p_as is not None and side in ("buy", "sell"):
                    # 124# モデル世代フィルタ: trained_at 以前のレコードを除外
                    if model_trained_ts is not None:
                        rec_ts = r.get("timestamp")
                        if rec_ts is not None and float(rec_ts) < model_trained_ts:
                            stale_skipped += 1
                            continue
                    file_records.append((side, float(p_as)))
            prob_records = file_records + prob_records
        if len(prob_records) >= need:
            break

    if stale_skipped > 0:
        logger.info(
            f"[skip_gate] 124# warm_start: skipped {stale_skipped} stale records "
            f"from previous model (trained_at={trained_at_str}). "
            f"Usable records: {len(prob_records)}"
        )

    # 直近 window 件を復元
    recent = prob_records[-window * 2:] if len(prob_records) > window * 2 else prob_records
    buy_probs = [p for s, p in recent if s == "buy"][-window:]
    sell_probs = [p for s, p in recent if s == "sell"][-window:]

    gate._pas_history_buy = buy_probs
    gate._pas_history_sell = sell_probs

    # 100# P0-1: 復元した履歴から閾値を即座に較正
    # warm_start 前は YAML 初期値 (0.65 等) が使われるが、
    # P(AS) 分布 [0.42, 0.56] に対して高すぎて 6+ サイクル収束ラグが発生する。
    # 履歴が十分あれば _calibrate_threshold で閾値を事前設定する。
    if gate.config.adaptive_threshold:
        # 118# A2: warm_start で閾値を即座に収束させる
        # (旧: _calibrate_threshold 1 回 → adaptive_step しか進まず 6+ cycle ラグ)
        # 直接分位点を算出して閾値を即座設定 (_calibrate_threshold ループだと
        # 履歴に重複追加される副作用があるため、直接計算に変更)
        cfg = gate.config
        for side_name, probs, target_rate in [
            ("buy", buy_probs, cfg.target_skip_rate_buy),
            ("sell", sell_probs, cfg.target_skip_rate_sell),
        ]:
            if not probs or len(probs) < cfg.adaptive_min_samples:
                continue
            prev_th = (
                (cfg.as_threshold_buy if side_name == "buy" else cfg.as_threshold_sell)
                or cfg.as_threshold
            )
            sorted_probs = sorted(probs)
            q_idx = min(int(len(sorted_probs) * (1.0 - target_rate)),
                        len(sorted_probs) - 1)
            target_th = sorted_probs[q_idx]
            new_th = max(cfg.adaptive_floor, min(cfg.adaptive_ceiling, target_th))
            if side_name == "buy":
                cfg.as_threshold_buy = new_th
            else:
                cfg.as_threshold_sell = new_th
            logger.info(
                f"[skip_gate] 118# warm_start calibration: {side_name} threshold "
                f"{prev_th:.3f}→{new_th:.3f} (immediate, "
                f"quantile={target_th:.3f}, n={len(probs)})"
            )

    logger.info(
        f"[skip_gate] 096# warm_start: restored buy={len(buy_probs)}, "
        f"sell={len(sell_probs)} P(AS) samples for adaptive threshold"
    )


def train_and_save_skip_gate(
    results_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    *,
    alpha: float = 10.0,
    threshold_bps: float = 0.0,
) -> SkipGate:
    """fill records からスキップゲートモデルを学習・保存.

    Args:
        results_dir: fill_records_*.jsonl のディレクトリ.
        raw_dir: raw data ディレクトリ.
        output_path: 出力先 pkl.
        alpha: Ridge の正則化パラメータ.
        threshold_bps: スキップ閾値 (bps).

    Returns:
        学習済み SkipGate.
    """
    from scripts.v460.ml.data_loader import load_fill_records
    from scripts.v460.ml.feature_enricher import (
        build_pnl_features,
        enrich_fill_records,
    )

    df = load_fill_records(results_dir)
    enriched_df = enrich_fill_records(df, raw_dir=raw_dir)
    X, y = build_pnl_features(enriched_df)

    # 059# P0-1: Pipeline化
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
    pipe.fit(X, y.values)

    model = pipe.named_steps["model"]
    scaler = pipe.named_steps["scaler"]

    # Feature importances
    fi = dict(zip(X.columns.tolist(), np.abs(model.coef_).tolist()))
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    gate = SkipGate(
        model=model,
        scaler=scaler,
        feature_cols=X.columns.tolist(),
        config=SkipGateConfig(threshold_bps=threshold_bps),
        metadata={
            "n_samples": len(X),
            "mean_pnl_bps": float(y.mean()),
            "std_pnl_bps": float(y.std()),
            "alpha": alpha,
            "feature_importances": dict(sorted_fi),
            "trained_at": datetime.now().isoformat(),
        },
        pipeline=pipe,  # 059# NEW-02: 完全 Pipeline を保存
    )

    p = gate.save(output_path)
    logger.info(
        f"SkipGate trained: {len(X)} samples, "
        f"top features: {sorted_fi[:3]}"
    )
    return gate


def train_and_save_as_skip_gate(
    results_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    *,
    as_threshold: float = 0.6,
    k: int = 8,
) -> SkipGate:
    """061# AS 分類器ベースのスキップゲートを学習・保存.

    LR(C=0.01, l2, k=8) で P(adverse_selection) を予測し、
    確率が as_threshold 以上ならスキップを推奨する。

    Args:
        results_dir: fill_records_*.jsonl のディレクトリ.
        raw_dir: raw data ディレクトリ.
        output_path: 出力先 pkl.
        as_threshold: AS 確率のスキップ閾値.
        k: SelectKBest の k.

    Returns:
        学習済み SkipGate (mode=as).
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression

    from scripts.v460.ml.data_loader import load_fill_records
    from scripts.v460.ml.feature_enricher import (
        build_preorder_as_features,
        enrich_fill_records,
    )

    df = load_fill_records(results_dir)
    enriched_df = enrich_fill_records(df, raw_dir=raw_dir)
    # 096# feature contract 統一: 注文前に観測可能な特徴量のみで学習
    X, y = build_preorder_as_features(enriched_df)

    k_actual = min(k, X.shape[1])
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_classif, k=k_actual)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.01, max_iter=2000, class_weight="balanced", random_state=42,
        )),
    ])
    pipe.fit(X, y.values)

    model = pipe.named_steps["model"]
    scaler = pipe.named_steps["scaler"]

    # Feature importances (selected features)
    # SimpleImputer may drop all-NaN columns, so track surviving columns
    imputer = pipe.named_steps["imputer"]
    survived_mask = np.isfinite(imputer.statistics_)
    survived_cols = X.columns[survived_mask]
    selector = pipe.named_steps["selector"]
    selected_cols = survived_cols[selector.get_support()].tolist()
    fi = dict(zip(selected_cols, np.abs(model.coef_[0]).tolist()))
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    out_path = output_path or Path("models/v460/skip_gate_as.pkl")
    gate = SkipGate(
        model=model,
        scaler=scaler,
        feature_cols=X.columns.tolist(),
        config=SkipGateConfig(
            mode="as",
            as_threshold=as_threshold,
            threshold_bps=0.0,
        ),
        metadata={
            "n_samples": len(X),
            "as_rate": float(y.mean()),
            "k": k_actual,
            "selected_features": selected_cols,
            "feature_importances": dict(sorted_fi),
            "trained_at": datetime.now().isoformat(),
        },
        pipeline=pipe,
    )

    p = gate.save(out_path)
    logger.info(
        f"AS SkipGate trained: {len(X)} samples, k={k_actual}, "
        f"selected={selected_cols}, top features: {sorted_fi[:3]}"
    )
    return gate
