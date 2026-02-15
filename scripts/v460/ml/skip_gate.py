"""058# Skip Gate: PnL 予測ベースの注文スキップ判定.

Ridge PnL regressor を使って注文前に期待 PnL を予測し、
負のPnLが見込まれる注文をスキップする。

Usage:
    from scripts.v460.ml.skip_gate import SkipGate

    gate = SkipGate.load("models/v460/skip_gate.pkl")
    should_skip, pred_pnl = gate.evaluate(features_dict)
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("models/v460/skip_gate.pkl")

#: skip gate で使用する特徴量カラム (順序固定)
GATE_FEATURE_COLS = [
    # base features
    "side_buy",
    "hour_sin",
    "hour_cos",
    "spread_jpy",
    "offset_ratio",
    "regime_trending",
    "regime_ranging",
    "regime_high_vol",
    # micro features
    "spread_bps_ob",
    "depth_imbalance_ob",
    "trade_count_60s",
    "buy_ratio",
    "trade_flow_imbalance_60s",
    "avg_trade_size",
    "price_velocity_60s",
    "vpin_60s",
    # interaction features
    "side_aligned_imbalance",
    "side_aligned_tfi",
    "side_aligned_velocity",
]


@dataclass
class SkipGateConfig:
    """Skip gate 設定."""

    threshold_bps: float = 0.0  # PnL 予測がこれ以下ならスキップ
    enabled: bool = True
    max_skip_rate: float = 0.7  # 連続スキップ率の上限


@dataclass
class SkipDecision:
    """スキップ判定の結果."""

    should_skip: bool
    predicted_pnl_bps: float
    threshold_bps: float
    features_used: int
    reason: str = ""


class SkipGate:
    """PnL 予測ベースのスキップゲート.

    Ridge 回帰で post_fill_30s_pnl (bps) を予測し、
    予測値が閾値以下ならスキップを推奨する。
    """

    def __init__(
        self,
        model: Ridge,
        scaler: StandardScaler,
        feature_cols: list[str],
        config: SkipGateConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.config = config or SkipGateConfig()
        self.metadata = metadata or {}
        self._recent_skips: list[bool] = []

    def evaluate(
        self,
        features: dict[str, float],
    ) -> SkipDecision:
        """注文前にスキップ判定.

        Args:
            features: 特徴量辞書. GATE_FEATURE_COLS のキーを含む.

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

        # 特徴量ベクトル構築
        x = np.zeros(len(self.feature_cols))
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

        # 予測
        import pandas as pd
        x_df = pd.DataFrame([x], columns=self.feature_cols)
        x_scaled = self.scaler.transform(x_df)
        pred_pnl = float(self.model.predict(x_scaled)[0])

        # スキップ判定
        should_skip = pred_pnl < self.config.threshold_bps

        # 連続スキップ率チェック
        self._recent_skips.append(should_skip)
        if len(self._recent_skips) > 20:
            self._recent_skips = self._recent_skips[-20:]
        recent_rate = sum(self._recent_skips) / len(self._recent_skips)
        if recent_rate > self.config.max_skip_rate and should_skip:
            should_skip = False
            reason = f"skip_rate_limit({recent_rate:.0%}>{self.config.max_skip_rate:.0%})"
        else:
            reason = "skip" if should_skip else "pass"

        return SkipDecision(
            should_skip=should_skip,
            predicted_pnl_bps=pred_pnl,
            threshold_bps=self.config.threshold_bps,
            features_used=n_used,
            reason=reason,
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """モデルを pickle 保存."""
        p = path or _DEFAULT_MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "config": self.config,
            "metadata": self.metadata,
        }
        with open(p, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"SkipGate saved to {p}")
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SkipGate":
        """pickle からロード."""
        p = path or _DEFAULT_MODEL_PATH
        with open(p, "rb") as f:
            payload = pickle.load(f)
        gate = cls(
            model=payload["model"],
            scaler=payload["scaler"],
            feature_cols=payload["feature_cols"],
            config=payload.get("config", SkipGateConfig()),
            metadata=payload.get("metadata", {}),
        )
        logger.info(f"SkipGate loaded from {p} ({len(gate.feature_cols)} features)")
        return gate


def build_features_from_market_state(
    *,
    side: str,
    spread_jpy: float,
    offset_ratio: float,
    regime: str,
    best_bid: float,
    best_ask: float,
    bid_vol_5: float,
    ask_vol_5: float,
    recent_trades: list[dict] | None = None,
    trade_window_sec: int = 60,
) -> dict[str, float]:
    """現在のマーケット状態から skip gate 用特徴量を構築.

    run_fill_test.py から呼び出される。

    Args:
        side: "buy" or "sell".
        spread_jpy: 現在のスプレッド (JPY).
        offset_ratio: spread_offset_ratio.
        regime: "trending", "ranging", "high_vol", etc.
        best_bid, best_ask: 最良気配.
        bid_vol_5, ask_vol_5: 上位5レベルの板厚.
        recent_trades: 直近の約定リスト [{ts, price, amount, side}, ...].
        trade_window_sec: 約定統計のウィンドウ (秒).

    Returns:
        GATE_FEATURE_COLS に対応する特徴量辞書.
    """
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

    # Micro features from orderbook
    mid = (best_bid + best_ask) / 2
    if mid > 0:
        features["spread_bps_ob"] = (best_ask - best_bid) / mid * 10000
    else:
        features["spread_bps_ob"] = 0.0

    total_depth = bid_vol_5 + ask_vol_5
    features["depth_imbalance_ob"] = (
        (bid_vol_5 - ask_vol_5) / total_depth if total_depth > 0 else 0.0
    )

    # Trade features
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

    # Interaction features
    side_sign = 1.0 if side == "buy" else -1.0
    features["side_aligned_imbalance"] = (
        features["depth_imbalance_ob"] * side_sign
    )
    features["side_aligned_tfi"] = (
        features["trade_flow_imbalance_60s"] * side_sign
    )
    features["side_aligned_velocity"] = (
        features["price_velocity_60s"] * side_sign
    )

    return features


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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y.values)

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
    )

    p = gate.save(output_path)
    logger.info(
        f"SkipGate trained: {len(X)} samples, "
        f"top features: {sorted_fi[:3]}"
    )
    return gate
