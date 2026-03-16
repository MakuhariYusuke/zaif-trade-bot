"""141# P1-12: SkipGate オンラインパフォーマンスモニター.

直近 N fill のみを使って skip gate の判定品質を online で評価し、
全履歴平均への依存を排除する。

機能:
  - 直近 N fill で skip/pass グループの PnL 比較
  - skip 精度の rolling 評価 (skip した注文が実際に損失だったか)
  - degradation 検知: 直近の pass 平均 PnL が閾値以下ならアラート

Usage:
    from ztb.ml.online_monitor import OnlineMonitor, OnlineMonitorConfig

    monitor = OnlineMonitor(OnlineMonitorConfig(window=100))
    summary = monitor.evaluate(fill_records_df)
    if summary["degraded"]:
        logger.warning("Skip gate performance degraded")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

def _to_bool(s: pd.Series) -> pd.Series:
    """NaN → False の安全な bool 変換 (FutureWarning 回避)."""
    return s.fillna(False).infer_objects(copy=False).astype(bool)

@dataclass
class OnlineMonitorConfig:
    """オンラインモニター設定."""

    #: 評価に使う直近 fill 数 (window)
    window: int = 100
    #: pass 群の平均 PnL がこの値以下なら degraded 判定 (bps)
    degraded_threshold_bps: float = -0.3
    #: skip 精度の最低基準 (skip したうち実際に損失だった割合)
    min_skip_precision: float = 0.4
    #: 評価に必要な最小 fill 数
    min_samples: int = 20
    #: PnL カラム名 (post_fill 30s or 120s)
    pnl_column: str = "post_fill_30s_pnl"

@dataclass
class OnlineMonitorResult:
    """オンラインモニター評価結果."""

    #: 評価に使用した fill 数
    n_total: int = 0
    #: pass (skip_gate を通過) した fill 数
    n_passed: int = 0
    #: skip (skip_gate でスキップ) された fill 数
    n_skipped: int = 0
    #: pass 群の平均 PnL (bps)
    pass_mean_pnl: float = 0.0
    #: skip 群の平均予測 PnL (bps) — skip されたため実 PnL はないが score を使用
    skip_mean_score: float = 0.0
    #: skip 精度: skip したうち実際に score < 0 だった割合
    skip_precision: float = 0.0
    #: pass 群の win rate (PnL > 0 の割合)
    pass_win_rate: float = 0.0
    #: degraded 判定
    degraded: bool = False
    #: degraded 理由 (None = 正常)
    degraded_reason: str | None = None
    #: side 別サマリー
    side_summary: dict[str, dict[str, float]] | None = None

    def to_dict(self) -> dict:
        """テレメトリ/ログ用辞書変換."""
        d: dict = {
            "n_total": self.n_total,
            "n_passed": self.n_passed,
            "n_skipped": self.n_skipped,
            "pass_mean_pnl": round(self.pass_mean_pnl, 4),
            "skip_mean_score": round(self.skip_mean_score, 4),
            "skip_precision": round(self.skip_precision, 4),
            "pass_win_rate": round(self.pass_win_rate, 4),
            "degraded": self.degraded,
        }
        if self.degraded_reason:
            d["degraded_reason"] = self.degraded_reason
        if self.side_summary:
            d["side_summary"] = self.side_summary
        return d

class OnlineMonitor:
    """141# P1-12: 直近 N fill ベースのオンライン比較モニター.

    全履歴平均ではなく直近 window 件の fill のみで
    skip gate の判定品質を評価する。
    """

    def __init__(self, config: OnlineMonitorConfig | None = None) -> None:
        self._config = config or OnlineMonitorConfig()

    @property
    def config(self) -> OnlineMonitorConfig:
        return self._config

    def evaluate(self, records: pd.DataFrame) -> OnlineMonitorResult:
        """直近 N fill で skip gate パフォーマンスを評価.

        Args:
            records: fill_records DataFrame. 必須カラム:
                - skip_gate_skipped (bool)
                - filled (bool)
                - pnl_column (float, filled 行のみ)
                - skip_gate_score (float, skip 行の予測スコア)
                Optional:
                - side (str, "buy"/"sell")

        Returns:
            OnlineMonitorResult.
        """
        cfg = self._config
        result = OnlineMonitorResult()

        if records is None or len(records) == 0:
            return result

        # 143# A.1 #3: 評価対象レコードのみにフィルタ (unfilled 監査レコードを除外)
        # skip_gate_skipped=True (skip 判定) または filled=True (約定済み) のみ
        skip_col = "skip_gate_skipped"
        if skip_col in records.columns:
            skip_mask = _to_bool(records[skip_col])
            filled_mask = _to_bool(records.get(
                "filled", pd.Series(True, index=records.index)
            ))
            evaluable = records[skip_mask | filled_mask]
        else:
            evaluable = records

        # 直近 window 件に絞り込み
        recent = evaluable.tail(cfg.window).copy()
        result.n_total = len(recent)

        if result.n_total < cfg.min_samples:
            return result

        # skip/pass 分離
        if skip_col not in recent.columns:
            return result

        skip_mask = _to_bool(recent[skip_col])
        passed = recent[~skip_mask]
        skipped = recent[skip_mask]
        result.n_passed = len(passed)
        result.n_skipped = len(skipped)

        # pass 群: 実 PnL 分析 (filled のみ)
        pnl_col = cfg.pnl_column
        if pnl_col in passed.columns:
            filled_passed = passed[
                _to_bool(passed.get("filled", pd.Series(True, index=passed.index)))
            ]
            if len(filled_passed) > 0:
                pnl_vals = pd.to_numeric(filled_passed[pnl_col], errors="coerce").dropna()
                if len(pnl_vals) > 0:
                    result.pass_mean_pnl = float(pnl_vals.mean())
                    result.pass_win_rate = float((pnl_vals > 0).mean())

        # skip 群: 予測スコア分析
        score_col = "skip_gate_score"
        if score_col in skipped.columns and len(skipped) > 0:
            score_vals = pd.to_numeric(skipped[score_col], errors="coerce").dropna()
            if len(score_vals) > 0:
                result.skip_mean_score = float(score_vals.mean())
                # skip 精度: skip したうち score < 0 (損失予測) だった割合
                result.skip_precision = float((score_vals < 0).mean())

        # degradation 判定
        if result.n_passed >= cfg.min_samples:
            if result.pass_mean_pnl < cfg.degraded_threshold_bps:
                result.degraded = True
                result.degraded_reason = (
                    f"pass_mean_pnl={result.pass_mean_pnl:.3f}bps "
                    f"< threshold={cfg.degraded_threshold_bps}bps"
                )

        if (
            result.n_skipped >= cfg.min_samples
            and result.skip_precision < cfg.min_skip_precision
        ):
            reason = (
                f"skip_precision={result.skip_precision:.1%} "
                f"< min={cfg.min_skip_precision:.1%}"
            )
            if result.degraded:
                result.degraded_reason = f"{result.degraded_reason}; {reason}"
            else:
                result.degraded = True
                result.degraded_reason = reason

        # side 別サマリー
        if "side" in recent.columns:
            result.side_summary = {}
            for side_val in ("buy", "sell"):
                side_df = recent[recent["side"] == side_val]
                if len(side_df) == 0:
                    continue
                side_skip = side_df[_to_bool(side_df[skip_col])]
                side_pass = side_df[~_to_bool(side_df[skip_col])]
                side_info: dict[str, float] = {
                    "n_total": float(len(side_df)),
                    "n_passed": float(len(side_pass)),
                    "n_skipped": float(len(side_skip)),
                    "skip_rate": float(len(side_skip) / len(side_df)) if len(side_df) > 0 else 0.0,
                }
                # pass PnL
                if pnl_col in side_pass.columns and len(side_pass) > 0:
                    filled_side = side_pass[
                        _to_bool(side_pass.get("filled", pd.Series(True, index=side_pass.index)))
                    ]
                    if len(filled_side) > 0:
                        pnl_s = pd.to_numeric(filled_side[pnl_col], errors="coerce").dropna()
                        if len(pnl_s) > 0:
                            side_info["pass_mean_pnl"] = round(float(pnl_s.mean()), 4)
                            side_info["pass_win_rate"] = round(float((pnl_s > 0).mean()), 4)
                result.side_summary[side_val] = side_info

        return result

def log_online_monitor_summary(result: OnlineMonitorResult) -> None:
    """OnlineMonitorResult をログ出力."""
    if result.n_total == 0:
        logger.debug("[online_monitor] No fill records to evaluate")
        return

    level = logging.WARNING if result.degraded else logging.INFO
    msg = (
        f"[online_monitor] 141# P1-12: "
        f"n={result.n_total} (pass={result.n_passed}, skip={result.n_skipped}), "
        f"pass_mean_pnl={result.pass_mean_pnl:.3f}bps, "
        f"pass_win_rate={result.pass_win_rate:.1%}, "
        f"skip_precision={result.skip_precision:.1%}"
    )
    if result.degraded:
        msg += f" [DEGRADED: {result.degraded_reason}]"

    logger.log(level, msg)

    if result.side_summary:
        for side_name, info in result.side_summary.items():
            side_pnl = info.get("pass_mean_pnl", 0.0)
            side_wr = info.get("pass_win_rate", 0.0)
            logger.log(
                level,
                f"[online_monitor]   {side_name}: "
                f"n={info['n_total']:.0f} "
                f"(pass={info['n_passed']:.0f}, skip={info['n_skipped']:.0f}, "
                f"skip_rate={info['skip_rate']:.1%}), "
                f"pass_pnl={side_pnl:.3f}bps, "
                f"win_rate={side_wr:.1%}",
            )
