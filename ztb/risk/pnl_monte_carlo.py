"""
PnL モンテカルロシミュレータ — fill_test 実測データから月次 PnL 信頼区間を推定.

014# T5: 012# §3 #4「PnL モンテカルロ化」の実装.

実測した fill_records (JSONL) からスプレッド・adverse selection・fill rate の
経験分布を構築し、n=10,000 の月次シミュレーションで PnL レンジを推定する。

設計ポイント:
  - n が少なくても動作する (bootstrap resampling)
  - n が増えれば精度が自動的に向上・分布も安定する
  - 000# §3.3 の G1.1 判定指標メトリクスを同時出力
  - 時間帯別・レジーム別の層化サンプリングに拡張可能 (将来)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FillRecord:
    """fill_test 1 サイクルの結果."""
    cycle_id: str
    timestamp: float
    side: str
    order_price: float
    order_quantity: float
    fill_price: Optional[float]
    filled: bool
    cancelled: bool
    queue_wait_sec: float
    mid_at_fill: Optional[float]
    mid_30s_after: Optional[float]
    post_fill_30s_pnl: Optional[float]  # bps
    adverse_selected: Optional[bool]


@dataclass
class MonteCarloConfig:
    """シミュレーション設定."""
    n_simulations: int = 10_000
    cycles_per_day: int = 720      # 120s 間隔 = 1日 720 サイクル
    days_per_month: int = 30
    lot_size_btc: float = 0.001    # BTC per cycle
    btc_price_jpy: float = 10_300_000.0  # approx
    maker_fee_rate: float = 0.0    # Coincheck maker fee = 0%
    random_seed: int = 42
    confidence_levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class MonteCarloResult:
    """シミュレーション結果."""
    # Input summary
    n_records: int
    n_filled: int
    n_cancelled: int
    observed_fill_rate: float
    observed_pnl_mean_bps: float
    observed_pnl_std_bps: float
    observed_as_ratio: float

    # Simulation summary
    n_simulations: int
    cycles_per_month: int

    # Monthly PnL distribution (JPY)
    pnl_mean_jpy: float
    pnl_std_jpy: float
    pnl_percentiles_jpy: dict[str, float]  # "5%": xxx, "25%": xxx, ...

    # Monthly PnL distribution (bps, per-cycle average)
    pnl_mean_bps: float
    pnl_std_bps: float

    # Risk metrics
    var_95_jpy: float             # 95% VaR (neg = loss)
    cvar_95_jpy: float            # Conditional VaR
    prob_loss: float              # P(monthly PnL < 0)
    prob_profit: float            # P(monthly PnL > 0)

    # G1.1 criteria pass/fail (000# §3.3)
    g11_fill_rate: float
    g11_cancel_ratio: float
    g11_queue_wait_median: float
    g11_pnl_mean_bps: float
    g11_as_ratio: float
    g11_pass: bool

    # Break-even analysis
    breakeven_fill_rate: float    # fill rate needed for PnL ≥ 0
    breakeven_pnl_bps: float     # per-cycle bps needed for PnL ≥ 0

    raw_monthly_pnls: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class PnLMonteCarloSimulator:
    """fill_records から月次 PnL 分布を推定するモンテカルロシミュレータ.

    使用例::

        records = PnLMonteCarloSimulator.load_fill_records("results/v460/fill_test/")
        sim = PnLMonteCarloSimulator(records)
        result = sim.run()
        sim.print_report(result)
    """

    def __init__(
        self,
        records: Sequence[FillRecord],
        config: Optional[MonteCarloConfig] = None,
    ) -> None:
        if not records:
            raise ValueError("No fill records provided")
        self.records = list(records)
        self.config = config or MonteCarloConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_fill_records(
        path: str | Path,
        glob_pattern: str = "fill_records_*.jsonl",
    ) -> list[FillRecord]:
        """Load fill records from JSONL file(s) in a directory or single file."""
        p = Path(path)
        files: list[Path] = []
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(p.glob(glob_pattern))
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        records: list[FillRecord] = []
        for f in files:
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                d = json.loads(line)
                records.append(FillRecord(
                    cycle_id=d["cycle_id"],
                    timestamp=d["timestamp"],
                    side=d["side"],
                    order_price=d["order_price"],
                    order_quantity=d["order_quantity"],
                    fill_price=d.get("fill_price"),
                    filled=d["filled"],
                    cancelled=d["cancelled"],
                    queue_wait_sec=d["queue_wait_sec"],
                    mid_at_fill=d.get("mid_at_fill"),
                    mid_30s_after=d.get("mid_30s_after"),
                    post_fill_30s_pnl=d.get("post_fill_30s_pnl"),
                    adverse_selected=d.get("adverse_selected"),
                ))
        logger.info(f"Loaded {len(records)} fill records from {len(files)} file(s)")
        return records

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        cfg = self.config

        # --- Observed statistics ---
        filled = [r for r in self.records if r.filled]
        cancelled = [r for r in self.records if r.cancelled]
        n_total = len(self.records)
        n_filled = len(filled)
        n_cancelled = len(cancelled)
        fill_rate = n_filled / n_total if n_total > 0 else 0.0
        cancel_ratio = n_cancelled / n_total if n_total > 0 else 0.0

        # PnL distribution (bps) — only from filled orders
        pnl_bps = np.array([
            r.post_fill_30s_pnl for r in filled
            if r.post_fill_30s_pnl is not None
        ])
        if len(pnl_bps) == 0:
            pnl_bps = np.array([0.0])  # fallback for 0 fills

        pnl_mean = float(np.mean(pnl_bps))
        pnl_std = float(np.std(pnl_bps, ddof=1)) if len(pnl_bps) > 1 else 0.0

        # Adverse selection
        as_filled = [r for r in filled if r.adverse_selected is not None]
        as_count = sum(1 for r in as_filled if r.adverse_selected)
        as_ratio = as_count / len(as_filled) if as_filled else 0.0

        # Queue wait (filled only)
        waits = [r.queue_wait_sec for r in filled]
        queue_wait_median = float(np.median(waits)) if waits else 0.0

        # --- Monte Carlo ---
        cycles_per_month = cfg.cycles_per_day * cfg.days_per_month
        notional = cfg.lot_size_btc * cfg.btc_price_jpy  # JPY per cycle

        # Bootstrap: resample per-cycle PnL from observed distribution
        # Each simulation = 1 month of cycles
        monthly_pnls = np.zeros(cfg.n_simulations)

        for i in range(cfg.n_simulations):
            # 1. Sample fill/cancel outcomes (Bernoulli with observed fill_rate)
            fills_this_month = self._rng.binomial(cycles_per_month, fill_rate)

            # 2. Bootstrap sample PnL (bps) for each filled cycle
            if fills_this_month > 0:
                sampled_pnl = self._rng.choice(pnl_bps, size=fills_this_month, replace=True)
                # Convert bps → JPY: pnl_bps * 1e-4 * notional
                cycle_pnls_jpy = sampled_pnl * 1e-4 * notional
                monthly_pnls[i] = float(np.sum(cycle_pnls_jpy))

        # --- Percentiles ---
        percentiles: dict[str, float] = {}
        for p in cfg.confidence_levels:
            key = f"{p*100:.0f}%"
            percentiles[key] = float(np.percentile(monthly_pnls, p * 100))

        # VaR / CVaR
        var_95 = float(np.percentile(monthly_pnls, 5))  # 5th percentile (loss side)
        cvar_95 = float(np.mean(monthly_pnls[monthly_pnls <= var_95])) if np.any(monthly_pnls <= var_95) else var_95

        # P(loss) / P(profit)
        prob_loss = float(np.mean(monthly_pnls < 0))
        prob_profit = float(np.mean(monthly_pnls > 0))

        # --- Break-even analysis ---
        # What fill_rate would make expected PnL = 0?
        # E[monthly PnL] = cycles * fill_rate * E[pnl_bps] * 1e-4 * notional
        # = 0  when fill_rate = 0 or E[pnl_bps] = 0
        if pnl_mean != 0:
            # breakeven_fill_rate is trivial: any fill_rate gives E[PnL] > 0 if mean > 0
            # More useful: at what mean_bps do we break even at current fill_rate?
            breakeven_pnl_bps = 0.0  # always break even at 0 by definition
        else:
            breakeven_pnl_bps = 0.0

        # At current mean PnL, minimum fill_rate for positive expected PnL
        if pnl_mean > 0:
            breakeven_fill_rate = 0.0  # any positive fill_rate is profitable
        elif pnl_mean < 0:
            breakeven_fill_rate = 1.0  # can't break even with negative mean PnL
        else:
            breakeven_fill_rate = fill_rate

        # --- G1.1 pass/fail (000# §3.3) ---
        g11_pass = (
            fill_rate >= 0.90
            and cancel_ratio <= 0.30
            and queue_wait_median <= 60.0
            and pnl_mean >= 0.0
            and as_ratio <= 0.20
        )

        return MonteCarloResult(
            n_records=n_total,
            n_filled=n_filled,
            n_cancelled=n_cancelled,
            observed_fill_rate=fill_rate,
            observed_pnl_mean_bps=pnl_mean,
            observed_pnl_std_bps=pnl_std,
            observed_as_ratio=as_ratio,
            n_simulations=cfg.n_simulations,
            cycles_per_month=cycles_per_month,
            pnl_mean_jpy=float(np.mean(monthly_pnls)),
            pnl_std_jpy=float(np.std(monthly_pnls)),
            pnl_percentiles_jpy=percentiles,
            pnl_mean_bps=pnl_mean,
            pnl_std_bps=pnl_std,
            var_95_jpy=var_95,
            cvar_95_jpy=cvar_95,
            prob_loss=prob_loss,
            prob_profit=prob_profit,
            g11_fill_rate=fill_rate,
            g11_cancel_ratio=cancel_ratio,
            g11_queue_wait_median=queue_wait_median,
            g11_pnl_mean_bps=pnl_mean,
            g11_as_ratio=as_ratio,
            g11_pass=g11_pass,
            breakeven_fill_rate=breakeven_fill_rate,
            breakeven_pnl_bps=breakeven_pnl_bps,
            raw_monthly_pnls=monthly_pnls,
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        fill_rates: Optional[Sequence[float]] = None,
        pnl_adjustments_bps: Optional[Sequence[float]] = None,
    ) -> list[dict[str, Any]]:
        """fill_rate / pnl_mean を変えた感度分析.

        Args:
            fill_rates: テストする fill_rate の一覧 (default: 0.5~1.0)
            pnl_adjustments_bps: pnl に加算する値の一覧 (default: -2~+2)

        Returns:
            各シナリオの {fill_rate, pnl_adj, mean_jpy, var95_jpy, prob_loss} リスト
        """
        if fill_rates is None:
            fill_rates = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
        if pnl_adjustments_bps is None:
            pnl_adjustments_bps = [-2.0, -1.0, 0.0, 1.0, 2.0]

        cfg = self.config
        filled = [r for r in self.records if r.filled]
        pnl_bps = np.array([
            r.post_fill_30s_pnl for r in filled
            if r.post_fill_30s_pnl is not None
        ])
        if len(pnl_bps) == 0:
            pnl_bps = np.array([0.0])

        notional = cfg.lot_size_btc * cfg.btc_price_jpy
        cycles_per_month = cfg.cycles_per_day * cfg.days_per_month
        results: list[dict[str, Any]] = []

        for fr in fill_rates:
            for adj in pnl_adjustments_bps:
                adjusted_pnl = pnl_bps + adj
                monthly = np.zeros(cfg.n_simulations)
                for i in range(cfg.n_simulations):
                    fills = self._rng.binomial(cycles_per_month, fr)
                    if fills > 0:
                        sampled = self._rng.choice(adjusted_pnl, size=fills, replace=True)
                        monthly[i] = float(np.sum(sampled * 1e-4 * notional))

                results.append({
                    "fill_rate": fr,
                    "pnl_adj_bps": adj,
                    "mean_jpy": float(np.mean(monthly)),
                    "std_jpy": float(np.std(monthly)),
                    "var_95_jpy": float(np.percentile(monthly, 5)),
                    "prob_loss": float(np.mean(monthly < 0)),
                    "p50_jpy": float(np.median(monthly)),
                })

        return results

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self, result: MonteCarloResult) -> str:
        """Pretty-print report, also returns as string."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  PnL Monte Carlo Report (014# T5)")
        lines.append("=" * 60)
        lines.append("")

        # Observed data
        lines.append("▶ Observed Data")
        lines.append(f"  Records: {result.n_records} (filled={result.n_filled}, cancelled={result.n_cancelled})")
        lines.append(f"  Fill rate: {result.observed_fill_rate:.1%}")
        lines.append(f"  PnL mean:  {result.observed_pnl_mean_bps:+.3f} bps")
        lines.append(f"  PnL stdev: {result.observed_pnl_std_bps:.3f} bps")
        lines.append(f"  Adverse selection: {result.observed_as_ratio:.1%}")
        lines.append(f"  Queue wait median: {result.g11_queue_wait_median:.1f}s")
        lines.append("")

        # Monthly simulation
        lines.append(f"▶ Monthly Simulation ({result.n_simulations:,} paths, {result.cycles_per_month:,} cycles/mo)")
        lines.append(f"  E[PnL]:  {result.pnl_mean_jpy:+,.0f} JPY/mo")
        lines.append(f"  σ[PnL]:  {result.pnl_std_jpy:,.0f} JPY/mo")
        for k, v in result.pnl_percentiles_jpy.items():
            lines.append(f"    {k}: {v:+,.0f} JPY")
        lines.append("")

        # Risk
        lines.append("▶ Risk Metrics")
        lines.append(f"  VaR 95%:  {result.var_95_jpy:+,.0f} JPY")
        lines.append(f"  CVaR 95%: {result.cvar_95_jpy:+,.0f} JPY")
        lines.append(f"  P(loss):  {result.prob_loss:.1%}")
        lines.append(f"  P(profit): {result.prob_profit:.1%}")
        lines.append("")

        # G1.1
        lines.append("▶ G1.1 Criteria (000# §3.3)")
        g = "✅" if result.g11_fill_rate >= 0.90 else "❌"
        lines.append(f"  {g} fill_rate:    {result.g11_fill_rate:.1%} (≥90%)")
        g = "✅" if result.g11_cancel_ratio <= 0.30 else "❌"
        lines.append(f"  {g} cancel_ratio: {result.g11_cancel_ratio:.1%} (≤30%)")
        g = "✅" if result.g11_queue_wait_median <= 60.0 else "❌"
        lines.append(f"  {g} queue_wait:   {result.g11_queue_wait_median:.1f}s (≤60s)")
        g = "✅" if result.g11_pnl_mean_bps >= 0 else "❌"
        lines.append(f"  {g} pnl_mean:     {result.g11_pnl_mean_bps:+.3f} bps (≥0)")
        g = "✅" if result.g11_as_ratio <= 0.20 else "❌"
        lines.append(f"  {g} AS_ratio:     {result.g11_as_ratio:.1%} (≤20%)")
        verdict = "PASS" if result.g11_pass else "FAIL"
        lines.append(f"  → G1.1 = **{verdict}** (n={result.n_records}, 統計的意味は n≥200 で確定)")
        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)
        return report

    def to_dict(self, result: MonteCarloResult) -> dict[str, Any]:
        """Serialize result to dict (JSON-safe)."""
        d = {
            "n_records": result.n_records,
            "n_filled": result.n_filled,
            "n_cancelled": result.n_cancelled,
            "observed_fill_rate": result.observed_fill_rate,
            "observed_pnl_mean_bps": result.observed_pnl_mean_bps,
            "observed_pnl_std_bps": result.observed_pnl_std_bps,
            "observed_as_ratio": result.observed_as_ratio,
            "n_simulations": result.n_simulations,
            "cycles_per_month": result.cycles_per_month,
            "pnl_mean_jpy": result.pnl_mean_jpy,
            "pnl_std_jpy": result.pnl_std_jpy,
            "pnl_percentiles_jpy": result.pnl_percentiles_jpy,
            "pnl_mean_bps": result.pnl_mean_bps,
            "var_95_jpy": result.var_95_jpy,
            "cvar_95_jpy": result.cvar_95_jpy,
            "prob_loss": result.prob_loss,
            "prob_profit": result.prob_profit,
            "g11_fill_rate": result.g11_fill_rate,
            "g11_cancel_ratio": result.g11_cancel_ratio,
            "g11_queue_wait_median": result.g11_queue_wait_median,
            "g11_pnl_mean_bps": result.g11_pnl_mean_bps,
            "g11_as_ratio": result.g11_as_ratio,
            "g11_pass": result.g11_pass,
            "breakeven_fill_rate": result.breakeven_fill_rate,
        }
        return d
